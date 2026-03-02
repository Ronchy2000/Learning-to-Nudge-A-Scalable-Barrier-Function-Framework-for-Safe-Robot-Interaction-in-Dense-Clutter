from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from dcbf.data.collect import ShardWriter
from dcbf.data.labeling import next_state_labels
from dcbf.envs.isaaclab_env import EnvConfig, PandaClutterEnv
from dcbf.models.dcbf_net import DCBFNet
from dcbf.safety.compose import LearnedGlobalBarrier
from dcbf.training.train import train_model
from dcbf.utils.geometry import ObservationHistoryBuffer, object_centric_transform
from dcbf.utils.io import dump_json, load_yaml
from dcbf.utils.seeding import set_seed


def load_model_from_checkpoint(checkpoint: str, train_cfg: Dict, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    model_cfg = train_cfg["model"]
    model = DCBFNet(
        robot_dim=model_cfg["robot_dim"],
        object_dim=model_cfg["object_dim"],
        history_len=model_cfg["history_len"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        mlp_hidden=model_cfg["mlp_hidden"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _predict_barrier(model, robot_tp1: np.ndarray, obj_hist_curr: np.ndarray, device: str, batch_size: int = 1024):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, robot_tp1.shape[0], batch_size):
            end = min(start + batch_size, robot_tp1.shape[0])
            robot_t = torch.from_numpy(robot_tp1[start:end]).to(device=device, dtype=torch.float32)
            obj_t = torch.from_numpy(obj_hist_curr[start:end]).to(device=device, dtype=torch.float32)
            b = model(robot_t, obj_t).squeeze(-1).cpu().numpy()
            outputs.append(b)
    return np.concatenate(outputs, axis=0)


def select_near_boundary_states(
    dataset_files: List[str],
    model,
    delta: float,
    max_states: int,
    device: str,
) -> List[Dict]:
    selected = []
    selected_keys = set()
    for file in tqdm(dataset_files, desc="Selecting near-boundary"):
        data = np.load(file)
        b_vals = _predict_barrier(model, data["robot_tp1"], data["obj_hist_curr"], device=device)
        near_idx = np.where(np.abs(b_vals) <= delta)[0]
        for idx in near_idx.tolist():
            key = (
                int(data["scene_seed"][idx]),
                int(data["episode_idx"][idx]),
                int(data["step_idx"][idx]),
            )
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(
                {
                    "scene_seed": key[0],
                    "episode_idx": key[1],
                    "step_idx": key[2],
                    "snap_ee": data["snap_ee"][idx],
                    "snap_goal": data["snap_goal"][idx],
                    "snap_object_pos": data["snap_object_pos"][idx],
                    "snap_object_tilt_rad": data["snap_object_tilt_rad"][idx],
                    "snap_step_count": int(data["snap_step_count"][idx]),
                    "obj_index": int(data["obj_index"][idx]),
                    "pred_b": float(b_vals[idx]),
                }
            )
            if len(selected) >= max_states:
                return selected
    return selected


def choose_safest_action(
    obs: Dict[str, np.ndarray],
    history,
    barrier: LearnedGlobalBarrier,
    max_step: float,
    candidate_actions: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    candidates = rng.uniform(low=-max_step, high=max_step, size=(candidate_actions, 2)).astype(np.float32)
    goal_dir = np.asarray(obs["goal_xy"], dtype=np.float32) - np.asarray(obs["ee_pos"], dtype=np.float32)[:2]
    candidates[0] = goal_dir
    scores = np.array([barrier.score_action(obs, history, act) for act in candidates], dtype=np.float32)
    idx = int(np.argmax(scores))
    return candidates[idx], float(scores[idx])


def rollout_refinement_data(
    states: List[Dict],
    env: PandaClutterEnv,
    model,
    train_cfg: Dict,
    refine_cfg: Dict,
    output_dir: Path,
) -> List[str]:
    model_cfg = train_cfg["model"]
    env_cfg = env.cfg
    writer = ShardWriter(output_dir / "refined_data", prefix="refined", shard_size=12000)
    barrier = LearnedGlobalBarrier(
        model=model,
        device=train_cfg["optim"].get("device", "cpu"),
        top_m_objects=refine_cfg.get("top_m_objects"),
    )
    rng = np.random.default_rng(refine_cfg.get("seed", 123))

    for state in tqdm(states, desc="Refinement rollouts"):
        snap = {
            "ee_pos": state["snap_ee"],
            "goal": state["snap_goal"],
            "object_pos": state["snap_object_pos"],
            "object_tilt_rad": state["snap_object_tilt_rad"],
            "step_count": state["snap_step_count"],
            "seed": state["scene_seed"],
        }
        env.restore_snapshot(snap)
        obs = env.get_obs()
        hist = ObservationHistoryBuffer(model_cfg["history_len"])
        hist.pad_with(obs)

        for local_step in range(int(refine_cfg["rollout_steps"])):
            hist_prev = hist.view()
            best_action, best_b = choose_safest_action(
                obs=obs,
                history=hist_prev,
                barrier=barrier,
                max_step=env_cfg.max_action_step,
                candidate_actions=int(refine_cfg["candidate_actions"]),
                rng=rng,
            )
            next_obs, reward, terminated, truncated, info = env.step(best_action)
            _ = (reward, info)
            hist.push(next_obs)
            hist_curr = hist.view()
            obj_safe_labels, global_safe = next_state_labels(next_obs, threshold_deg=env_cfg.tilt_threshold_deg)
            next_tilt_deg = np.rad2deg(next_obs["objects_tilt_rad"])

            for obj_idx in range(env_cfg.num_objects):
                anchor = hist_prev.object_hist[0, obj_idx, :3]
                r_t_i, o_prev_i = object_centric_transform(
                    obs["ee_pos"], hist_prev.object_hist, object_index=obj_idx, anchor_xyz=anchor
                )
                r_tp1_i, o_curr_i = object_centric_transform(
                    next_obs["ee_pos"], hist_curr.object_hist, object_index=obj_idx, anchor_xyz=anchor
                )
                sample = {
                    "robot_t": r_t_i,
                    "robot_tp1": r_tp1_i,
                    "obj_hist_prev": o_prev_i,
                    "obj_hist_curr": o_curr_i,
                    "label_safe_obj": np.array(obj_safe_labels[obj_idx], dtype=np.float32),
                    "label_safe_global": np.array(global_safe, dtype=np.float32),
                    "next_tilt_deg": np.array(next_tilt_deg[obj_idx], dtype=np.float32),
                    "next_max_tilt_deg": np.array(np.max(next_tilt_deg), dtype=np.float32),
                    "obj_index": np.array(obj_idx, dtype=np.int32),
                    "episode_idx": np.array(state["episode_idx"], dtype=np.int32),
                    "step_idx": np.array(state["step_idx"] + local_step, dtype=np.int32),
                    "scene_seed": np.array(state["scene_seed"], dtype=np.int32),
                    "action_xy": np.asarray(best_action, dtype=np.float32),
                    "selected_b_global": np.array(best_b, dtype=np.float32),
                    "snap_ee": np.asarray(obs["ee_pos"], dtype=np.float32),
                    "snap_goal": np.asarray(obs["goal_xy"], dtype=np.float32),
                    "snap_object_pos": np.asarray(obs["objects_pos"], dtype=np.float32),
                    "snap_object_tilt_rad": np.asarray(obs["objects_tilt_rad"], dtype=np.float32),
                    "snap_step_count": np.array(state["snap_step_count"] + local_step, dtype=np.int32),
                }
                writer.append(sample)

            obs = next_obs
            if terminated or truncated:
                break
    writer.flush()
    return writer.saved_files


def main() -> None:
    parser = argparse.ArgumentParser(description="DCBF refinement: near-boundary selection + safest-action rollout + finetune.")
    parser.add_argument("--config", type=str, default="configs/refine.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 123))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_yaml(cfg["env_config"])
    train_cfg = load_yaml(cfg["train_config"])
    device = train_cfg["optim"].get("device", "cpu")

    env = PandaClutterEnv(EnvConfig.from_dict(env_cfg["env"]))
    model = load_model_from_checkpoint(cfg["checkpoint"], train_cfg, device=device)
    dataset_files = sorted(glob.glob(cfg["dataset_glob"]))
    if len(dataset_files) == 0:
        raise FileNotFoundError(f"No dataset files found: {cfg['dataset_glob']}")

    selected = select_near_boundary_states(
        dataset_files=dataset_files,
        model=model,
        delta=float(cfg["near_boundary_delta"]),
        max_states=int(cfg["max_refine_states"]),
        device=device,
    )
    refined_files = rollout_refinement_data(
        states=selected,
        env=env,
        model=model,
        train_cfg=train_cfg,
        refine_cfg=cfg,
        output_dir=out_dir,
    )

    original_train_files = dataset_files
    val_glob = train_cfg["data"].get("val_glob")
    if val_glob:
        val_files = sorted(glob.glob(val_glob))
    else:
        val_files = original_train_files
    combined_train_files = original_train_files + refined_files

    ft_cfg = dict(train_cfg)
    ft_cfg["optim"] = dict(train_cfg["optim"])
    ft_cfg["optim"]["epochs"] = int(cfg["finetune"]["epochs"])
    ft_cfg["optim"]["lr"] = float(cfg["finetune"]["lr"])
    ft_cfg["logging"] = dict(train_cfg["logging"])
    ft_cfg["logging"]["out_dir"] = str(out_dir)
    ft_cfg["logging"]["run_name"] = cfg["finetune"]["run_name"]

    best_ckpt = train_model(
        cfg=ft_cfg,
        train_files=combined_train_files,
        val_files=val_files,
        resume=cfg["checkpoint"],
        out_dir=str(out_dir),
        run_name=cfg["finetune"]["run_name"],
        override_epochs=int(cfg["finetune"]["epochs"]),
        override_lr=float(cfg["finetune"]["lr"]),
    )

    summary = {
        "selected_near_boundary_states": len(selected),
        "refined_files": refined_files,
        "combined_train_file_count": len(combined_train_files),
        "best_refined_checkpoint": str(best_ckpt),
    }
    dump_json(summary, out_dir / "refine_summary.json")
    print(f"[refine] done. summary: {out_dir / 'refine_summary.json'}")


if __name__ == "__main__":
    main()
