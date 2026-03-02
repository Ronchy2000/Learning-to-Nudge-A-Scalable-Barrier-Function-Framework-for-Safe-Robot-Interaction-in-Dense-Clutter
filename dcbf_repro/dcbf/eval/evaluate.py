from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from dcbf.envs.isaaclab_env import EnvConfig, PandaClutterEnv
from dcbf.models.dcbf_net import DCBFNet
from dcbf.safety.compose import LearnedGlobalBarrier
from dcbf.safety.filter import SamplingSafetyFilter, nominal_apf, nominal_go_to_goal
from dcbf.utils.geometry import ObservationHistoryBuffer
from dcbf.utils.io import dump_json, load_yaml
from dcbf.utils.seeding import set_seed


def load_model_and_filter(checkpoint_path: str, filter_cfg: Dict, max_step: float):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    optim_cfg = ckpt["config"]["optim"]
    model = DCBFNet(
        robot_dim=model_cfg["robot_dim"],
        object_dim=model_cfg["object_dim"],
        history_len=model_cfg["history_len"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        mlp_hidden=model_cfg["mlp_hidden"],
    )
    model.load_state_dict(ckpt["model_state"])
    device = optim_cfg.get("device", "cpu")
    model.to(device).eval()
    barrier = LearnedGlobalBarrier(model=model, device=device, top_m_objects=filter_cfg.get("top_m_objects"))
    filt = SamplingSafetyFilter(
        barrier=barrier,
        max_step=max_step,
        num_candidates=int(filter_cfg.get("num_candidates", 64)),
        noise_std=float(filter_cfg.get("noise_std", 0.003)),
        fallback_scale=float(filter_cfg.get("fallback_scale", 0.2)),
        seed=0,
    )
    return model_cfg["history_len"], filt


def run_episode(
    env: PandaClutterEnv,
    method: str,
    history_len: int,
    safety_filter: Optional[SamplingSafetyFilter],
    seed: int,
) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    hist = ObservationHistoryBuffer(history_len)
    hist.pad_with(obs)

    success = False
    violation = False
    stall = False
    steps = 0
    for step in range(env.cfg.max_episode_steps):
        steps = step + 1
        if method == "do_nothing":
            action = nominal_go_to_goal(obs, env.cfg.max_action_step)
        elif method == "apf":
            action = nominal_apf(obs, env.cfg.max_action_step)
        elif safety_filter is not None:
            u_nom = nominal_go_to_goal(obs, env.cfg.max_action_step)
            action, _ = safety_filter.step(obs, u_nom, history=hist.view())
        else:
            raise ValueError(f"Unsupported method: {method}")

        next_obs, reward, terminated, truncated, info = env.step(action)
        _ = reward
        hist.push(next_obs)
        obs = next_obs
        success = success or bool(info.get("success", False))
        violation = violation or bool(info.get("violation", False))
        stall = stall or bool(info.get("stall", False))
        if terminated or truncated:
            break

    success_no_violation = bool(success and not violation)
    return {
        "success": float(success_no_violation),
        "violation": float(violation),
        "stall": float(stall),
        "steps": float(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCBF and baselines on clutter settings.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="Override methods list.")
    parser.add_argument("--num_objects_list", type=int, nargs="+", default=None, help="Override clutter sizes.")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes_per_setting.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--initial_checkpoint", type=str, default=None, help="Override initial DCBF checkpoint.")
    parser.add_argument("--refined_checkpoint", type=str, default=None, help="Override refined DCBF checkpoint.")
    parser.add_argument(
        "--learned_method",
        type=str,
        action="append",
        default=None,
        help="Extra learned method mapping: name=checkpoint_path. Repeatable.",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.methods is not None:
        cfg["methods"] = args.methods
    if args.num_objects_list is not None:
        cfg["num_objects_list"] = [int(v) for v in args.num_objects_list]
    if args.episodes is not None:
        cfg["episodes_per_setting"] = int(args.episodes)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.initial_checkpoint is not None:
        cfg["initial_checkpoint"] = args.initial_checkpoint
    if args.refined_checkpoint is not None:
        cfg["refined_checkpoint"] = args.refined_checkpoint

    learned_methods = dict(cfg.get("learned_methods", {}))
    if "initial_checkpoint" in cfg:
        learned_methods.setdefault("initial_dcbf", cfg["initial_checkpoint"])
    if "refined_checkpoint" in cfg:
        learned_methods.setdefault("refined_dcbf", cfg["refined_checkpoint"])
    if args.learned_method:
        for pair in args.learned_method:
            if "=" not in pair:
                raise ValueError(f"Invalid --learned_method '{pair}', expected name=checkpoint_path")
            name, ckpt = pair.split("=", 1)
            learned_methods[name.strip()] = ckpt.strip()

    set_seed(cfg.get("seed", 7))

    env_cfg_yaml = load_yaml(cfg["env_config"])
    base_env_cfg = EnvConfig.from_dict(env_cfg_yaml["env"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] output_dir={out_dir}")
    print(f"[eval] learned_methods={learned_methods}")

    methods: List[str] = cfg["methods"]
    num_objects_list: List[int] = cfg["num_objects_list"]
    episodes_per_setting: int = int(cfg["episodes_per_setting"])

    filters = {}
    history_lens = {}
    filter_cfg = cfg["safety_filter"]
    for method in methods:
        if method in learned_methods:
            ckpt_path = learned_methods[method]
            if not Path(ckpt_path).exists():
                print(f"[eval] warning: checkpoint not found for {method}: {ckpt_path}, skip this method.")
                continue
            hist_len, filt = load_model_and_filter(ckpt_path, filter_cfg=filter_cfg, max_step=base_env_cfg.max_action_step)
            filters[method] = filt
            history_lens[method] = hist_len
        elif method in {"do_nothing", "apf"}:
            history_lens[method] = 10
            filters[method] = None
        else:
            print(f"[eval] warning: unknown method={method}, skip.")

    episode_rows = []
    summary_rows = []

    for method in methods:
        if method not in filters:
            continue
        for n_obj in num_objects_list:
            env_cfg = EnvConfig.from_dict(env_cfg_yaml["env"])
            env_cfg.num_objects = int(n_obj)
            env = PandaClutterEnv(env_cfg)
            metrics = {"success": [], "violation": [], "stall": [], "steps": []}
            for epi in tqdm(range(episodes_per_setting), desc=f"{method}-N{n_obj}"):
                ep_metric = run_episode(
                    env=env,
                    method=method,
                    history_len=history_lens[method],
                    safety_filter=filters[method],
                    seed=cfg["seed"] + epi + 1000 * n_obj,
                )
                for key in metrics:
                    metrics[key].append(ep_metric[key])
                episode_rows.append(
                    {
                        "method": method,
                        "num_objects": n_obj,
                        "episode_idx": epi,
                        **ep_metric,
                    }
                )
            summary_rows.append(
                {
                    "method": method,
                    "num_objects": n_obj,
                    "success_rate": float(np.mean(metrics["success"])),
                    "violation_rate": float(np.mean(metrics["violation"])),
                    "stalling_rate": float(np.mean(metrics["stall"])),
                    "avg_episode_steps": float(np.mean(metrics["steps"])),
                }
            )

    episode_csv = out_dir / "episodes.csv"
    summary_csv = out_dir / "metrics.csv"

    if episode_rows:
        with episode_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    dump_json({"summary_rows": summary_rows}, out_dir / "metrics.json")
    print(f"[eval] summary csv: {summary_csv}")


if __name__ == "__main__":
    main()
