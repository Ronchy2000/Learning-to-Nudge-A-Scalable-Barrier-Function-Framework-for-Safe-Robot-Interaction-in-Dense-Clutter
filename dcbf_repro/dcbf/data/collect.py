from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from dcbf.data.labeling import next_state_labels
from dcbf.envs.isaaclab_env import EnvConfig, PandaClutterEnv
from dcbf.safety.compose import ToyDistanceBarrier
from dcbf.safety.filter import SamplingSafetyFilter, nominal_apf, nominal_go_to_goal
from dcbf.utils.geometry import ObservationHistoryBuffer, object_centric_transform
from dcbf.utils.io import dump_json, load_yaml
from dcbf.utils.seeding import set_seed


def _as_array(v: Any) -> np.ndarray:
    arr = np.asarray(v)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


@dataclass
class ShardWriter:
    output_dir: Path
    prefix: str
    shard_size: int = 20000
    shard_idx: int = 0
    count: int = 0
    buffer: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    saved_files: List[str] = field(default_factory=list)

    def append(self, sample: Dict[str, Any]) -> None:
        for key, value in sample.items():
            self.buffer.setdefault(key, []).append(_as_array(value))
        self.count += 1
        if self.count >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self.count == 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.prefix}_{self.shard_idx:05d}.npz"
        packed = {k: np.stack(v, axis=0) for k, v in self.buffer.items()}
        np.savez_compressed(path, **packed)
        self.saved_files.append(str(path))
        self.shard_idx += 1
        self.count = 0
        self.buffer = {}


def make_policy(name: str):
    name = name.lower()
    if name == "do_nothing":
        return nominal_go_to_goal
    if name == "apf":
        return nominal_apf
    raise ValueError(f"Unknown policy: {name}")


def collect_dataset(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    set_seed(args.seed if args.seed is not None else cfg.get("seed", 42))

    env_cfg = EnvConfig.from_dict(cfg["env"])
    if args.num_objects is not None:
        env_cfg.num_objects = args.num_objects
    if args.table_half_extent is not None:
        env_cfg.table_half_extent = float(args.table_half_extent)
    if args.contact_distance is not None:
        env_cfg.contact_distance = float(args.contact_distance)
    if args.tilt_gain is not None:
        env_cfg.tilt_gain = float(args.tilt_gain)
    if args.tilt_decay is not None:
        env_cfg.tilt_decay = float(args.tilt_decay)
    if args.goal_tolerance is not None:
        env_cfg.goal_tolerance = float(args.goal_tolerance)
    if args.max_episode_steps is not None:
        env_cfg.max_episode_steps = int(args.max_episode_steps)
    env = PandaClutterEnv(env_cfg)

    policy_fn = make_policy(args.policy)
    toy_barrier = ToyDistanceBarrier(margin=args.toy_margin, object_radius=env_cfg.object_radius)
    safety_filter = SamplingSafetyFilter(
        barrier=toy_barrier,
        max_step=env_cfg.max_action_step,
        num_candidates=args.filter_candidates,
        noise_std=args.filter_noise_std,
        fallback_scale=args.filter_fallback_scale,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    train_writer = ShardWriter(out_dir, prefix="train", shard_size=args.shard_size)
    val_writer = ShardWriter(out_dir, prefix="val", shard_size=args.shard_size)
    rng = np.random.default_rng(args.seed)

    episode_rows = []
    all_tilt_deg = []
    obj_safe_total = 0.0
    obj_count_total = 0
    global_safe_total = 0.0
    global_count_total = 0

    for ep in tqdm(range(args.num_traj), desc="Collecting trajectories"):
        obs, _ = env.reset(seed=args.seed + ep)
        hist = ObservationHistoryBuffer(args.history_len)
        hist.pad_with(obs)

        episode_success = False
        episode_violation = False
        episode_stall = False
        last_step_idx = 0

        for step_idx in range(env_cfg.max_episode_steps):
            last_step_idx = step_idx
            hist_prev = hist.view()
            u_nom = policy_fn(obs, env_cfg.max_action_step)
            if args.use_filter:
                action_xy, _ = safety_filter.step(obs, u_nom, history=hist_prev)
            else:
                action_xy = u_nom

            max_tilt_now = float(np.max(np.rad2deg(obs["objects_tilt_rad"])))
            if max_tilt_now > (env_cfg.tilt_threshold_deg - args.backstep_margin_deg):
                action_xy = -action_xy

            snap = env.snapshot()
            next_obs, reward, terminated, truncated, info = env.step(action_xy)
            _ = reward
            hist.push(next_obs)
            hist_curr = hist.view()

            obj_safe_labels, global_safe = next_state_labels(next_obs, threshold_deg=env_cfg.tilt_threshold_deg)
            next_tilt_deg = np.rad2deg(next_obs["objects_tilt_rad"])
            all_tilt_deg.extend(next_tilt_deg.tolist())
            obj_safe_total += float(np.sum(obj_safe_labels))
            obj_count_total += int(obj_safe_labels.shape[0])
            global_safe_total += float(global_safe)
            global_count_total += 1

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
                    "episode_idx": np.array(ep, dtype=np.int32),
                    "step_idx": np.array(step_idx, dtype=np.int32),
                    "scene_seed": np.array(args.seed + ep, dtype=np.int32),
                    "action_xy": np.asarray(action_xy, dtype=np.float32),
                    "snap_ee": np.asarray(snap["ee_pos"], dtype=np.float32),
                    "snap_goal": np.asarray(snap["goal"], dtype=np.float32),
                    "snap_object_pos": np.asarray(snap["object_pos"], dtype=np.float32),
                    "snap_object_tilt_rad": np.asarray(snap["object_tilt_rad"], dtype=np.float32),
                    "snap_step_count": np.array(snap["step_count"], dtype=np.int32),
                }
                if rng.random() < args.train_ratio:
                    train_writer.append(sample)
                else:
                    val_writer.append(sample)

            obs = next_obs
            episode_success = episode_success or bool(info.get("success", False))
            episode_violation = episode_violation or bool(info.get("violation", False))
            episode_stall = episode_stall or bool(info.get("stall", False))
            if terminated or truncated:
                break

        episode_rows.append(
            {
                "episode_idx": ep,
                "steps": last_step_idx + 1,
                "success": int(episode_success),
                "violation": int(episode_violation),
                "stall": int(episode_stall),
            }
        )

    train_writer.flush()
    val_writer.flush()

    ep_csv = out_dir / "episode_stats.csv"
    ep_csv.parent.mkdir(parents=True, exist_ok=True)
    with ep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()) if episode_rows else [])
        if episode_rows:
            writer.writeheader()
            writer.writerows(episode_rows)

    summary = {
        "num_trajectories": args.num_traj,
        "history_len": args.history_len,
        "env_num_objects": env_cfg.num_objects,
        "env_table_half_extent": env_cfg.table_half_extent,
        "env_contact_distance": env_cfg.contact_distance,
        "env_tilt_gain": env_cfg.tilt_gain,
        "env_tilt_decay": env_cfg.tilt_decay,
        "train_files": train_writer.saved_files,
        "val_files": val_writer.saved_files,
        "object_safe_ratio": float(obj_safe_total / max(obj_count_total, 1)),
        "global_safe_ratio": float(global_safe_total / max(global_count_total, 1)),
        "theta_min_deg": float(np.min(all_tilt_deg) if all_tilt_deg else 0.0),
        "theta_mean_deg": float(np.mean(all_tilt_deg) if all_tilt_deg else 0.0),
        "theta_max_deg": float(np.max(all_tilt_deg) if all_tilt_deg else 0.0),
        "episode_len_mean": float(np.mean([r["steps"] for r in episode_rows]) if episode_rows else 0.0),
        "episode_len_max": int(np.max([r["steps"] for r in episode_rows]) if episode_rows else 0),
    }
    dump_json(summary, out_dir / "collect_summary.json")
    print(f"[collect] saved summary: {out_dir / 'collect_summary.json'}")
    print(
        f"[collect] safe(obj/global)={summary['object_safe_ratio']:.3f}/{summary['global_safe_ratio']:.3f}, "
        f"theta(mean/max)={summary['theta_mean_deg']:.2f}/{summary['theta_max_deg']:.2f}"
    )


def stats_dataset(args: argparse.Namespace) -> None:
    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f"No data files found with pattern: {args.data_glob}")

    all_obj_labels = []
    all_global_labels = []
    all_tilts = []
    episode_lengths: Dict[int, int] = {}
    for file in tqdm(files, desc="Reading shards"):
        data = np.load(file)
        all_obj_labels.append(data["label_safe_obj"].astype(np.float32))
        all_global_labels.append(data["label_safe_global"].astype(np.float32))
        all_tilts.append(data["next_tilt_deg"].astype(np.float32))
        ep = data["episode_idx"].astype(np.int64)
        steps = data["step_idx"].astype(np.int64)
        for epi, step in zip(ep.tolist(), steps.tolist()):
            episode_lengths[epi] = max(episode_lengths.get(epi, -1), int(step) + 1)

    obj_labels = np.concatenate(all_obj_labels, axis=0)
    global_labels = np.concatenate(all_global_labels, axis=0)
    tilts = np.concatenate(all_tilts, axis=0)
    ep_lens = np.array(list(episode_lengths.values()), dtype=np.float32)

    summary = {
        "num_files": len(files),
        "num_samples": int(obj_labels.shape[0]),
        "safe_ratio_object": float(np.mean(obj_labels)),
        "unsafe_ratio_object": float(1.0 - np.mean(obj_labels)),
        "safe_ratio_global": float(np.mean(global_labels)),
        "theta_min_deg": float(np.min(tilts)),
        "theta_mean_deg": float(np.mean(tilts)),
        "theta_max_deg": float(np.max(tilts)),
        "episode_len_mean": float(np.mean(ep_lens)),
        "episode_len_std": float(np.std(ep_lens)),
        "episode_len_max": int(np.max(ep_lens)),
    }
    out_path = Path(args.output_json)
    dump_json(summary, out_path)
    print(f"[stats] saved: {out_path}")
    print(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and summarize DCBF object-centric training data.")
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="Run trajectory collection.")
    collect_parser.add_argument("--config", type=str, default="configs/env.yaml")
    collect_parser.add_argument("--output_dir", type=str, default="outputs/data")
    collect_parser.add_argument("--num_traj", type=int, default=1200)
    collect_parser.add_argument("--num_objects", type=int, default=None)
    collect_parser.add_argument("--table_half_extent", type=float, default=None)
    collect_parser.add_argument("--contact_distance", type=float, default=None)
    collect_parser.add_argument("--tilt_gain", type=float, default=None)
    collect_parser.add_argument("--tilt_decay", type=float, default=None)
    collect_parser.add_argument("--goal_tolerance", type=float, default=None)
    collect_parser.add_argument("--max_episode_steps", type=int, default=None)
    collect_parser.add_argument("--history_len", type=int, default=10)
    collect_parser.add_argument("--policy", type=str, default="do_nothing", choices=["do_nothing", "apf"])
    collect_parser.add_argument("--use_filter", action="store_true", help="Use toy-barrier safety filter in collection.")
    collect_parser.add_argument("--toy_margin", type=float, default=0.02)
    collect_parser.add_argument("--filter_candidates", type=int, default=48)
    collect_parser.add_argument("--filter_noise_std", type=float, default=0.003)
    collect_parser.add_argument("--filter_fallback_scale", type=float, default=0.2)
    collect_parser.add_argument("--backstep_margin_deg", type=float, default=1.0)
    collect_parser.add_argument("--train_ratio", type=float, default=0.9)
    collect_parser.add_argument("--shard_size", type=int, default=15000)
    collect_parser.add_argument("--seed", type=int, default=42)

    stats_parser = subparsers.add_parser("stats", help="Compute dataset distribution stats.")
    stats_parser.add_argument("--data_glob", type=str, default="outputs/data/train_*.npz")
    stats_parser.add_argument("--output_json", type=str, default="outputs/data/stats_summary.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "collect"
    if command == "collect":
        collect_dataset(args)
    elif command == "stats":
        stats_dataset(args)
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
