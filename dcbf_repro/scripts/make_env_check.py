#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dcbf.envs.isaaclab_env import EnvConfig, PandaClutterEnv
from dcbf.utils.io import dump_json, load_yaml
from dcbf.utils.seeding import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Environment sanity check for DCBF pipeline.")
    parser.add_argument("--config", type=str, default="configs/env.yaml", help="Path to env yaml.")
    parser.add_argument("--resets", type=int, default=100, help="How many resets.")
    parser.add_argument("--steps", type=int, default=50, help="Random steps per reset.")
    parser.add_argument("--output_dir", type=str, default="outputs/env_check", help="Output folder.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg_dict = load_yaml(args.config)
    set_seed(cfg_dict.get("seed", 42))
    env_cfg = EnvConfig.from_dict(cfg_dict["env"])
    env = PandaClutterEnv(env_cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "env_check_steps.csv"

    tilt_values = []
    violation_checks = 0
    rows = []

    for reset_idx in range(args.resets):
        obs, info = env.reset(seed=cfg_dict.get("seed", 42) + reset_idx)
        _ = (obs, info)
        for step_idx in range(args.steps):
            action = np.random.uniform(-env_cfg.max_action_step, env_cfg.max_action_step, size=(2,))
            obs, reward, terminated, truncated, info = env.step(action)
            _ = (obs, reward)
            max_tilt = info["tilt_max_deg"]
            expected_violation = max_tilt > env_cfg.tilt_threshold_deg
            if expected_violation == bool(info["violation"]):
                violation_checks += 1
            tilt_values.append(max_tilt)
            rows.append(
                {
                    "reset_idx": reset_idx,
                    "step_idx": step_idx,
                    "tilt_min_deg": info["tilt_min_deg"],
                    "tilt_mean_deg": info["tilt_mean_deg"],
                    "tilt_max_deg": max_tilt,
                    "success": int(info["success"]),
                    "violation": int(info["violation"]),
                    "stall": int(info["stall"]),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                }
            )
            if terminated or truncated:
                break

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "resets": args.resets,
        "steps_per_reset": args.steps,
        "samples": len(tilt_values),
        "tilt_min_deg": float(np.min(tilt_values) if tilt_values else 0.0),
        "tilt_mean_deg": float(np.mean(tilt_values) if tilt_values else 0.0),
        "tilt_max_deg": float(np.max(tilt_values) if tilt_values else 0.0),
        "violation_check_pass_ratio": float(violation_checks / max(len(tilt_values), 1)),
        "csv_path": str(csv_path),
    }
    dump_json(summary, output_dir / "summary.json")

    print("=== DCBF Env Check ===")
    print(f"samples={summary['samples']}")
    print(
        f"tilt(min/mean/max)={summary['tilt_min_deg']:.3f}/{summary['tilt_mean_deg']:.3f}/{summary['tilt_max_deg']:.3f}"
    )
    print(f"violation_label_consistency={summary['violation_check_pass_ratio']:.3f}")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
