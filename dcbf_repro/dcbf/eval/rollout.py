from __future__ import annotations

import argparse
from pathlib import Path

from dcbf.eval.evaluate import load_model_and_filter, run_episode
from dcbf.envs.isaaclab_env import EnvConfig, PandaClutterEnv
from dcbf.utils.io import dump_json, load_yaml
from dcbf.utils.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick rollout and save metrics JSON.")
    parser.add_argument("--env_config", type=str, default="configs/env.yaml")
    parser.add_argument("--method", type=str, default="do_nothing", choices=["do_nothing", "apf", "initial_dcbf", "refined_dcbf"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/eval/rollout_summary.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_objects", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    env_cfg_yaml = load_yaml(args.env_config)
    env_cfg = EnvConfig.from_dict(env_cfg_yaml["env"])
    if args.num_objects is not None:
        env_cfg.num_objects = int(args.num_objects)

    env = PandaClutterEnv(env_cfg)
    history_len = 10
    safety_filter = None
    if args.method in {"initial_dcbf", "refined_dcbf"}:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for DCBF rollout methods.")
        history_len, safety_filter = load_model_and_filter(
            checkpoint_path=args.checkpoint,
            filter_cfg={"num_candidates": 64, "noise_std": 0.003, "fallback_scale": 0.2, "top_m_objects": 12},
            max_step=env_cfg.max_action_step,
        )

    metrics = {"success": 0.0, "violation": 0.0, "stall": 0.0, "steps": 0.0}
    for epi in range(args.episodes):
        out = run_episode(env, args.method, history_len=history_len, safety_filter=safety_filter, seed=args.seed + epi)
        for key in metrics:
            metrics[key] += out[key]
    summary = {f"{k}_mean": metrics[k] / max(args.episodes, 1) for k in metrics}
    summary["method"] = args.method
    summary["episodes"] = args.episodes
    summary["num_objects"] = env_cfg.num_objects
    dump_json(summary, Path(args.output))
    print(f"[rollout] saved summary: {args.output}")


if __name__ == "__main__":
    main()
