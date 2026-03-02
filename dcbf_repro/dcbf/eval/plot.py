from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics for DCBF reproduction.")
    parser.add_argument("--csv", type=str, default="outputs/eval/metrics.csv")
    parser.add_argument("--output", type=str, default="outputs/eval/metrics_plot.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics csv not found: {csv_path}")
    df = pd.read_csv(csv_path)

    metrics = [
        ("success_rate", "Success Rate"),
        ("violation_rate", "Violation Rate"),
        ("stalling_rate", "Stalling Rate"),
        ("avg_episode_steps", "Avg Episode Steps"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (metric_key, metric_name) in zip(axes, metrics):
        for method in sorted(df["method"].unique().tolist()):
            sub = df[df["method"] == method].sort_values("num_objects")
            ax.plot(sub["num_objects"], sub[metric_key], marker="o", label=method)
        ax.set_title(metric_name)
        ax.set_xlabel("Number of Cylinders (N)")
        ax.grid(alpha=0.3)
        if "rate" in metric_key:
            ax.set_ylim(0.0, 1.0)
    axes[0].legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"[plot] saved: {out_path}")


if __name__ == "__main__":
    main()
