from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def method_order(df: pd.DataFrame):
    preferred = ["do_nothing", "apf", "initial_dcbf", "refined_dcbf"]
    existing = df["method"].unique().tolist()
    ordered = [name for name in preferred if name in existing]
    for name in existing:
        if name not in ordered:
            ordered.append(name)
    return ordered


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
    num_objects = sorted(df["num_objects"].unique().tolist())
    methods = method_order(df)
    x = np.arange(len(num_objects), dtype=np.float32)
    width = 0.8 / max(len(methods), 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (metric_key, metric_name) in zip(axes, metrics):
        for idx, method in enumerate(methods):
            sub = df[df["method"] == method]
            values = []
            for n in num_objects:
                hit = sub[sub["num_objects"] == n]
                values.append(float(hit.iloc[0][metric_key]) if len(hit) > 0 else np.nan)
            if "rate" in metric_key:
                values = [v * 100.0 if not np.isnan(v) else v for v in values]
            offset = (idx - (len(methods) - 1) / 2.0) * width
            bars = ax.bar(x + offset, values, width=width, label=method, alpha=0.9)
            if "rate" in metric_key:
                ax.bar_label(
                    bars,
                    labels=[f"{v:.0f}" if not np.isnan(v) else "" for v in values],
                    padding=2,
                    fontsize=8,
                )
        ax.set_title(metric_name)
        ax.set_xlabel("Number of Cylinders (N)")
        ax.set_xticks(x)
        ax.set_xticklabels(num_objects)
        ax.grid(alpha=0.3)
        if "rate" in metric_key:
            ax.set_ylabel(f"{metric_name} (%)")
            ax.set_ylim(0.0, 105.0)
    axes[0].legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"[plot] saved: {out_path}")


if __name__ == "__main__":
    main()
