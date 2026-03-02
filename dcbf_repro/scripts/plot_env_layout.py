#!/usr/bin/env python3
"""
可视化 mock 环境的 2D 俯视物体布局。

生成 env_layout_densities.png:
  2×2 网格，N=4/10/20/40 在同一张 0.7m×0.7m 桌面。
  N=4 是训练密度，N=10/20/40 是泛化评估密度（论文 Fig.3 & Fig.4）。
  只画桌面和瓶子分布。

物体放置策略（仅代码内部实现细节，不在图中体现）:
  N≤12 用拒绝采样（rejection sampling）：随机坐标 → 检查重叠 → 重叠就丢弃重来。
  N>12 用六角网格+抖动（hex grid + jitter）：先铺六角网格 → 随机选 N 个格点 →
       加小随机偏移，保证高密度也能放满。阈值和参数见 _sample_objects_grid()。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 项目根目录（dcbf_repro/），所有输出路径以此为基准，
# 不受 cwd、IDE、操作系统影响。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dcbf.envs.isaaclab_env import EnvConfig, MockPandaClutterEnv


# ── 配色 ─────────────────────────────────────────────
TABLE_BG = "#F7F3EB"
TABLE_EDGE = "#888888"
OBJ_FACE = "#5AADE0"
OBJ_EDGE = "#2B6E8E"


# ── 绘制单个俯视场景（只画桌面 + 瓶子分布）────────────
def draw_scene(
    ax: plt.Axes,
    env: MockPandaClutterEnv,
    *,
    title: str = "",
    show_legend: bool = False,
    show_dimension: bool = False,
    subtitle: str = "",
):
    cfg = env.cfg
    half = cfg.table_half_extent
    radius = cfg.object_radius

    # 桌面
    rect = mpatches.FancyBboxPatch(
        (-half, -half), 2 * half, 2 * half,
        boxstyle="round,pad=0.006",
        linewidth=1.8, edgecolor=TABLE_EDGE, facecolor=TABLE_BG, zorder=0,
    )
    ax.add_patch(rect)

    # 瓶子
    for i in range(cfg.num_objects):
        ox, oy = env.object_pos[i, 0], env.object_pos[i, 1]
        body = plt.Circle(
            (ox, oy), radius,
            facecolor=OBJ_FACE, edgecolor=OBJ_EDGE,
            linewidth=0.8, alpha=0.88, zorder=3,
        )
        ax.add_patch(body)
        # 中心标记（模仿论文 Fig.2 风格）
        ax.plot(ox, oy, ".", color="#C0392B", markersize=2.5, zorder=4)

    # 尺寸标注
    if show_dimension:
        y_dim = -half - 0.025
        ax.annotate(
            "", xy=(half, y_dim), xytext=(-half, y_dim),
            arrowprops=dict(arrowstyle="<->", color="#666", linewidth=1.0),
            zorder=7,
        )
        ax.text(0, y_dim - 0.015, f"{2 * half:.1f} m",
                ha="center", va="top", fontsize=8, color="#555")

    margin = 0.05
    ax.set_xlim(-half - margin, half + margin)
    ax.set_ylim(-half - margin - (0.04 if show_dimension else 0), half + margin)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if subtitle:
        ax.text(
            0.5, -0.06, subtitle,
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#666", style="italic",
        )
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    if show_legend:
        handles = [
            mpatches.Patch(fc=OBJ_FACE, ec=OBJ_EDGE,
                           label=f"Cylinder (r={radius} m, h={cfg.object_height} m)"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  framealpha=0.85, edgecolor="#CCC")


# ── 图 1: 四种密度 (2×2) ──────────────────────────────
def plot_densities(output_path: str, seed: int = 42):
    densities = [4, 10, 20, 40]
    labels = {
        4:  "N = 4  (training)",
        10: "N = 10",
        20: "N = 20",
        40: "N = 40  (densest eval)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        "Clutter Density — Same Table (0.7 m × 0.7 m)\n"
        "Trained on N=4, evaluated on N=4 / 10 / 20 / 40",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ax, n in zip(axes.flatten(), densities):
        cfg = EnvConfig(num_objects=n, table_half_extent=0.35)
        env = MockPandaClutterEnv(cfg)
        env.reset(seed=seed)
        draw_scene(
            ax, env,
            title=labels[n],
            show_legend=(n == 4),
            show_dimension=(n == 4),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"[plot] saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mock environment object placement layouts (2D top-down view)."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "outputs" / "env_layout"),
        help="Output directory (default: <project_root>/outputs/env_layout)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    plot_densities(str(out / "env_layout_densities.png"), seed=args.seed)
    print(f"\n[plot_env_layout] saved to {out}/")


if __name__ == "__main__":
    main()
