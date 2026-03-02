#!/usr/bin/env python3
"""
绘制 CBF 值的 2D 热力图（论文 Fig.2 风格）。

将 EE 的 (x,y) 在桌面网格上逐点扫描，对每个位置计算全局 barrier 值
（所有物体的 h_i 取 min），用蓝-红 diverging colormap 渲染。

支持两种模式：
  1. 单 checkpoint：生成 1 张图
  2. 双 checkpoint（--checkpoint_init + --checkpoint_refined）：
     生成论文 Fig.2 风格的左右对比图（Initial CBF / Refinement CBF）

物体位置可以手动指定（--objects），也可以从 mock 环境随机采样
（--num_objects + --seed）。

用法示例：

  # 单图
  python scripts/plot_cbf_heatmap.py \
      --checkpoint outputs/train/ours_sigma_001/best.pt \
      --num_objects 20 --seed 42

  # 双图对比（论文 Fig.2）
  python scripts/plot_cbf_heatmap.py \
      --checkpoint_init  outputs/train/ours_sigma_001/best.pt \
      --checkpoint_refined outputs/refine/sigma_001/best.pt \
      --num_objects 20 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

# 项目根目录（dcbf_repro/），所有输出路径以此为基准，
# 不受 cwd、IDE、操作系统影响。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dcbf.models.dcbf_net import DCBFNet
from dcbf.envs.isaaclab_env import EnvConfig, MockPandaClutterEnv


# ── 模型加载 ──────────────────────────────────────────
def load_model(path: str, device: str = "cpu") -> Tuple[DCBFNet, dict]:
    ckpt = torch.load(path, map_location=device)
    mc = ckpt["config"]["model"]
    model = DCBFNet(
        robot_dim=mc["robot_dim"],
        object_dim=mc["object_dim"],
        history_len=mc["history_len"],
        lstm_hidden=mc["lstm_hidden"],
        lstm_layers=mc["lstm_layers"],
        mlp_hidden=mc["mlp_hidden"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, mc


# ── 特征构建 & CBF 值计算 ─────────────────────────────
def compute_cbf_grid(
    model: DCBFNet,
    grid_xy: np.ndarray,       # (G, 2)
    object_xy: np.ndarray,     # (N, 2)
    fixed_z: float,
    history_len: int,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """
    对每个网格点 (x,y)，与每个物体 i 构建 object-centric 特征:
      robot_feat  = (ee_x - obj_x, ee_y - obj_y, fixed_z)    # (3,)
      obj_hist    = zeros(T, 4)   # 静态场景，历史状态全为 0
    前向推理得 h_i(x)，取 min_i h_i 作为全局 barrier 值。

    Returns: (G,) array
    """
    G, N, T = grid_xy.shape[0], object_xy.shape[0], history_len

    # robot_feat: (G, N, 3)
    rf = np.zeros((G, N, 3), dtype=np.float32)
    for i in range(N):
        rf[:, i, 0] = grid_xy[:, 0] - object_xy[i, 0]
        rf[:, i, 1] = grid_xy[:, 1] - object_xy[i, 1]
        rf[:, i, 2] = fixed_z
    rf = rf.reshape(G * N, 3)

    # object_hist_feat: 全 0（静态场景）
    of = np.zeros((G * N, T, 4), dtype=np.float32)

    barriers = np.zeros(G * N, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, G * N, batch_size):
            e = min(s + batch_size, G * N)
            r_t = torch.from_numpy(rf[s:e]).to(device)
            o_t = torch.from_numpy(of[s:e]).to(device)
            barriers[s:e] = model(r_t, o_t).squeeze(-1).cpu().numpy()

    return barriers.reshape(G, N).min(axis=1)  # (G,)


# ── 绘制单个子图 ─────────────────────────────────────
def draw_cbf_panel(
    ax: plt.Axes,
    cbf_2d: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    object_xy: np.ndarray,
    object_radius: float,
    *,
    title: str = "",
    ee_xy: Optional[np.ndarray] = None,
):
    """论文 Fig.2 风格: 蓝-白-红 diverging colormap + h=0 虚线等值线 + 瓶子圆"""
    vmax = max(abs(cbf_2d.min()), abs(cbf_2d.max()), 0.01)
    im = ax.pcolormesh(
        xs, ys, cbf_2d,
        cmap="RdBu_r", shading="auto",
        vmin=-vmax, vmax=vmax,
    )
    # h=0 等值线（安全/不安全分界）
    ax.contour(
        xs, ys, cbf_2d,
        levels=[0.0], colors="k", linewidths=1.2, linestyles="--",
    )
    # 瓶子
    for ox, oy in object_xy:
        c = Circle(
            (ox, oy), object_radius,
            fill=True, facecolor="#E88070", edgecolor="k",
            alpha=0.75, linewidth=0.8, zorder=5,
        )
        ax.add_patch(c)
        ax.plot(ox, oy, ".", color="#C0392B", markersize=2, zorder=6)

    # EE 位置
    if ee_xy is not None:
        ax.plot(ee_xy[0], ee_xy[1], marker="*", color="red",
                markersize=10, markeredgecolor="k", markeredgewidth=0.5, zorder=7)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # legend
    legend_handles = [
        mpatches.Patch(fc="#E88070", ec="k", label="bottle 1"),
    ]
    if ee_xy is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                       markeredgecolor="k", markersize=8, label="end-effector")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              framealpha=0.85, edgecolor="#CCC")

    return im


# ── 解析物体坐标 ─────────────────────────────────────
def parse_objects(strs: List[str]) -> np.ndarray:
    coords = []
    for s in strs:
        parts = s.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected 'x,y' but got: '{s}'")
        coords.append([float(parts[0]), float(parts[1])])
    return np.array(coords, dtype=np.float32)


# ── 从 mock 环境采样物体 ─────────────────────────────
def sample_objects_from_env(
    num_objects: int, seed: int, table_half_extent: float = 0.35
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (object_xy (N,2), ee_xy (2,))"""
    cfg = EnvConfig(num_objects=num_objects, table_half_extent=table_half_extent)
    env = MockPandaClutterEnv(cfg)
    env.reset(seed=seed)
    obj_xy = env.object_pos[:, :2].copy()
    ee_xy = env.ee_pos[:2].copy()
    return obj_xy, ee_xy


# ── main ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Plot 2D CBF heatmap (paper Fig.2).")
    # checkpoint 参数组: 单图模式用 --checkpoint, 双图模式用 --checkpoint_init + --checkpoint_refined
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Single checkpoint path (single-panel mode)")
    p.add_argument("--checkpoint_init", type=str, default=None,
                   help="Initial model checkpoint (two-panel mode, left)")
    p.add_argument("--checkpoint_refined", type=str, default=None,
                   help="Refined model checkpoint (two-panel mode, right)")
    # 物体位置
    p.add_argument("--objects", type=str, nargs="+", default=None,
                   help="Manual object positions as 'x,y' strings")
    p.add_argument("--num_objects", type=int, default=20,
                   help="Sample this many objects from mock env (ignored if --objects given)")
    p.add_argument("--seed", type=int, default=42)
    # 绘图参数
    p.add_argument("--grid_range", type=float, nargs=2, default=[-0.35, 0.35])
    p.add_argument("--grid_res", type=int, default=200)
    p.add_argument("--fixed_z", type=float, default=0.12)
    p.add_argument("--object_radius", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "outputs" / "cbf_heatmap.png"),
        help="Output path (default: <project_root>/outputs/cbf_heatmap.png)",
    )
    args = p.parse_args()

    # 确定是单图还是双图模式
    two_panel = (args.checkpoint_init is not None and args.checkpoint_refined is not None)
    if not two_panel and args.checkpoint is None:
        p.error("需要指定 --checkpoint (单图) 或 --checkpoint_init + --checkpoint_refined (双图)")

    # 物体位置
    ee_xy = None
    if args.objects:
        object_xy = parse_objects(args.objects)
    else:
        object_xy, ee_xy = sample_objects_from_env(
            args.num_objects, args.seed,
            table_half_extent=args.grid_range[1],
        )
    print(f"[heatmap] {object_xy.shape[0]} objects")

    # 网格
    lo, hi = args.grid_range
    xs = np.linspace(lo, hi, args.grid_res)
    ys = np.linspace(lo, hi, args.grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    print(f"[heatmap] grid {args.grid_res}x{args.grid_res}, range [{lo}, {hi}]")

    if two_panel:
        # ── 双图模式 (论文 Fig.2) ──
        model_init, mc_init = load_model(args.checkpoint_init, args.device)
        model_ref, mc_ref = load_model(args.checkpoint_refined, args.device)

        cbf_init = compute_cbf_grid(
            model_init, grid_xy, object_xy, args.fixed_z,
            mc_init["history_len"], args.device, args.batch_size,
        ).reshape(args.grid_res, args.grid_res)

        cbf_ref = compute_cbf_grid(
            model_ref, grid_xy, object_xy, args.fixed_z,
            mc_ref["history_len"], args.device, args.batch_size,
        ).reshape(args.grid_res, args.grid_res)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

        im_l = draw_cbf_panel(
            ax_l, cbf_init, xs, ys, object_xy, args.object_radius,
            title="Initial CBF", ee_xy=ee_xy,
        )
        fig.colorbar(im_l, ax=ax_l, shrink=0.82, pad=0.02)

        im_r = draw_cbf_panel(
            ax_r, cbf_ref, xs, ys, object_xy, args.object_radius,
            title="Refinement CBF", ee_xy=ee_xy,
        )
        fig.colorbar(im_r, ax=ax_r, shrink=0.82, pad=0.02)

        fig.tight_layout()

    else:
        # ── 单图模式 ──
        model, mc = load_model(args.checkpoint, args.device)
        cbf_flat = compute_cbf_grid(
            model, grid_xy, object_xy, args.fixed_z,
            mc["history_len"], args.device, args.batch_size,
        ).reshape(args.grid_res, args.grid_res)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im = draw_cbf_panel(
            ax, cbf_flat, xs, ys, object_xy, args.object_radius,
            title="Learned CBF", ee_xy=ee_xy,
        )
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"[heatmap] saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
