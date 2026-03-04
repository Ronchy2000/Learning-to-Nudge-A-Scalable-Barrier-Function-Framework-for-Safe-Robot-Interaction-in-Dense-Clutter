from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def slice_history(history: np.ndarray, history_len: Optional[int]) -> np.ndarray:
    if history_len is None:
        return history
    if history.shape[0] <= history_len:
        return history
    return history[-history_len:]


class DCBFDataset(Dataset):
    """
    Object-centric transition dataset.

    Each sample stores:
    - (r_i^t, O_i^{t-1})
    - (r_i^{t+1}, O_i^t)
    - label_safe_obj, label_safe_global

    Free-space discarding (论文 Sec. V-A):
        当 d_thresh > 0 时，丢弃 EE 距目标物体超过 d_thresh 且安全标签为 safe 的样本。
        所有 unsafe 样本以及正在倾斜(tilt > 0)的样本始终保留。
    """

    def __init__(
        self,
        files: Sequence[str],
        history_len: Optional[int] = None,
        use_global_label: bool = False,
        d_thresh: float = 0.0,
    ) -> None:
        self.files = [str(f) for f in files]
        self.history_len = history_len
        self.use_global_label = use_global_label
        self.d_thresh = d_thresh
        if len(self.files) == 0:
            raise ValueError("DCBFDataset requires at least one data file.")

        arrays: Dict[str, List[np.ndarray]] = {}
        # 收集每个文件的字段名，取交集 — 确保合并 original + refined 数据时
        # 不会因 extra fields (e.g. selected_b_global) 导致大小不一致
        file_keys: List[set] = []
        for file in self.files:
            data = np.load(file)
            file_keys.append(set(data.files))
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
        common_keys = set.intersection(*file_keys) if file_keys else set()
        # 只保留在所有文件中都出现的字段
        arrays = {k: v for k, v in arrays.items() if k in common_keys}

        merged = {key: np.concatenate(vals, axis=0) for key, vals in arrays.items()}

        # --- Free-space discarding ---
        n_raw = merged["robot_t"].shape[0]
        if d_thresh > 0.0:
            label_key = "label_safe_global" if use_global_label else "label_safe_obj"
            labels = merged[label_key]
            # robot_t 已处于 object-centric 坐标，norm(robot_t[:2]) = EE 到物体的 XY 距离
            dist_xy = np.linalg.norm(merged["robot_t"][:, :2], axis=1)
            is_unsafe = labels <= 0.5
            # 有 tilt 的样本也保留（可能正在接触）
            has_tilt = False
            if "next_tilt_deg" in merged:
                has_tilt = merged["next_tilt_deg"] > 0.5  # > 0.5° 视为有 tilt
            keep = is_unsafe | (dist_xy <= d_thresh)
            if isinstance(has_tilt, np.ndarray):
                keep = keep | has_tilt
            keep_idx = np.where(keep)[0]
            merged = {k: v[keep_idx] for k, v in merged.items()}
            n_kept = len(keep_idx)
            print(
                f"[free-space] d_thresh={d_thresh:.3f}m: "
                f"raw={n_raw} → kept={n_kept} ({n_kept/n_raw*100:.1f}%), "
                f"discarded={n_raw - n_kept} free-space safe samples"
            )
        else:
            print(f"[free-space] disabled (d_thresh=0), samples={n_raw}")

        self.data = merged
        self.length = self.data["robot_t"].shape[0]

    @staticmethod
    def from_glob(
        pattern: str,
        history_len: Optional[int] = None,
        use_global_label: bool = False,
        d_thresh: float = 0.0,
    ) -> "DCBFDataset":
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No data files found: {pattern}")
        return DCBFDataset(files=files, history_len=history_len, use_global_label=use_global_label, d_thresh=d_thresh)

    def __len__(self) -> int:
        return self.length

    def make_balanced_sampler(
        self,
        near_boundary_range: Optional[Tuple[float, float]] = None,
        near_boundary_weight: float = 3.0,
    ) -> WeightedRandomSampler:
        """
        根据 safe/unsafe 标签构造 WeightedRandomSampler，使每个 epoch 中
        safe 和 unsafe 样本被采样的期望次数相等。

        论文 Sec. V-A: "balanced, by discarding samples in free space"
        用过采样替代丢弃，效果等价且不丢数据。

        near_boundary_range: (theta_lo, theta_hi) — 度数范围，在此范围内的样本
            额外乘以 near_boundary_weight 的采样权重。
            论文 Sec. V-B: 近边界样本对 refinement 至关重要，训练时也应加强。
        """
        label_key = "label_safe_global" if self.use_global_label else "label_safe_obj"
        labels = self.data[label_key]
        n_safe = int((labels > 0.5).sum())
        n_unsafe = int((labels <= 0.5).sum())
        if n_unsafe == 0:
            raise ValueError(
                f"Dataset has 0 unsafe samples (all {n_safe} safe). "
                f"Cannot balance. Increase tilt_gain or reduce table_half_extent."
            )
        # 基础权重: 每个安全样本 = 1/(2*n_safe)，每个 unsafe = 1/(2*n_unsafe)
        w_safe = 1.0 / (2.0 * n_safe)
        w_unsafe = 1.0 / (2.0 * n_unsafe)
        weights = np.where(labels > 0.5, w_safe, w_unsafe).astype(np.float64)

        # 近边界加权
        n_boundary = 0
        if near_boundary_range is not None and "next_tilt_deg" in self.data:
            theta_lo, theta_hi = near_boundary_range
            tilt = self.data["next_tilt_deg"]
            boundary_mask = (tilt >= theta_lo) & (tilt <= theta_hi)
            n_boundary = int(boundary_mask.sum())
            weights[boundary_mask] *= near_boundary_weight

        print(
            f"[balance] safe={n_safe} ({n_safe / len(labels) * 100:.1f}%)  "
            f"unsafe={n_unsafe} ({n_unsafe / len(labels) * 100:.1f}%)  "
            f"w_safe/w_unsafe={w_safe / w_unsafe:.4f}"
        )
        if n_boundary > 0:
            print(
                f"[balance] near-boundary θ∈[{near_boundary_range[0]:.0f}°,{near_boundary_range[1]:.0f}°]: "
                f"{n_boundary} samples ({n_boundary / len(labels) * 100:.1f}%), "
                f"weight boost ×{near_boundary_weight:.1f}"
            )
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(labels),
            replacement=True,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obj_prev = slice_history(self.data["obj_hist_prev"][idx], self.history_len)
        obj_curr = slice_history(self.data["obj_hist_curr"][idx], self.history_len)
        safe_label = self.data["label_safe_global"][idx] if self.use_global_label else self.data["label_safe_obj"][idx]
        return {
            "robot_t": torch.as_tensor(self.data["robot_t"][idx], dtype=torch.float32),
            "robot_tp1": torch.as_tensor(self.data["robot_tp1"][idx], dtype=torch.float32),
            "obj_hist_prev": torch.as_tensor(obj_prev, dtype=torch.float32),
            "obj_hist_curr": torch.as_tensor(obj_curr, dtype=torch.float32),
            "safe_label": torch.as_tensor(safe_label, dtype=torch.float32),
            "safe_label_obj": torch.as_tensor(self.data["label_safe_obj"][idx], dtype=torch.float32),
            "safe_label_global": torch.as_tensor(self.data["label_safe_global"][idx], dtype=torch.float32),
        }


def split_files_by_ratio(files: Sequence[str], train_ratio: float = 0.9):
    files = list(files)
    if len(files) == 1:
        return files, files
    split = max(1, int(len(files) * train_ratio))
    split = min(split, len(files) - 1)
    return files[:split], files[split:]


def discover_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return files
