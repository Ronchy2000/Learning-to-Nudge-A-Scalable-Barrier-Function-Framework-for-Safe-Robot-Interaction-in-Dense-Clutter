from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


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
    """

    def __init__(
        self,
        files: Sequence[str],
        history_len: Optional[int] = None,
        use_global_label: bool = False,
    ) -> None:
        self.files = [str(f) for f in files]
        self.history_len = history_len
        self.use_global_label = use_global_label
        if len(self.files) == 0:
            raise ValueError("DCBFDataset requires at least one data file.")

        arrays: Dict[str, List[np.ndarray]] = {}
        for file in self.files:
            data = np.load(file)
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])

        self.data = {key: np.concatenate(vals, axis=0) for key, vals in arrays.items()}
        self.length = self.data["robot_t"].shape[0]

    @staticmethod
    def from_glob(pattern: str, history_len: Optional[int] = None, use_global_label: bool = False) -> "DCBFDataset":
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No data files found: {pattern}")
        return DCBFDataset(files=files, history_len=history_len, use_global_label=use_global_label)

    def __len__(self) -> int:
        return self.length

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
