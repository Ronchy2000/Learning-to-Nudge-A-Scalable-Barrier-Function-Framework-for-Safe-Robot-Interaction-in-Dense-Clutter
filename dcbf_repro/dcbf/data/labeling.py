from __future__ import annotations

from typing import Dict

import numpy as np


def per_object_safe_labels(tilt_deg: np.ndarray, threshold_deg: float = 15.0) -> np.ndarray:
    tilt_deg = np.asarray(tilt_deg, dtype=np.float32)
    return (tilt_deg <= threshold_deg).astype(np.float32)


def global_safe_label(tilt_deg: np.ndarray, threshold_deg: float = 15.0) -> float:
    tilt_deg = np.asarray(tilt_deg, dtype=np.float32)
    return float(np.all(tilt_deg <= threshold_deg))


def next_state_labels(next_obs: Dict[str, np.ndarray], threshold_deg: float = 15.0):
    tilt_deg = np.rad2deg(np.asarray(next_obs["objects_tilt_rad"], dtype=np.float32))
    return per_object_safe_labels(tilt_deg, threshold_deg), global_safe_label(tilt_deg, threshold_deg)
