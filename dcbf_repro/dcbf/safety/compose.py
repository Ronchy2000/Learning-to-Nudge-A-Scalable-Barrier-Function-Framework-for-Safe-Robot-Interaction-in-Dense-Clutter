from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from dcbf.utils.geometry import HistoryView, batch_object_centric_from_history, nearest_object_indices

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def compose_min(barrier_values: np.ndarray) -> float:
    if barrier_values.size == 0:
        return float("inf")
    return float(np.min(barrier_values))


class BarrierScorer(Protocol):
    def score_action(self, obs, history: Optional[HistoryView], action_xy: np.ndarray) -> float:
        ...


@dataclass
class ToyDistanceBarrier:
    margin: float = 0.02
    object_radius: float = 0.025

    def score_action(self, obs, history: Optional[HistoryView], action_xy: np.ndarray) -> float:
        ee_next_xy = np.asarray(obs["ee_pos"], dtype=np.float32)[:2] + np.asarray(action_xy, dtype=np.float32).reshape(2)
        objects_xy = np.asarray(obs["objects_pos"], dtype=np.float32)[:, :2]
        dists = np.linalg.norm(objects_xy - ee_next_xy[None, :], axis=1)
        clearances = dists - self.object_radius
        barrier_values = clearances - self.margin
        return compose_min(barrier_values)


class LearnedGlobalBarrier:
    def __init__(
        self,
        model,
        device: str = "cpu",
        top_m_objects: Optional[int] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for LearnedGlobalBarrier.")
        self.model = model
        self.model.eval()
        self.device = device
        self.top_m_objects = top_m_objects

    def score_action(self, obs, history: Optional[HistoryView], action_xy: np.ndarray) -> float:
        if history is None or history.object_hist.shape[0] == 0:
            return 0.0
        action_xy = np.asarray(action_xy, dtype=np.float32).reshape(2)
        robot_next = np.asarray(obs["ee_pos"], dtype=np.float32).copy()
        robot_next[:2] += action_xy

        num_objects = history.object_hist.shape[1]
        if self.top_m_objects is None:
            object_indices = np.arange(num_objects)
        else:
            object_indices = nearest_object_indices(robot_next[:2], obs["objects_pos"], self.top_m_objects)

        robot_feats, object_feats = batch_object_centric_from_history(robot_next, history, object_indices)
        if robot_feats.shape[0] == 0:
            return 0.0
        with torch.no_grad():
            robot_t = torch.from_numpy(robot_feats).to(self.device)
            obj_t = torch.from_numpy(object_feats).to(self.device)
            barrier_values = self.model(robot_t, obj_t).squeeze(-1)
            return float(torch.min(barrier_values).item())
