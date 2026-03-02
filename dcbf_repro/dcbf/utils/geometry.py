from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np


def obs_to_object_states(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["objects_pos"], dtype=np.float32),
            np.asarray(obs["objects_tilt_rad"], dtype=np.float32)[:, None],
        ],
        axis=1,
    )


def relative_transform_points(points_xyz: np.ndarray, anchor_xyz: np.ndarray) -> np.ndarray:
    out = np.asarray(points_xyz, dtype=np.float32).copy()
    out[..., :3] -= np.asarray(anchor_xyz[:3], dtype=np.float32)
    return out


def object_centric_transform(
    robot_state_xyz: np.ndarray,
    object_history: np.ndarray,
    object_index: int,
    anchor_step: int = 0,
    anchor_xyz: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Object-centric transform R(·) with translation-only anchor.

    robot_state_xyz: (3,)
    object_history: (T, N, 4) where 4 = (x,y,z,tilt)
    object_index: target i
    anchor_step: index used as o_i^{t-T} anchor (default: oldest step)
    """
    if anchor_xyz is None:
        anchor = np.asarray(object_history[anchor_step, object_index, :3], dtype=np.float32)
    else:
        anchor = np.asarray(anchor_xyz[:3], dtype=np.float32)
    robot_rel = np.asarray(robot_state_xyz, dtype=np.float32).copy()
    robot_rel[:3] -= anchor
    obj_rel = np.asarray(object_history[:, object_index, :], dtype=np.float32).copy()
    obj_rel[:, :3] -= anchor
    return robot_rel, obj_rel


@dataclass
class HistoryView:
    robot_hist: np.ndarray  # (T,3)
    object_hist: np.ndarray  # (T,N,4)


class ObservationHistoryBuffer:
    def __init__(self, history_len: int):
        self.history_len = int(history_len)
        self._robot_hist: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self._object_hist: Deque[np.ndarray] = deque(maxlen=self.history_len)

    def clear(self) -> None:
        self._robot_hist.clear()
        self._object_hist.clear()

    def push(self, obs: Dict[str, np.ndarray]) -> None:
        robot = np.asarray(obs["ee_pos"], dtype=np.float32).copy()
        obj = obs_to_object_states(obs)
        self._robot_hist.append(robot)
        self._object_hist.append(obj)

    def pad_with(self, obs: Dict[str, np.ndarray]) -> None:
        if len(self._robot_hist) > 0:
            return
        for _ in range(self.history_len):
            self.push(obs)

    @property
    def ready(self) -> bool:
        return len(self._robot_hist) >= self.history_len and len(self._object_hist) >= self.history_len

    def view(self) -> HistoryView:
        if not self.ready:
            raise RuntimeError("History buffer not ready.")
        return HistoryView(
            robot_hist=np.stack(list(self._robot_hist), axis=0),
            object_hist=np.stack(list(self._object_hist), axis=0),
        )

    @property
    def size(self) -> int:
        return len(self._robot_hist)


def nearest_object_indices(ee_xy: np.ndarray, objects_xyz: np.ndarray, top_m: int) -> np.ndarray:
    ee_xy = np.asarray(ee_xy, dtype=np.float32).reshape(2)
    obj_xy = np.asarray(objects_xyz, dtype=np.float32)[:, :2]
    dists = np.linalg.norm(obj_xy - ee_xy[None, :], axis=1)
    order = np.argsort(dists)
    return order[: min(top_m, len(order))]


def batch_object_centric_from_history(
    robot_state_xyz: np.ndarray,
    history: HistoryView,
    object_indices: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    robot_feats: List[np.ndarray] = []
    object_feats: List[np.ndarray] = []
    for idx in object_indices:
        r_rel, o_rel = object_centric_transform(
            robot_state_xyz,
            history.object_hist,
            object_index=int(idx),
            anchor_step=0,
        )
        robot_feats.append(r_rel)
        object_feats.append(o_rel)
    if not robot_feats:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, history.object_hist.shape[0], 4), dtype=np.float32)
    return np.stack(robot_feats, axis=0), np.stack(object_feats, axis=0)
