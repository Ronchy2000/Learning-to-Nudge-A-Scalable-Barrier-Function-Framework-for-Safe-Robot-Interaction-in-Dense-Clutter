from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    backend: str = "mock"
    num_objects: int = 20
    table_half_extent: float = 0.35
    object_radius: float = 0.025
    max_episode_steps: int = 200
    max_action_step: float = 0.01
    goal_tolerance: float = 0.02
    fixed_z: float = 0.12
    tilt_threshold_deg: float = 15.0
    tilt_warning_margin_deg: float = 2.0
    stall_window: int = 20
    stall_movement_eps: float = 0.001
    stall_progress_eps: float = 0.0005
    contact_distance: float = 0.05
    tilt_gain: float = 1.8
    tilt_decay: float = 0.002

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
        return cls(**data)


class MockPandaClutterEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.object_pos = np.zeros((cfg.num_objects, 3), dtype=np.float32)
        self.object_tilt_rad = np.zeros(cfg.num_objects, dtype=np.float32)
        self.step_count = 0
        self._stall_counter = 0
        self._prev_goal_dist = 0.0
        self._last_seed = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._last_seed = seed
        self.step_count = 0
        self._stall_counter = 0
        self.ee_pos[:] = np.array(
            [
                self.rng.uniform(-0.18, 0.18),
                self.rng.uniform(-0.18, 0.18),
                self.cfg.fixed_z,
            ],
            dtype=np.float32,
        )
        self.goal[:] = np.array(
            [self.rng.uniform(-0.24, 0.24), self.rng.uniform(-0.24, 0.24)],
            dtype=np.float32,
        )
        self.object_pos[:] = self._sample_objects()
        self.object_tilt_rad[:] = 0.0
        self._prev_goal_dist = self._goal_distance()
        return self._get_obs(), {"seed": self._last_seed}

    def _sample_objects(self) -> np.ndarray:
        sampled = np.zeros((self.cfg.num_objects, 3), dtype=np.float32)
        min_dist = self.cfg.object_radius * 2.2
        placed = 0
        retries = 0
        while placed < self.cfg.num_objects:
            if retries > 20000:
                raise RuntimeError("Cannot place objects without overlap in current table setup.")
            candidate = np.array(
                [
                    self.rng.uniform(-self.cfg.table_half_extent, self.cfg.table_half_extent),
                    self.rng.uniform(-self.cfg.table_half_extent, self.cfg.table_half_extent),
                    0.0,
                ],
                dtype=np.float32,
            )
            retries += 1
            if np.linalg.norm(candidate[:2] - self.ee_pos[:2]) < 0.06:
                continue
            if np.linalg.norm(candidate[:2] - self.goal[:2]) < 0.06:
                continue
            if placed > 0 and np.min(np.linalg.norm(sampled[:placed, :2] - candidate[:2], axis=1)) < min_dist:
                continue
            sampled[placed] = candidate
            placed += 1
        return sampled

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.ee_pos[:2]))

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(action))
        if norm > self.cfg.max_action_step and norm > 1e-9:
            action = action / norm * self.cfg.max_action_step
        return action

    def _apply_contacts(self, action: np.ndarray) -> None:
        ee_xy = self.ee_pos[:2]
        for idx in range(self.cfg.num_objects):
            rel = self.object_pos[idx, :2] - ee_xy
            dist = float(np.linalg.norm(rel))
            if dist < self.cfg.contact_distance:
                contact_scale = (self.cfg.contact_distance - dist) / max(self.cfg.contact_distance, 1e-6)
                push = action * (0.8 + 0.2 * contact_scale)
                self.object_pos[idx, :2] += push
                self.object_pos[idx, :2] = np.clip(
                    self.object_pos[idx, :2], -self.cfg.table_half_extent, self.cfg.table_half_extent
                )
                tilt_delta = self.cfg.tilt_gain * float(np.linalg.norm(push)) * (1.0 + 0.5 * contact_scale)
                self.object_tilt_rad[idx] += tilt_delta
        self.object_tilt_rad -= self.cfg.tilt_decay
        self.object_tilt_rad = np.clip(self.object_tilt_rad, 0.0, np.deg2rad(45.0))

    def _compute_stall(self, ee_prev: np.ndarray) -> Tuple[bool, float, float]:
        movement = float(np.linalg.norm(self.ee_pos[:2] - ee_prev[:2]))
        goal_dist = self._goal_distance()
        progress = float(self._prev_goal_dist - goal_dist)
        self._prev_goal_dist = goal_dist
        if movement < self.cfg.stall_movement_eps and progress < self.cfg.stall_progress_eps:
            self._stall_counter += 1
        else:
            self._stall_counter = 0
        stall = self._stall_counter >= self.cfg.stall_window
        return stall, movement, progress

    def step(self, action: np.ndarray):
        action = self._clip_action(action)
        self.step_count += 1
        ee_prev = self.ee_pos.copy()

        self.ee_pos[:2] += action
        self.ee_pos[:2] = np.clip(self.ee_pos[:2], -self.cfg.table_half_extent, self.cfg.table_half_extent)
        self.ee_pos[2] = self.cfg.fixed_z

        self._apply_contacts(action)
        tilts_deg = self.get_tilts_deg()
        violation = bool(np.any(tilts_deg > self.cfg.tilt_threshold_deg))
        success = self._goal_distance() < self.cfg.goal_tolerance
        stall, movement, progress = self._compute_stall(ee_prev)
        terminated = success or violation or stall
        truncated = self.step_count >= self.cfg.max_episode_steps

        reward = -self._goal_distance()
        if success:
            reward += 1.0
        if violation:
            reward -= 1.0
        if stall:
            reward -= 0.3

        info = {
            "success": success,
            "violation": violation,
            "stall": stall,
            "movement": movement,
            "progress": progress,
            "step": self.step_count,
            "tilt_min_deg": float(np.min(tilts_deg)),
            "tilt_mean_deg": float(np.mean(tilts_deg)),
            "tilt_max_deg": float(np.max(tilts_deg)),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "ee_pos": self.ee_pos.astype(np.float32).copy(),
            "goal_xy": self.goal.astype(np.float32).copy(),
            "objects_pos": self.object_pos.astype(np.float32).copy(),
            "objects_tilt_rad": self.object_tilt_rad.astype(np.float32).copy(),
            "objects_quat": self._fake_quat_from_tilt(self.object_tilt_rad),
        }

    @staticmethod
    def _fake_quat_from_tilt(tilt_rad: np.ndarray) -> np.ndarray:
        quat = np.zeros((tilt_rad.shape[0], 4), dtype=np.float32)
        quat[:, 0] = np.cos(tilt_rad / 2.0)
        quat[:, 1] = np.sin(tilt_rad / 2.0)
        return quat

    def get_tilts_deg(self) -> np.ndarray:
        return np.rad2deg(self.object_tilt_rad).astype(np.float32)

    def get_object_states(self) -> np.ndarray:
        return np.concatenate([self.object_pos, self.object_tilt_rad[:, None]], axis=1).astype(np.float32)

    def get_obs(self) -> Dict[str, np.ndarray]:
        return self._get_obs()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ee_pos": self.ee_pos.copy(),
            "goal": self.goal.copy(),
            "object_pos": self.object_pos.copy(),
            "object_tilt_rad": self.object_tilt_rad.copy(),
            "step_count": self.step_count,
            "stall_counter": self._stall_counter,
            "prev_goal_dist": self._prev_goal_dist,
            "rng_state": self.rng.bit_generator.state,
            "seed": self._last_seed,
        }

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.ee_pos[:] = np.asarray(snap["ee_pos"], dtype=np.float32)
        self.goal[:] = np.asarray(snap["goal"], dtype=np.float32)
        self.object_pos[:] = np.asarray(snap["object_pos"], dtype=np.float32)
        self.object_tilt_rad[:] = np.asarray(snap["object_tilt_rad"], dtype=np.float32)
        self.step_count = int(snap.get("step_count", 0))
        self._stall_counter = int(snap.get("stall_counter", 0))
        self._prev_goal_dist = float(snap.get("prev_goal_dist", self._goal_distance()))
        self._last_seed = snap.get("seed", None)
        if "rng_state" in snap:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = snap["rng_state"]

    def close(self) -> None:
        return None


class PandaClutterEnv:
    """
    DCBF environment wrapper.

    Backend choices:
    - mock: built-in light-weight simulator (always available).
    - isaaclab: TODO integration placeholder for IsaacLab/Isaac Sim API.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.backend = cfg.backend.lower()
        if self.backend == "isaaclab":
            self._env = self._build_isaaclab_or_fallback(cfg)
        else:
            self._env = MockPandaClutterEnv(cfg)

    def _build_isaaclab_or_fallback(self, cfg: EnvConfig):
        try:
            import omni  # type: ignore  # noqa: F401
        except Exception:
            print("[WARN] Isaac backend requested but unavailable; fallback to mock backend.")
            return MockPandaClutterEnv(cfg)
        raise NotImplementedError(
            "IsaacLab backend TODO: replace this with Panda + cylinder scene construction "
            "using IsaacLab task/scene APIs while keeping the same observation/action schema."
        )

    def __getattr__(self, item: str):
        return getattr(self._env, item)
