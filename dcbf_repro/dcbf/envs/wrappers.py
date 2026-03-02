from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)


class ActionScalingWrapper(EnvWrapper):
    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        scaled_action = np.asarray(action, dtype=np.float32) * self.scale
        return self.env.step(scaled_action)


class PlanarConstraintWrapper(EnvWrapper):
    def __init__(self, env, fixed_z: Optional[float] = None):
        super().__init__(env)
        self.fixed_z = fixed_z

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.fixed_z is not None:
            obs["ee_pos"][2] = self.fixed_z
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.fixed_z is not None:
            obs["ee_pos"][2] = self.fixed_z
        return obs, reward, terminated, truncated, info


@dataclass
class EpisodeLog:
    steps: int = 0
    success: bool = False
    violation: bool = False
    stall: bool = False
    max_tilt_deg: float = 0.0
    min_tilt_deg: float = float("inf")
    mean_tilt_deg_sum: float = 0.0

    def update(self, info: Dict[str, Any]):
        self.steps += 1
        self.success = self.success or bool(info.get("success", False))
        self.violation = self.violation or bool(info.get("violation", False))
        self.stall = self.stall or bool(info.get("stall", False))
        self.max_tilt_deg = max(self.max_tilt_deg, float(info.get("tilt_max_deg", 0.0)))
        self.min_tilt_deg = min(self.min_tilt_deg, float(info.get("tilt_min_deg", 0.0)))
        self.mean_tilt_deg_sum += float(info.get("tilt_mean_deg", 0.0))

    @property
    def mean_tilt_deg(self) -> float:
        return self.mean_tilt_deg_sum / max(self.steps, 1)


class LoggingWrapper(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_log = EpisodeLog()
        self.last_log: Optional[EpisodeLog] = None

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self.current_log.steps > 0:
            self.last_log = self.current_log
        self.current_log = EpisodeLog()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_log.update(info)
        if terminated or truncated:
            self.last_log = self.current_log
        return obs, reward, terminated, truncated, info
