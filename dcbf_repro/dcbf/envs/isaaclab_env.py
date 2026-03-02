from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    backend: str = "mock"
    num_objects: int = 20
    table_half_extent: float = 0.35
    object_radius: float = 0.05        # Paper: 5cm radius
    object_height: float = 0.20        # Paper: 20cm height
    max_episode_steps: int = 200
    max_action_step: float = 0.01
    goal_tolerance: float = 0.02
    fixed_z: float = 0.12
    tilt_threshold_deg: float = 15.0
    tilt_warning_margin_deg: float = 2.0
    stall_window: int = 20
    stall_movement_eps: float = 0.001
    stall_progress_eps: float = 0.0005
    contact_distance: float = 0.07
    tilt_gain: float = 1.8
    tilt_decay: float = 0.002
    # Per-object physical property randomization (paper Sec. VI)
    mass_range: Tuple[float, float] = (1.3, 2.0)
    static_friction_range: Tuple[float, float] = (0.5, 0.7)
    dynamic_friction_range: Tuple[float, float] = (0.3, 0.49)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
        # Handle tuple fields that may come as lists from YAML
        filtered = {}
        for k, v in data.items():
            if k in ("mass_range", "static_friction_range", "dynamic_friction_range") and isinstance(v, list):
                filtered[k] = tuple(v)
            else:
                filtered[k] = v
        return cls(**filtered)


class MockPandaClutterEnv:
    """Mock simulator with robot-object AND object-object contact interactions.

    Matches the paper's setup:
    - Per-object mass and friction randomisation (Sec. VI)
    - Object-object collision propagation (core to implicit interaction CBF)
    - Cylindrical bottles on a table with tilt dynamics
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)
        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.object_pos = np.zeros((cfg.num_objects, 3), dtype=np.float32)
        self.object_tilt_rad = np.zeros(cfg.num_objects, dtype=np.float32)
        # Per-object physical properties (paper Sec. VI)
        self.object_mass = np.ones(cfg.num_objects, dtype=np.float32)
        self.object_static_friction = np.ones(cfg.num_objects, dtype=np.float32) * 0.6
        self.object_dynamic_friction = np.ones(cfg.num_objects, dtype=np.float32) * 0.4
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
        # Randomise per-object physical properties (paper Sec. VI)
        self.object_mass[:] = self.rng.uniform(
            self.cfg.mass_range[0], self.cfg.mass_range[1], size=self.cfg.num_objects
        ).astype(np.float32)
        self.object_static_friction[:] = self.rng.uniform(
            self.cfg.static_friction_range[0], self.cfg.static_friction_range[1], size=self.cfg.num_objects
        ).astype(np.float32)
        self.object_dynamic_friction[:] = self.rng.uniform(
            self.cfg.dynamic_friction_range[0], self.cfg.dynamic_friction_range[1], size=self.cfg.num_objects
        ).astype(np.float32)
        self._prev_goal_dist = self._goal_distance()
        return self._get_obs(), {"seed": self._last_seed}

    def _sample_objects(self) -> np.ndarray:
        min_dist = self.cfg.object_radius * 2.2
        n = self.cfg.num_objects
        half = self.cfg.table_half_extent
        r = self.cfg.object_radius
        # Inset by object_radius so circles stay fully inside the table
        inner = half - r

        # For dense settings (many objects), use grid + jitter for reliable placement
        if n > 12:
            # Slightly more than diameter → 0.5 mm visual gap, avoids float32 touching
            grid_min_dist = self.cfg.object_radius * 2.02
            return self._sample_objects_grid(n, half, grid_min_dist)

        # Rejection sampling with restart for sparse settings
        sampled = np.zeros((n, 3), dtype=np.float32)
        placed = 0
        budget = 10000
        retries = 0
        while placed < n:
            if retries > budget:
                placed = 0
                retries = 0
                budget += 5000
                if budget > 100000:
                    raise RuntimeError("Cannot place objects without overlap in current table setup.")
            candidate = np.array(
                [self.rng.uniform(-inner, inner), self.rng.uniform(-inner, inner), 0.0],
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

    def _sample_objects_grid(self, n: int, half: float, min_dist: float) -> np.ndarray:
        """Hexagonal grid placement with adaptive jitter.

        Guarantees:
          1. Every circle center is within [-inner, inner] so circles stay
             strictly inside the table (inner = half - radius - pad).
          2. No two circles overlap: jitter magnitude is derived from the
             actual minimum pairwise distance of the chosen grid centres.
        """
        r = self.cfg.object_radius
        pad = 0.005  # 5 mm inward padding — circles won't touch table edge
        inner = half - r - pad
        spacing = min_dist
        row_h = spacing * np.sqrt(3) / 2  # hex row height

        # --- build hex grid ------------------------------------------------
        grid = []
        y = -inner
        row_idx = 0
        while y <= inner:
            x_offset = (spacing / 2) if (row_idx % 2) else 0.0
            x = -inner + x_offset
            while x <= inner:
                grid.append((x, y))
                x += spacing
            y += row_h
            row_idx += 1
        grid = np.array(grid, dtype=np.float32)

        # --- soft EE / goal exclusion --------------------------------------
        valid = []
        for pos in grid:
            if np.linalg.norm(pos - self.ee_pos[:2]) < 0.06:
                continue
            if np.linalg.norm(pos - self.goal[:2]) < 0.06:
                continue
            valid.append(pos)
        valid = np.array(valid, dtype=np.float32) if valid else grid[:0]
        if len(valid) < n:
            valid = grid  # fall back to all grid points
        if len(valid) < n:
            raise RuntimeError(
                f"Cannot place {n} objects: hex grid has {len(valid)} valid slots "
                f"(table_half_extent={half}, min_dist={min_dist:.3f})"
            )

        # --- choose n positions from the grid ------------------------------
        chosen_idx = self.rng.choice(len(valid), size=n, replace=False)
        sampled = np.zeros((n, 3), dtype=np.float32)
        for i, idx in enumerate(chosen_idx):
            sampled[i, :2] = valid[idx]

        # --- adaptive jitter ----------------------------------------------
        # Compute the minimum pairwise distance among chosen centres;
        # jitter must not exceed the "spare" margin above min_dist.
        if n > 1:
            pts = sampled[:, :2]
            diff = pts[:, None, :] - pts[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_pair = float(d.min())
            # Each of the two neighbours can shift by at most margin/2
            max_jitter = max(0.0, (min_pair - min_dist) * 0.45)
        else:
            max_jitter = spacing * 0.15

        if max_jitter > 1e-4:
            for i in range(n):
                jittered = sampled[i, :2] + self.rng.uniform(
                    -max_jitter, max_jitter, size=2
                ).astype(np.float32)
                jittered = np.clip(jittered, -inner, inner)
                # Verify this jittered position doesn't overlap any other
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    if np.linalg.norm(jittered - sampled[j, :2]) < min_dist:
                        ok = False
                        break
                if ok:
                    sampled[i, :2] = jittered

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
        """Apply robot-object contacts AND object-object collision propagation.

        This is critical for the paper's core claim: the CBF must implicitly
        encode inter-object interaction effects on safety.  Without object-object
        collisions the 'implicit interaction' insight cannot be validated.
        """
        ee_xy = self.ee_pos[:2]
        contact_dist = self.cfg.contact_distance
        obj_collision_dist = self.cfg.object_radius * 2.0  # diameter
        ref_mass = float(np.mean(self.cfg.mass_range))  # reference mass for scaling

        # --- Phase 1: Robot-object contacts ---
        push_vectors = np.zeros((self.cfg.num_objects, 2), dtype=np.float32)
        for idx in range(self.cfg.num_objects):
            rel = self.object_pos[idx, :2] - ee_xy
            dist = float(np.linalg.norm(rel))
            if dist < contact_dist and dist > 1e-6:
                contact_scale = (contact_dist - dist) / max(contact_dist, 1e-6)
                # Lighter objects are pushed more, heavier objects less
                mass_factor = ref_mass / max(float(self.object_mass[idx]), 0.1)
                # Friction reduces push magnitude
                friction_factor = 1.0 - 0.3 * float(self.object_dynamic_friction[idx])
                push = action * (0.8 + 0.2 * contact_scale) * mass_factor * friction_factor
                push_vectors[idx] = push

        # --- Phase 2: Object-object collision propagation ---
        # When robot pushes object A into object B, B should also be affected.
        # This cascade is what makes the "implicit interaction" concept meaningful.
        max_propagation_iters = 3  # limit cascade depth
        for _ in range(max_propagation_iters):
            new_pushes = np.zeros_like(push_vectors)
            for i in range(self.cfg.num_objects):
                if np.linalg.norm(push_vectors[i]) < 1e-7:
                    continue
                # Move object i tentatively
                tentative_pos_i = self.object_pos[i, :2] + push_vectors[i]
                for j in range(self.cfg.num_objects):
                    if i == j:
                        continue
                    rel_ij = self.object_pos[j, :2] - tentative_pos_i
                    dist_ij = float(np.linalg.norm(rel_ij))
                    if dist_ij < obj_collision_dist and dist_ij > 1e-6:
                        # Transfer fraction of push from i to j
                        overlap = (obj_collision_dist - dist_ij) / obj_collision_dist
                        direction = rel_ij / dist_ij
                        mass_ratio = float(self.object_mass[i]) / max(float(self.object_mass[j]), 0.1)
                        transfer = direction * overlap * float(np.linalg.norm(push_vectors[i])) * 0.5 * mass_ratio
                        new_pushes[j] += transfer
            # Add cascaded pushes (attenuated)
            if np.max(np.linalg.norm(new_pushes, axis=1)) < 1e-7:
                break
            push_vectors += new_pushes * 0.6  # attenuation factor

        # --- Phase 3: Apply pushes and compute tilt changes ---
        for idx in range(self.cfg.num_objects):
            push = push_vectors[idx]
            push_mag = float(np.linalg.norm(push))
            if push_mag < 1e-7:
                continue
            self.object_pos[idx, :2] += push
            self.object_pos[idx, :2] = np.clip(
                self.object_pos[idx, :2], -self.cfg.table_half_extent, self.cfg.table_half_extent
            )
            # Tilt depends on push magnitude, mass (heavier = harder to tip), friction
            mass_tilt_factor = ref_mass / max(float(self.object_mass[idx]), 0.1)
            friction_tilt_factor = 1.0 - 0.2 * float(self.object_static_friction[idx])
            rel_to_ee = self.object_pos[idx, :2] - ee_xy
            dist_to_ee = float(np.linalg.norm(rel_to_ee))
            contact_scale = max(0.0, (contact_dist - dist_to_ee) / max(contact_dist, 1e-6))
            tilt_delta = (
                self.cfg.tilt_gain * push_mag * (1.0 + 0.5 * contact_scale)
                * mass_tilt_factor * friction_tilt_factor
            )
            self.object_tilt_rad[idx] += tilt_delta

        # Natural tilt decay (gravity restoring force)
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
            "object_mass": self.object_mass.copy(),
            "object_static_friction": self.object_static_friction.copy(),
            "object_dynamic_friction": self.object_dynamic_friction.copy(),
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
        if "object_mass" in snap:
            self.object_mass[:] = np.asarray(snap["object_mass"], dtype=np.float32)
        if "object_static_friction" in snap:
            self.object_static_friction[:] = np.asarray(snap["object_static_friction"], dtype=np.float32)
        if "object_dynamic_friction" in snap:
            self.object_dynamic_friction[:] = np.asarray(snap["object_dynamic_friction"], dtype=np.float32)
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
