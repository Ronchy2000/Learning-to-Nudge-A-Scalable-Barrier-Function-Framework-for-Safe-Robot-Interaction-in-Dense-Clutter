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

    # ------------------------------------------------------------------
    #  Object placement
    # ------------------------------------------------------------------

    # Pre-computed layout for N=40 (half=0.35, r=0.05).
    # Generated via Metropolis-Hastings MCMC targeting maximum
    # nearest-neighbor distance variance (organic: some touching,
    # some with visible gaps — matches paper Fig.3 style).
    # nn_std=0.01567, nn_range=[0.102, 0.204], no overlap, inside table.
    _FIXED_POS_40 = np.array([
        [-0.081109, -0.009490],
        [-0.190421, +0.185277],
        [+0.033352, -0.003557],
        [+0.234069, -0.201984],
        [-0.132857, -0.098988],
        [+0.187336, -0.293772],
        [+0.294696, +0.088458],
        [+0.138887, -0.006304],
        [-0.294464, -0.177340],
        [+0.187777, -0.109313],
        [+0.075639, +0.289246],
        [-0.291355, -0.002578],
        [-0.047786, -0.293828],
        [+0.294369, +0.294083],
        [-0.037277, +0.288377],
        [-0.280998, +0.292818],
        [+0.294560, -0.117692],
        [+0.120006, +0.188301],
        [-0.024777, -0.098783],
        [+0.062214, -0.292912],
        [-0.193236, -0.192501],
        [+0.125331, -0.209971],
        [+0.021153, -0.195848],
        [-0.138323, +0.091067],
        [-0.247098, +0.090757],
        [-0.030278, +0.085701],
        [-0.087466, +0.181987],
        [-0.139020, +0.279935],
        [-0.254682, -0.287296],
        [+0.082472, -0.108734],
        [+0.016590, +0.194836],
        [+0.195140, +0.116344],
        [+0.252868, -0.018583],
        [-0.292823, +0.183647],
        [-0.151753, -0.291583],
        [-0.235992, -0.092131],
        [-0.090278, -0.198004],
        [+0.080622, +0.086948],
        [+0.293177, -0.291074],
        [-0.184890, -0.000274],
    ], dtype=np.float32)

    def _sample_objects(self) -> np.ndarray:
        """Place N non-overlapping circles inside the table.

        - N >= 40  → use pre-computed hardcoded layout (instant).
        - small table (tight packing) → systematic corner placement + jitter.
        - otherwise → pure rejection sampling (continuous random).
        All circles stay strictly inside the table boundary.
        """
        n = self.cfg.num_objects
        r = self.cfg.object_radius
        half = self.cfg.table_half_extent
        pad = 0.005
        inner = half - r - pad
        min_sep = 2 * r + 0.002

        # --- N=40: hardcoded layout (instant) -----------------------------
        if n >= 40:
            out = np.zeros((n, 3), dtype=np.float32)
            out[:n, :2] = self._FIXED_POS_40[:n]
            return out

        # --- Small table: packing density too high for rejection sampling --
        # When the available area is tight, systematic placement is needed.
        # Compute packing density: if average available area per object is
        # less than 4× the exclusion area, use grid-based placement.
        exclusion_area = np.pi * (min_sep / 2) ** 2
        total_area = (2 * inner) ** 2
        if total_area < n * exclusion_area * 3.5:
            return self._sample_objects_tight(n, inner, min_sep)

        # --- Normal table: rejection sampling (continuous random) ---------
        for _restart in range(50):
            pts = np.zeros((n, 2), dtype=np.float32)
            placed = 0
            for _ in range(300_000):
                xy = self.rng.uniform(-inner, inner, size=2).astype(np.float32)
                if placed > 0:
                    if np.linalg.norm(pts[:placed] - xy, axis=1).min() < min_sep:
                        continue
                pts[placed] = xy
                placed += 1
                if placed == n:
                    break
            if placed == n:
                out = np.zeros((n, 3), dtype=np.float32)
                out[:, :2] = pts
                return out

        raise RuntimeError(
            f"Cannot place {n} objects (r={r:.3f}m) in "
            f"table_half_extent={half:.3f}m after 50 restarts."
        )

    def _sample_objects_tight(self, n: int, inner: float, min_sep: float) -> np.ndarray:
        """Place objects in a tight space using grid + jitter + rejection.

        Generate candidate positions on a centred grid covering the inner area,
        then pick N with random jitter, validating no overlaps.
        """
        # Build centred grid with spacing = min_sep
        n_per_dim = int(np.floor(2 * inner / min_sep)) + 1
        half_span = (n_per_dim - 1) * min_sep / 2
        coords_1d = np.linspace(-half_span, half_span, n_per_dim)
        candidates = []
        for x in coords_1d:
            for y in coords_1d:
                if abs(x) <= inner + 1e-9 and abs(y) <= inner + 1e-9:
                    candidates.append((x, y))
        candidates = np.array(candidates, dtype=np.float32)

        if len(candidates) < n:
            raise RuntimeError(
                f"Cannot fit {n} objects in tight table "
                f"(inner={inner:.3f}m, min_sep={min_sep:.3f}m, "
                f"only {len(candidates)} grid candidates)."
            )

        # Try multiple times: shuffle grid points, pick N, add jitter
        max_jitter = max(0.0, inner - half_span - 0.001)
        for _ in range(200):
            order = self.rng.permutation(len(candidates))
            pts = np.zeros((n, 2), dtype=np.float32)
            placed = 0
            for idx in order:
                base = candidates[idx].copy()
                if max_jitter > 0:
                    jitter = self.rng.uniform(-max_jitter, max_jitter, size=2).astype(np.float32)
                    xy = np.clip(base + jitter, -inner, inner)
                else:
                    xy = base
                if placed > 0:
                    if np.linalg.norm(pts[:placed] - xy, axis=1).min() < min_sep:
                        continue
                pts[placed] = xy
                placed += 1
                if placed == n:
                    break
            if placed == n:
                out = np.zeros((n, 3), dtype=np.float32)
                out[:, :2] = pts
                return out

        raise RuntimeError(
            f"Cannot place {n} objects in tight table "
            f"(inner={inner:.3f}m, min_sep={min_sep:.3f}m) after 200 attempts."
        )

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
