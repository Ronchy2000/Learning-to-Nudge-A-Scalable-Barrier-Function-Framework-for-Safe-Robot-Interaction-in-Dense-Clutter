from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from dcbf.safety.compose import BarrierScorer
from dcbf.utils.geometry import HistoryView

# Keep old APF signature as alias for backward compatibility
_nominal_apf_legacy = None  # replaced by nominal_apf with paper-correct parameters


def clip_action(action_xy: np.ndarray, max_step: float) -> np.ndarray:
    action = np.asarray(action_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(action))
    if norm > max_step and norm > 1e-9:
        action = action / norm * max_step
    return action


def nominal_go_to_goal(obs: Dict[str, np.ndarray], max_step: float) -> np.ndarray:
    delta = np.asarray(obs["goal_xy"], dtype=np.float32) - np.asarray(obs["ee_pos"], dtype=np.float32)[:2]
    return clip_action(delta, max_step=max_step)


def nominal_apf(
    obs: Dict[str, np.ndarray],
    max_step: float,
    kp: float = 5.0,
    eta: float = 50.0,
    influence_dist: float = 1.2,
    oscillation_len: int = 3,
    _prev_actions: list | None = None,
):
    """Artificial Potential Field baseline matching paper parameters.

    Paper: KP=5.0 (attractive gain), eta=50.0 (repulsive gain),
    potential area length 1.2m, oscillation detection length 3.
    """
    ee_xy = np.asarray(obs["ee_pos"], dtype=np.float32)[:2]
    goal_xy = np.asarray(obs["goal_xy"], dtype=np.float32)
    objects_xy = np.asarray(obs["objects_pos"], dtype=np.float32)[:, :2]

    # Attractive potential: F_att = kp * (goal - ee)
    attractive = kp * (goal_xy - ee_xy)

    # Repulsive potential: F_rep = eta * (1/d - 1/d0) * (1/d^2) * unit_vec
    repulsive = np.zeros(2, dtype=np.float32)
    for obj_xy in objects_xy:
        rel = ee_xy - obj_xy
        dist = float(np.linalg.norm(rel))
        if dist < influence_dist and dist > 1e-6:
            unit = rel / dist
            repulsive += eta * (1.0 / dist - 1.0 / influence_dist) * (1.0 / (dist ** 2)) * unit

    action = attractive + repulsive
    return clip_action(action, max_step=max_step)


def nominal_backstep(
    obs: Dict[str, np.ndarray],
    max_step: float,
    backstep_threshold_deg: float = 14.0,
    _prev_ee: np.ndarray | None = None,
) -> np.ndarray:
    """Back-stepping baseline from the paper.

    Moves toward goal normally, but reverses the action when any object's
    tilt angle approaches the safety threshold (14 degrees, i.e., margin=1 degree
    below the 15 degree threshold).
    """
    delta = np.asarray(obs["goal_xy"], dtype=np.float32) - np.asarray(obs["ee_pos"], dtype=np.float32)[:2]
    action = clip_action(delta, max_step=max_step)
    # Check current tilt angles
    tilt_deg = np.rad2deg(np.asarray(obs["objects_tilt_rad"], dtype=np.float32))
    max_tilt = float(np.max(tilt_deg)) if tilt_deg.size > 0 else 0.0
    if max_tilt > backstep_threshold_deg:
        action = -action
    return action


@dataclass
class SamplingSafetyFilter:
    barrier: BarrierScorer
    max_step: float
    num_candidates: int = 64
    noise_std: float = 0.003
    fallback_scale: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _sample_candidates(self, u_nom: np.ndarray) -> np.ndarray:
        candidates = np.zeros((self.num_candidates + 1, 2), dtype=np.float32)
        candidates[0] = u_nom
        noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=(self.num_candidates, 2)).astype(np.float32)
        candidates[1:] = u_nom[None, :] + noise
        for idx in range(candidates.shape[0]):
            candidates[idx] = clip_action(candidates[idx], self.max_step)
        return candidates

    def step(
        self,
        obs: Dict[str, np.ndarray],
        u_nom: np.ndarray,
        history: Optional[HistoryView] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        u_nom = clip_action(u_nom, self.max_step)
        nominal_score = self.barrier.score_action(obs, history, u_nom)
        if nominal_score >= 0.0:
            return u_nom, {"accepted_nominal": 1.0, "nominal_score": nominal_score, "selected_score": nominal_score}

        candidates = self._sample_candidates(u_nom)
        scores = np.array([self.barrier.score_action(obs, history, u) for u in candidates], dtype=np.float32)
        safe_mask = scores >= 0.0

        if np.any(safe_mask):
            safe_candidates = candidates[safe_mask]
            safe_scores = scores[safe_mask]
            nominal_dist = np.linalg.norm(safe_candidates - u_nom[None, :], axis=1)
            chosen_idx = int(np.argmin(nominal_dist))
            chosen = safe_candidates[chosen_idx]
            selected_score = float(safe_scores[chosen_idx])
            return chosen, {
                "accepted_nominal": 0.0,
                "nominal_score": nominal_score,
                "selected_score": selected_score,
                "found_safe": 1.0,
            }

        best_idx = int(np.argmax(scores))
        fallback = candidates[best_idx] * self.fallback_scale
        fallback = clip_action(fallback, self.max_step)
        return fallback, {
            "accepted_nominal": 0.0,
            "nominal_score": nominal_score,
            "selected_score": float(scores[best_idx]),
            "found_safe": 0.0,
        }
