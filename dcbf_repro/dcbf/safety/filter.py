from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from dcbf.safety.compose import BarrierScorer
from dcbf.utils.geometry import HistoryView


def clip_action(action_xy: np.ndarray, max_step: float) -> np.ndarray:
    action = np.asarray(action_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(action))
    if norm > max_step and norm > 1e-9:
        action = action / norm * max_step
    return action


def nominal_go_to_goal(obs: Dict[str, np.ndarray], max_step: float) -> np.ndarray:
    delta = np.asarray(obs["goal_xy"], dtype=np.float32) - np.asarray(obs["ee_pos"], dtype=np.float32)[:2]
    return clip_action(delta, max_step=max_step)


def nominal_apf(obs: Dict[str, np.ndarray], max_step: float, repulsive_gain: float = 0.003, influence_dist: float = 0.08):
    ee_xy = np.asarray(obs["ee_pos"], dtype=np.float32)[:2]
    goal_xy = np.asarray(obs["goal_xy"], dtype=np.float32)
    objects_xy = np.asarray(obs["objects_pos"], dtype=np.float32)[:, :2]

    attractive = goal_xy - ee_xy
    repulsive = np.zeros(2, dtype=np.float32)
    for obj_xy in objects_xy:
        rel = ee_xy - obj_xy
        dist = float(np.linalg.norm(rel))
        if dist < influence_dist and dist > 1e-6:
            repulsive += repulsive_gain * (1.0 / dist - 1.0 / influence_dist) * (rel / (dist**3))
    action = attractive + repulsive
    return clip_action(action, max_step=max_step)


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
