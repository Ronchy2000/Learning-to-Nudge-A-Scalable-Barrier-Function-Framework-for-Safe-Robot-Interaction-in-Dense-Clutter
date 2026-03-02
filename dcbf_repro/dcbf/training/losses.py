from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    gamma: float = 0.98
    sigma: float = 0.02
    eta_s: float = 1.0
    eta_u: float = 1.0
    eta_d: float = 0.2

    @classmethod
    def from_dict(cls, cfg):
        return cls(**cfg)


def masked_relu_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() < 1:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return F.relu(values[mask]).mean()


def compute_dcbf_losses(
    b_t: torch.Tensor,
    b_tp1: torch.Tensor,
    safe_label: torch.Tensor,
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    L_s = mean(ReLU(-B)) on safe samples
    L_u = mean(ReLU(B)) on unsafe samples
    L_d = mean(ReLU((1-gamma)*B_t - B_{t+1} + sigma))
    """
    safe_mask = safe_label > 0.5
    unsafe_mask = ~safe_mask

    l_s = masked_relu_mean(-b_tp1, safe_mask)
    l_u = masked_relu_mean(b_tp1, unsafe_mask)
    drift_term = (1.0 - cfg.gamma) * b_t - b_tp1 + cfg.sigma
    l_d = torch.relu(drift_term).mean()

    total = cfg.eta_s * l_s + cfg.eta_u * l_u + cfg.eta_d * l_d
    return {
        "total": total,
        "l_s": l_s,
        "l_u": l_u,
        "l_d": l_d,
        "drift_violation_ratio": (drift_term > 0.0).float().mean(),
    }
