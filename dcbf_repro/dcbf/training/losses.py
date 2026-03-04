from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    gamma: float = 0.1       # 衰减率: B_{t+1} >= (1-γ)B_t, γ越小约束越紧
    sigma: float = 0.02      # 鲁棒性裕度
    eta_s: float = 1.0       # safe 分类损失权重
    eta_u: float = 5.0       # unsafe 分类损失权重
    eta_d: float = 1.0       # 导数损失权重（需较大值约束边界斜率）

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

    # Paper Eq.8: L_s penalises -B(r^t, O^{t-1}) on safe transitions (enforce b_t >= 0)
    # Paper Eq.9: L_u penalises  B(r^t, O^{t-1}) on unsafe transitions (enforce b_t <= 0)
    l_s = masked_relu_mean(-b_t, safe_mask)
    l_u = masked_relu_mean(b_t, unsafe_mask)
    # Paper Eq.10: discrete CBF decrease condition
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
