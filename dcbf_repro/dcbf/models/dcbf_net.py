from __future__ import annotations

import torch
import torch.nn as nn


class DCBFNet(nn.Module):
    def __init__(
        self,
        robot_dim: int = 3,
        object_dim: int = 4,
        history_len: int = 10,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.robot_dim = robot_dim
        self.object_dim = object_dim
        self.history_len = history_len

        self.hist_encoder = nn.LSTM(
            input_size=object_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, robot_feat: torch.Tensor, object_hist_feat: torch.Tensor) -> torch.Tensor:
        """
        robot_feat: (B, robot_dim)
        object_hist_feat: (B, T, object_dim)
        return: (B, 1) barrier value B_i
        """
        _, (h_n, _) = self.hist_encoder(object_hist_feat)
        h_hist = h_n[-1]
        h_robot = self.robot_encoder(robot_feat)
        h = torch.cat([h_hist, h_robot], dim=-1)
        return self.head(h)
