"""State-only baseline: first-N pose vectors → physics parameters."""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseHistoryMLP(nn.Module):
    """Predict (restitution, mass_kg, drop_z_m) from stacked pose features."""

    def __init__(
        self,
        *,
        history_steps: int = 16,
        features_per_step: int = 6,
        hidden: int = 128,
        num_params: int = 3,
    ) -> None:
        super().__init__()
        in_dim = history_steps * features_per_step
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_params),
        )
        self.register_buffer("low", torch.tensor([0.3, 0.1, 1.0]))
        self.register_buffer("high", torch.tensor([0.95, 2.0, 2.2]))
        self.history_steps = history_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 6) with [x,y,z,vx,vy,vz] per step."""
        b, t, f = x.shape
        if t < self.history_steps:
            pad = torch.zeros(b, self.history_steps - t, f, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x = x[:, : self.history_steps].reshape(b, -1)
        raw = self.net(x)
        return self.low + torch.sigmoid(raw) * (self.high - self.low)
