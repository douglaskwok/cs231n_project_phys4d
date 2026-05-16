"""Autoregressive state predictor: past states + visual features -> next state."""

from __future__ import annotations

import torch
import torch.nn as nn

from .feature_encoder import build_feature_encoder


class VisualDynamicsModel(nn.Module):
    """Predict delta state at t+1 from K past states and a visual feature vector."""

    def __init__(
        self,
        state_dim: int = 10,
        history: int = 8,
        visual_dim: int = 64,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.history = history
        self.state_dim = state_dim
        self.encoder = build_feature_encoder(out_dim=visual_dim)
        in_dim = history * state_dim + visual_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, state_dim),
        )

    def forward(
        self,
        state_history: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state_history: (B, K, state_dim)
            image: (B, 3, H, W) masked crop
        Returns:
            delta state (B, state_dim)
        """

        b, k, d = state_history.shape
        if k != self.history:
            raise ValueError(f"expected history {self.history}, got {k}")
        visual = self.encoder(image)
        flat = state_history.reshape(b, k * d)
        return self.head(torch.cat([flat, visual], dim=-1))

    def predict_next(
        self,
        state_history: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Return absolute next state = last + delta."""

        delta = self.forward(state_history, image)
        return state_history[:, -1, :] + delta
