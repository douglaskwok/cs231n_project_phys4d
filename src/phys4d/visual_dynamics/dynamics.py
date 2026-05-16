"""Object-token dynamics: past states + fixed t=0 visual feature → SE(3) delta."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectTokenDynamics(nn.Module):
    """
    Per-object token = concat(flattened state history, visual feature).
    Transformer self-attention over objects (N=1 for sphere bounce).
  Outputs delta in state space (10D); integrate as last + delta.
    """

    def __init__(
        self,
        *,
        state_dim: int = 10,
        history: int = 8,
        visual_dim: int = 64,
        hidden: int = 128,
        num_objects: int = 1,
        n_layers: int = 2,
        n_heads: int = 4,
        use_visual: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.history = history
        self.num_objects = num_objects
        self.use_visual = use_visual
        token_in = history * state_dim + (visual_dim if use_visual else 0)
        self.token_proj = nn.Linear(token_in, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, state_dim),
        )

    def forward(
        self,
        state_history: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state_history: (B, N, K, state_dim)
            visual_feat: (B, N, visual_dim) — fixed at t=0
        Returns:
            delta: (B, N, state_dim)
        """
        b, n, k, d = state_history.shape
        flat = state_history.reshape(b, n, k * d)
        if self.use_visual:
            tokens = torch.cat([flat, visual_feat], dim=-1)
        else:
            tokens = flat
        x = self.token_proj(tokens)
        x = self.transformer(x)
        return self.delta_head(x)

    def predict_next(
        self,
        state_history: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate delta on position/velocity; keep quaternion on S3 (sphere has no spin)."""
        delta = self.forward(state_history, visual_feat)
        last = state_history[:, :, -1, :]
        nxt = last.clone()
        nxt[..., :3] = last[..., :3] + delta[..., :3]
        nxt[..., 7:10] = last[..., 7:10] + delta[..., 7:10]
        nxt[..., 3:7] = F.normalize(last[..., 3:7], dim=-1, eps=1e-6)
        return nxt
