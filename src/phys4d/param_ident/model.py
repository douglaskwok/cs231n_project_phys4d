"""Multi-view CNN + temporal Transformer → physics parameter vector."""

from __future__ import annotations

import torch
import torch.nn as nn

from phys4d.param_ident.encoder import build_view_encoder


class MultiViewParamPredictor(nn.Module):
    """Predict restitution, mass_kg, drop_z_m from masked multi-view crops."""

    def __init__(
        self,
        *,
        num_params: int = 3,
        view_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = build_view_encoder(out_dim=view_dim)
        self.view_proj = nn.Linear(view_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.view_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_params),
        )
        self._init_bounds()

    def _init_bounds(self) -> None:
        # Soft bounds via sigmoid scaling in forward (restitution, mass, drop_z ranges).
        self.register_buffer("low", torch.tensor([0.3, 0.1, 1.0]))
        self.register_buffer("high", torch.tensor([0.95, 2.0, 2.2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V, 3, H, W) masked RGB crops.
        Returns:
            (B, num_params) in physical units.
        """
        b, t, v, c, h, w = x.shape
        flat = x.reshape(b * t * v, c, h, w)
        feats = self.encoder(flat).reshape(b, t, v, -1)
        feats = self.view_proj(feats)
        # Temporal: mean over views per timestep, then Transformer over T.
        per_t = feats.mean(dim=2)
        temporal_out = self.temporal(per_t)
        # Multi-view attention at last timestep.
        last_views = feats[:, -1]
        pooled, _ = self.view_attn(last_views, last_views, last_views)
        pooled = pooled.mean(dim=1)
        fused = temporal_out[:, -1] + pooled
        raw = self.head(fused)
        return self.low + torch.sigmoid(raw) * (self.high - self.low)
