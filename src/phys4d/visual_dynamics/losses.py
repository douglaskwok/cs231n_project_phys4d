"""Training losses for visual dynamics (project.md Phase 4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on unit quaternions [qx,qy,qz,qw]."""
    qp = F.normalize(q_pred, dim=-1)
    qg = F.normalize(q_gt, dim=-1)
    dot = torch.abs((qp * qg).sum(dim=-1)).clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    return (angle**2).mean()


def state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_state: position MSE + quaternion geodesic (+ velocity MSE)."""
    pos_mse = F.mse_loss(pred[..., :3], target[..., :3])
    quat_loss = quat_geodesic_loss(pred[..., 3:7], target[..., 3:7])
    vel_mse = F.mse_loss(pred[..., 7:10], target[..., 7:10])
    return pos_mse + quat_loss + 0.25 * vel_mse


def physics_energy_penalty(states: torch.Tensor, *, g: float = 9.81) -> torch.Tensor:
    """Optional: penalize increasing vertical energy (COM z + speed)."""
    z = states[..., 2]
    speed = torch.linalg.norm(states[..., 7:10], dim=-1)
    energy = g * z + 0.5 * speed**2
    dE = energy[:, 1:] - energy[:, :-1]
    return F.relu(dE).mean()
