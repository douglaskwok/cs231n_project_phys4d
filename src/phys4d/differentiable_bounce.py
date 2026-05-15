"""Minimal differentiable sphere-ground bounce simulator.

The simulator is intentionally small: it tracks only vertical position and
velocity for a single sphere, while keeping restitution differentiable through
the post-contact velocity update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class BounceConfig:
    """Configuration for a single vertical sphere-ground bounce."""

    dt: float = 1.0 / 60.0
    steps: int = 90
    gravity: float = -9.81
    radius: float = 0.1
    ground_z: float = 0.0
    z0: float = 1.6
    vz0: float = 0.0
    restitution_min: float = 0.0
    restitution_max: float = 1.0


@dataclass(frozen=True)
class RecoveryResult:
    """Summary returned by restitution fitting."""

    initial_restitution: float
    final_restitution: float
    initial_loss: float
    final_loss: float
    losses: List[float]


def _scalar_like(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(value, dtype=reference.dtype, device=reference.device)


def restitution_from_raw(
    raw: torch.Tensor,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    """Map an unconstrained scalar to a valid restitution value."""

    if max_value <= min_value:
        raise ValueError("max_value must be greater than min_value")
    return min_value + (max_value - min_value) * torch.sigmoid(raw)


def simulate_bounce(
    restitution: torch.Tensor,
    config: Optional[BounceConfig] = None,
    return_velocity: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Roll out vertical sphere-ground dynamics.

    Args:
        restitution: Scalar tensor in [0, 1]. Gradients can flow to it.
        config: Simulation constants.
        return_velocity: If true, return both position and velocity histories.

    Returns:
        A tensor of z positions with shape ``[steps + 1]``. If
        ``return_velocity`` is true, also returns vertical velocities.
    """

    cfg = config or BounceConfig()
    if cfg.steps < 1:
        raise ValueError("steps must be at least 1")

    restitution = torch.as_tensor(restitution)
    dt = _scalar_like(cfg.dt, restitution)
    gravity = _scalar_like(cfg.gravity, restitution)
    contact_z = _scalar_like(cfg.ground_z + cfg.radius, restitution)

    z = _scalar_like(cfg.z0, restitution)
    vz = _scalar_like(cfg.vz0, restitution)

    z_values = [z]
    vz_values = [vz]

    for _ in range(cfg.steps):
        vz_free = vz + gravity * dt
        z_free = z + vz_free * dt

        hit_ground = (z_free < contact_z) & (vz_free < 0)
        z = torch.where(hit_ground, contact_z, z_free)
        vz = torch.where(hit_ground, -restitution * vz_free, vz_free)

        z_values.append(z)
        vz_values.append(vz)

    z_history = torch.stack(z_values)
    if return_velocity:
        return z_history, torch.stack(vz_values)
    return z_history


def fit_restitution(
    observed_z: torch.Tensor,
    config: Optional[BounceConfig] = None,
    initial_raw: Optional[float] = None,
    lr: float = 0.08,
    iterations: int = 400,
) -> RecoveryResult:
    """Recover restitution from an observed z trajectory using autograd."""

    cfg = config or BounceConfig()
    observed_z = torch.as_tensor(observed_z, dtype=torch.float32)
    if observed_z.shape != (cfg.steps + 1,):
        raise ValueError(
            f"observed_z must have shape ({cfg.steps + 1},), got {tuple(observed_z.shape)}"
        )

    raw_value = 0.0 if initial_raw is None else initial_raw
    raw = torch.tensor(raw_value, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    with torch.no_grad():
        initial_restitution = restitution_from_raw(
            raw, cfg.restitution_min, cfg.restitution_max
        ).item()
        initial_pred = simulate_bounce(
            torch.tensor(initial_restitution, dtype=torch.float32), cfg
        )
        initial_loss = torch.mean((initial_pred - observed_z) ** 2).item()

    losses: List[float] = []
    for _ in range(iterations):
        optimizer.zero_grad()
        restitution = restitution_from_raw(raw, cfg.restitution_min, cfg.restitution_max)
        predicted_z = simulate_bounce(restitution, cfg)
        loss = torch.mean((predicted_z - observed_z) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        final_restitution = restitution_from_raw(
            raw, cfg.restitution_min, cfg.restitution_max
        ).item()
        final_pred = simulate_bounce(torch.tensor(final_restitution), cfg)
        final_loss = torch.mean((final_pred - observed_z) ** 2).item()

    return RecoveryResult(
        initial_restitution=initial_restitution,
        final_restitution=final_restitution,
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=losses,
    )
