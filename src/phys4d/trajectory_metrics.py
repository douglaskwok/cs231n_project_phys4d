"""Trajectory metrics for train/test physics extrapolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .differentiable_bounce import BounceConfig, fit_restitution, simulate_bounce
from .poses import ObjectPoseTrajectory


@dataclass(frozen=True)
class TrajectorySplitMetrics:
    """MSE on train vs held-out frames after fitting restitution on train only."""

    train_frames: tuple[int, int]
    test_frames: tuple[int, int]
    recovered_restitution: float
    train_mse: float
    test_mse: float
    full_mse: float


def fit_restitution_on_train_frames(
    trajectory: ObjectPoseTrajectory,
    *,
    bounce_cfg: BounceConfig,
    train_end_frame: int,
    z0: float,
) -> TrajectorySplitMetrics:
    """Fit ``e`` using z at frames ``0..train_end_frame`` (inclusive), eval on the rest."""

    z_all = trajectory.z_series()
    # CSV rows are post-step states; prepend z0 to align with simulate_bounce indexing.
    observed = torch.tensor([z0] + z_all.tolist(), dtype=torch.float32)

    train_end = int(train_end_frame)
    train_obs = observed[: train_end + 2]  # frames 0..train_end inclusive => +2 in z index
    if train_obs.shape[0] < 3:
        raise ValueError("Need at least 2 train frames for fitting")

    train_steps = train_obs.shape[0] - 1
    train_cfg = BounceConfig(
        steps=train_steps,
        dt=bounce_cfg.dt,
        z0=bounce_cfg.z0,
        vz0=bounce_cfg.vz0,
        gravity=bounce_cfg.gravity,
        radius=bounce_cfg.radius,
        ground_z=bounce_cfg.ground_z,
        restitution_min=bounce_cfg.restitution_min,
        restitution_max=bounce_cfg.restitution_max,
    )
    result = fit_restitution(train_obs, config=train_cfg, initial_raw=-1.0, lr=0.08, iterations=600)

    with torch.no_grad():
        full_pred = simulate_bounce(torch.tensor(result.final_restitution), bounce_cfg)
        train_pred = simulate_bounce(torch.tensor(result.final_restitution), train_cfg)
        train_mse = torch.mean((train_pred - train_obs) ** 2).item()
        full_mse = torch.mean((full_pred - observed) ** 2).item()

        test_mask = torch.ones(observed.shape[0], dtype=torch.bool)
        test_mask[: train_obs.shape[0]] = False
        if torch.any(test_mask):
            test_mse = torch.mean((full_pred[test_mask] - observed[test_mask]) ** 2).item()
        else:
            test_mse = float("nan")

    return TrajectorySplitMetrics(
        train_frames=(0, train_end_frame),
        test_frames=(train_end_frame + 1, bounce_cfg.steps),
        recovered_restitution=result.final_restitution,
        train_mse=train_mse,
        test_mse=test_mse,
        full_mse=full_mse,
    )
