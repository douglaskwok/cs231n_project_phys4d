"""Pose extrapolation baselines for the temporal train/test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .differentiable_bounce import BounceConfig, fit_restitution, simulate_bounce
from .poses import ObjectPoseTrajectory


@dataclass(frozen=True)
class ExtrapolationResult:
    name: str
    train_mse: float
    test_mse: float
    z_test_mse: float


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _align_z_series(trajectory: ObjectPoseTrajectory, z0: float) -> np.ndarray:
    """z at frames 0..T-1 with z0 prepended (matches bounce simulator indexing)."""

    z = trajectory.z_series()
    return np.concatenate([[z0], z], axis=0)


def constant_hold_baseline(
    trajectory: ObjectPoseTrajectory,
    *,
    train_end_frame: int,
    z0: float,
) -> ExtrapolationResult:
    z = _align_z_series(trajectory, z0)
    hold = z[train_end_frame + 1]
    pred = z.copy()
    pred[train_end_frame + 2 :] = hold
    train_mask = slice(0, train_end_frame + 2)
    test_mask = slice(train_end_frame + 2, None)
    return ExtrapolationResult(
        name="constant_hold",
        train_mse=_mse(pred[train_mask], z[train_mask]),
        test_mse=_mse(pred[test_mask], z[test_mask]),
        z_test_mse=_mse(pred[test_mask], z[test_mask]),
    )


def linear_delta_baseline(
    trajectory: ObjectPoseTrajectory,
    *,
    train_end_frame: int,
    z0: float,
) -> ExtrapolationResult:
    z = _align_z_series(trajectory, z0)
    pred = z.copy()
    if train_end_frame + 1 < 1:
        raise ValueError("Need at least one train frame for linear delta")
    dz = z[train_end_frame + 1] - z[train_end_frame]
    for i in range(train_end_frame + 2, len(z)):
        pred[i] = pred[i - 1] + dz
    train_mask = slice(0, train_end_frame + 2)
    test_mask = slice(train_end_frame + 2, None)
    return ExtrapolationResult(
        name="linear_delta",
        train_mse=_mse(pred[train_mask], z[train_mask]),
        test_mse=_mse(pred[test_mask], z[test_mask]),
        z_test_mse=_mse(pred[test_mask], z[test_mask]),
    )


def physics_bounce_baseline(
    trajectory: ObjectPoseTrajectory,
    *,
    bounce_cfg: BounceConfig,
    train_end_frame: int,
    z0: float,
) -> ExtrapolationResult:
    z = _align_z_series(trajectory, z0)
    observed = torch.tensor(z.tolist(), dtype=torch.float32)
    train_end = int(train_end_frame)
    train_obs = observed[: train_end + 2]
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
        pred_t = simulate_bounce(torch.tensor(result.final_restitution), bounce_cfg)
    pred = pred_t.numpy()
    train_mask = slice(0, train_end + 2)
    test_mask = slice(train_end + 2, None)
    return ExtrapolationResult(
        name="physics_bounce_fit",
        train_mse=_mse(pred[train_mask], z[train_mask]),
        test_mse=_mse(pred[test_mask], z[test_mask]),
        z_test_mse=_mse(pred[test_mask], z[test_mask]),
    )


def predict_physics_z(
    trajectory: ObjectPoseTrajectory,
    *,
    bounce_cfg: BounceConfig,
    train_end_frame: int,
    z0: float,
) -> np.ndarray:
    """Full z trajectory from bounce model fit on train frames only."""

    z = _align_z_series(trajectory, z0)
    observed = torch.tensor(z.tolist(), dtype=torch.float32)
    train_end = int(train_end_frame)
    train_obs = observed[: train_end + 2]
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
        return simulate_bounce(torch.tensor(result.final_restitution), bounce_cfg).numpy()


def position_mse_3d(
    trajectory: ObjectPoseTrajectory,
    pred_z: np.ndarray,
    *,
    train_end_frame: int,
    z0: float,
) -> tuple[float, float]:
    """MSE on full (x,y,z) using GT xy and predicted z (sphere bounce is vertical)."""

    z_gt = _align_z_series(trajectory, z0)
    pos = trajectory.positions()
    xy = np.concatenate([pos[0:1, :2], pos[:, :2]], axis=0)
    pred_pos = np.column_stack([xy, pred_z])
    gt_pos = np.column_stack([xy, z_gt])
    train_end = train_end_frame + 1
    train_mse = _mse(pred_pos[: train_end + 1], gt_pos[: train_end + 1])
    test_mse = _mse(pred_pos[train_end + 1 :], gt_pos[train_end + 1 :])
    return train_mse, test_mse
