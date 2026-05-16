"""Trajectory metrics on held-out frame ranges."""

from __future__ import annotations

import numpy as np

from phys4d.poses import ObjectPoseTrajectory, load_object_poses_csv
from pathlib import Path


def trajectory_metrics(
    pred: ObjectPoseTrajectory,
    gt: ObjectPoseTrajectory,
    frame_start: int,
    frame_end: int,
) -> dict[str, float]:
    """MSE on position, z, and velocity; R² on z and speed magnitude."""
    pos_err: list[float] = []
    z_err: list[float] = []
    vel_err: list[float] = []
    z_pred: list[float] = []
    z_true: list[float] = []
    speed_pred: list[float] = []
    speed_true: list[float] = []

    for f in range(frame_start, frame_end + 1):
        pp = pred.by_frame(f)
        gp = gt.by_frame(f)
        pos_err.append(float(np.sum((pp.position - gp.position) ** 2)))
        z_err.append(float((pp.position[2] - gp.position[2]) ** 2))
        vel_err.append(float(np.sum((pp.linear_velocity - gp.linear_velocity) ** 2)))
        z_pred.append(float(pp.position[2]))
        z_true.append(float(gp.position[2]))
        speed_pred.append(float(np.linalg.norm(pp.linear_velocity)))
        speed_true.append(float(np.linalg.norm(gp.linear_velocity)))

    n = max(len(pos_err), 1)

    def _r2(y_pred: list[float], y_true: list[float]) -> float:
        yp = np.array(y_pred)
        yt = np.array(y_true)
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return 1.0 - ss_res / ss_tot

    return {
        "pos_mse": float(np.mean(pos_err)),
        "z_mse": float(np.mean(z_err)),
        "vel_mse": float(np.mean(vel_err)),
        "z_r2": _r2(z_pred, z_true),
        "speed_r2": _r2(speed_pred, speed_true),
        "num_frames": float(frame_end - frame_start + 1),
    }


def metrics_from_states(
    pred: ObjectPoseTrajectory,
    gt: ObjectPoseTrajectory,
    frame_start: int,
    frame_end: int,
) -> dict[str, float]:
    return trajectory_metrics(pred, gt, frame_start, frame_end)


def metrics_from_csv(
    pred_csv: Path,
    gt_csv: Path,
    frame_start: int,
    frame_end: int,
) -> dict[str, float]:
    return trajectory_metrics(
        load_object_poses_csv(pred_csv),
        load_object_poses_csv(gt_csv),
        frame_start,
        frame_end,
    )
