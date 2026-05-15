"""Lightweight physics utilities for the CS231N Phys4D milestone."""

from .differentiable_bounce import (
    BounceConfig,
    RecoveryResult,
    fit_restitution,
    restitution_from_raw,
    simulate_bounce,
)
from .gaussian_ply import GaussianCloud, load_gaussian_ply, save_gaussian_ply, warp_gaussians_to_frame
from .poses import ObjectPose, ObjectPoseTrajectory, load_object_poses_csv
from .trajectory_metrics import TrajectorySplitMetrics, fit_restitution_on_train_frames

__all__ = [
    "BounceConfig",
    "GaussianCloud",
    "ObjectPose",
    "ObjectPoseTrajectory",
    "RecoveryResult",
    "TrajectorySplitMetrics",
    "fit_restitution",
    "fit_restitution_on_train_frames",
    "load_gaussian_ply",
    "load_object_poses_csv",
    "restitution_from_raw",
    "save_gaussian_ply",
    "simulate_bounce",
    "warp_gaussians_to_frame",
]
