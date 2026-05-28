"""Phys4D utilities (bounce pipeline, poses, optional legacy modules)."""

from .poses import ObjectPose, ObjectPoseTrajectory, load_object_poses_csv

__all__ = [
    "ObjectPose",
    "ObjectPoseTrajectory",
    "load_object_poses_csv",
]

# Optional modules synced from cs231n_project_phys4d/ for Phase 4+.
try:
    from .gaussian_ply import (  # noqa: F401
        GaussianCloud,
        load_gaussian_ply,
        save_gaussian_ply,
        warp_gaussians_to_frame,
    )

    __all__ += [
        "GaussianCloud",
        "load_gaussian_ply",
        "save_gaussian_ply",
        "warp_gaussians_to_frame",
    ]
except ImportError:
    pass

try:
    from .differentiable_bounce import (  # noqa: F401
        BounceConfig,
        RecoveryResult,
        fit_restitution,
        restitution_from_raw,
        simulate_bounce,
    )

    __all__ += [
        "BounceConfig",
        "RecoveryResult",
        "fit_restitution",
        "restitution_from_raw",
        "simulate_bounce",
    ]
except ImportError:
    pass
