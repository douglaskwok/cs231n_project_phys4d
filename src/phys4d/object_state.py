"""Per-object state sequences for the visual-dynamics prediction stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .poses import ObjectPose, ObjectPoseTrajectory


@dataclass(frozen=True)
class ObjectState:
    """Rigid-body state at one frame (sim poses or future Gaussian COM tracks)."""

    frame: int
    time_s: float
    position: np.ndarray  # (3,)
    quat_xyzw: np.ndarray  # (4,)
    linear_velocity: np.ndarray  # (3,)

    @classmethod
    def from_pose(cls, pose: ObjectPose) -> ObjectState:
        return cls(
            frame=pose.frame,
            time_s=pose.time_s,
            position=pose.position.copy(),
            quat_xyzw=pose.quat_xyzw.copy(),
            linear_velocity=pose.linear_velocity.copy(),
        )


def trajectory_to_states(traj: ObjectPoseTrajectory) -> list[ObjectState]:
    return [ObjectState.from_pose(p) for p in traj.poses]


def stack_state_vectors(
    states: list[ObjectState],
    *,
    include_velocity: bool = True,
) -> np.ndarray:
    """Flatten states to (T, D) for dynamics heads. D=7 or 10 (pos + quat [+ vel])."""

    rows = []
    for s in states:
        parts = [s.position, s.quat_xyzw]
        if include_velocity:
            parts.append(s.linear_velocity)
        rows.append(np.concatenate(parts, axis=0))
    return np.stack(rows, axis=0).astype(np.float64)


def window_states(
    states: list[ObjectState],
    frame: int,
    history: int,
) -> list[ObjectState]:
    """Last ``history`` states ending at ``frame`` (inclusive)."""

    idx = next(i for i, s in enumerate(states) if s.frame == frame)
    start = max(0, idx - history + 1)
    return states[start : idx + 1]
