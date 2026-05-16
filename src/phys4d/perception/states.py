"""Per-frame object states with optional finite-difference velocity."""

from __future__ import annotations

import numpy as np

from phys4d.object_state import ObjectState, trajectory_to_states
from phys4d.poses import ObjectPoseTrajectory, load_object_poses_csv
from pathlib import Path


def finite_diff_velocity(
    positions: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Central difference velocity; forward/backward at ends."""
    n = positions.shape[0]
    vel = np.zeros_like(positions)
    if n < 2:
        return vel
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt_s)
    vel[0] = (positions[1] - positions[0]) / dt_s
    vel[-1] = (positions[-1] - positions[-2]) / dt_s
    return vel


def trajectory_to_states_fd(
    traj: ObjectPoseTrajectory,
    *,
    dt_s: float,
    use_sim_velocity: bool = False,
) -> list[ObjectState]:
    """Build states; optionally replace sim velocity with finite-difference z (or full 3D)."""
    states = trajectory_to_states(traj)
    if not use_sim_velocity:
        pos = np.stack([s.position for s in states], axis=0)
        vel = finite_diff_velocity(pos, dt_s)
        out: list[ObjectState] = []
        for i, s in enumerate(states):
            out.append(
                ObjectState(
                    frame=s.frame,
                    time_s=s.time_s,
                    position=s.position.copy(),
                    quat_xyzw=s.quat_xyzw.copy(),
                    linear_velocity=vel[i].astype(np.float64),
                )
            )
        return out
    return states


def load_states_csv_or_poses(
    poses_path: Path,
    *,
    dt_s: float,
    use_sim_velocity: bool = False,
) -> list[ObjectState]:
    traj = load_object_poses_csv(poses_path)
    return trajectory_to_states_fd(traj, dt_s=dt_s, use_sim_velocity=use_sim_velocity)
