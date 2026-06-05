"""Load PyBullet object pose trajectories from CSV exports (Step 4a GT overlay)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObjectPose:
    """Single timestep body pose in world coordinates."""

    frame: int
    time_s: float
    position: np.ndarray
    quat_xyzw: np.ndarray
    linear_velocity: np.ndarray


@dataclass
class ObjectPoseTrajectory:
    """Full pose log indexed by simulation frame."""

    poses: list[ObjectPose]

    def by_frame(self, frame: int) -> ObjectPose:
        for pose in self.poses:
            if pose.frame == frame:
                return pose
        raise KeyError(f"No pose for frame {frame}")


def load_object_poses_csv(path: Path) -> ObjectPoseTrajectory:
    """Read ``object_poses.csv`` from the PyBullet scene export."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    required = {
        "frame",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "qx",
        "qy",
        "qz",
        "qw",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
    }
    poses: list[ObjectPose] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            have = sorted(reader.fieldnames or [])
            raise ValueError(f"{path} missing columns; need {sorted(required)}; got {have}")
        for row in reader:
            poses.append(
                ObjectPose(
                    frame=int(float(row["frame"])),
                    time_s=float(row["time_s"]),
                    position=np.array(
                        [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                        dtype=np.float64,
                    ),
                    quat_xyzw=np.array(
                        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                        dtype=np.float64,
                    ),
                    linear_velocity=np.array(
                        [
                            float(row["vx_m_s"]),
                            float(row["vy_m_s"]),
                            float(row["vz_m_s"]),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
    if not poses:
        raise ValueError(f"No poses in {path}")
    return ObjectPoseTrajectory(poses=poses)
