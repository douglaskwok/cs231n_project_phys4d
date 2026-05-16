"""Load PyBullet object pose trajectories from CSV exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rigid import pose_matrix


@dataclass(frozen=True)
class ObjectPose:
    """Single timestep body pose in world coordinates."""

    frame: int
    time_s: float
    position: np.ndarray  # (3,)
    quat_xyzw: np.ndarray  # (4,) qx,qy,qz,qw
    linear_velocity: np.ndarray  # (3,)

    @property
    def matrix(self) -> np.ndarray:
        return pose_matrix(self.position, self.quat_xyzw)


@dataclass
class ObjectPoseTrajectory:
    """Full pose log indexed by simulation frame."""

    poses: list[ObjectPose]

    def __len__(self) -> int:
        return len(self.poses)

    def by_frame(self, frame: int) -> ObjectPose:
        for pose in self.poses:
            if pose.frame == frame:
                return pose
        raise KeyError(f"No pose for frame {frame}")

    def positions(self) -> np.ndarray:
        return np.stack([p.position for p in self.poses], axis=0)

    def z_series(self) -> np.ndarray:
        return self.positions()[:, 2]


def load_object_poses_csv(path: Path) -> ObjectPoseTrajectory:
    """Read ``object_poses.csv`` written by ``generate_sphere_bounce_dataset.py``."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        size = path.stat().st_size
        raise ValueError(
            f"{path} reads as empty (metadata size {size} B). "
            "On iCloud/Desktop, download the file locally in Finder or regenerate the scene."
        )
    poses: list[ObjectPose] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
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
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            have = sorted(reader.fieldnames or [])
            raise ValueError(
                f"{path} missing columns; need {sorted(required)}; got {have}"
            )
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
                        [
                            float(row["qx"]),
                            float(row["qy"]),
                            float(row["qz"]),
                            float(row["qw"]),
                        ],
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
