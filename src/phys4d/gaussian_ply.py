"""Load / warp 3D Gaussian Splatting ``point_cloud.ply`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .poses import ObjectPoseTrajectory
from .rigid import (
    apply_pose_to_gaussian_rotations,
    apply_pose_to_points,
    relative_pose,
)


@dataclass
class GaussianCloud:
    """Minimal 3DGS vertex buffer (graphdeco-inria PLY layout)."""

    vertices: np.ndarray  # structured array from plyfile

    @property
    def xyz(self) -> np.ndarray:
        return np.stack(
            [self.vertices["x"], self.vertices["y"], self.vertices["z"]], axis=1
        )

    @property
    def rot_wxyz(self) -> np.ndarray:
        return np.stack(
            [
                self.vertices["rot_0"],
                self.vertices["rot_1"],
                self.vertices["rot_2"],
                self.vertices["rot_3"],
            ],
            axis=1,
        )

    def copy(self) -> "GaussianCloud":
        return GaussianCloud(vertices=self.vertices.copy())


def load_gaussian_ply(path: Path) -> GaussianCloud:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plyfile is required to read 3DGS PLY files: pip install plyfile"
        ) from exc

    ply = PlyData.read(str(path.resolve()))
    return GaussianCloud(vertices=np.asarray(ply["vertex"].data))


def save_gaussian_ply(cloud: GaussianCloud, path: Path) -> None:
    from plyfile import PlyData, PlyElement

    vertex = PlyElement.describe(cloud.vertices, "vertex")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([vertex], text=False).write(str(path))


def filter_sphere_gaussians(
    cloud: GaussianCloud,
    center: np.ndarray,
    radius: float,
    margin: float = 1.35,
) -> GaussianCloud:
    """Keep Gaussians within ``margin * radius`` of a world-space center.

    Note: 3DGS coordinates may be offset from PyBullet poses; prefer
    ``filter_sphere_gaussians_by_masks`` when masks are available.
    """

    center = np.asarray(center, dtype=np.float64).reshape(3)
    dist = np.linalg.norm(cloud.xyz - center, axis=1)
    keep = dist <= float(radius) * float(margin)
    return GaussianCloud(vertices=cloud.vertices[keep].copy())


def filter_sphere_gaussians_by_opacity(
    cloud: GaussianCloud,
    percentile: float = 92.0,
) -> GaussianCloud:
    """Keep high-opacity Gaussians (robust when mask/projection frame mismatches)."""

    op = np.asarray(cloud.vertices["opacity"], dtype=np.float64)
    thresh = np.percentile(op, float(percentile))
    keep = op >= thresh
    return GaussianCloud(vertices=cloud.vertices[keep].copy())


def filter_sphere_gaussians_by_masks(
    cloud: GaussianCloud,
    masks_root: Path,
    frame: int,
    *,
    min_camera_hits: int = 1,
    gs_cameras: list[dict] | None = None,
    pybullet_cameras: list[dict] | None = None,
    gs_cam_indices: list[int] | None = None,
) -> GaussianCloud:
    """Keep Gaussians whose centers project into the sphere mask at ``frame``."""

    from .camera_project import filter_points_by_mask_union

    keep = filter_points_by_mask_union(
        cloud.xyz,
        masks_root=masks_root,
        frame=frame,
        min_camera_hits=min_camera_hits,
        gs_cameras=gs_cameras,
        pybullet_cameras=pybullet_cameras,
        gs_cam_indices=gs_cam_indices,
    )
    return GaussianCloud(vertices=cloud.vertices[keep].copy())


def estimate_pb_to_gs_z_scale(
    trajectory: ObjectPoseTrajectory,
    gs_sphere_z: float,
    *,
    ref_frame: int,
    align_frame: int,
    gs_z_at_align: float,
) -> float:
    """Fit scalar ``a`` in ``delta_z_gs = a * delta_z_pb`` from two timesteps."""

    z_ref = float(trajectory.by_frame(ref_frame).position[2])
    z_align = float(trajectory.by_frame(align_frame).position[2])
    delta_pb = z_align - z_ref
    if abs(delta_pb) < 1e-8:
        return 1.0
    return float((gs_z_at_align - gs_sphere_z) / delta_pb)


def warp_gaussians_to_frame(
    cloud: GaussianCloud,
    trajectory: ObjectPoseTrajectory,
    *,
    ref_frame: int,
    target_frame: int,
    z_scale: float | None = None,
) -> GaussianCloud:
    """Move Gaussians to ``target_frame`` using PyBullet pose deltas.

    For vertical bounce, GS training coords are not 1:1 with PyBullet ``z``.
    Pass ``z_scale`` (or use ``estimate_pb_to_gs_z_scale``) so image-space
    motion matches segmentation masks; otherwise only ``x/y`` rotation from
    ``relative_pose`` is applied and ``z`` uses the scaled increment.
    """

    ref_pose = trajectory.by_frame(ref_frame)
    target_pose = trajectory.by_frame(target_frame)
    delta = relative_pose(ref_pose.matrix, target_pose.matrix)

    out = cloud.copy()
    warped_xyz = apply_pose_to_points(cloud.xyz, delta)
    if z_scale is not None:
        delta_z_pb = float(target_pose.position[2] - ref_pose.position[2])
        warped_xyz[:, 2] = cloud.xyz[:, 2] + float(z_scale) * delta_z_pb
    out.vertices["x"] = warped_xyz[:, 0].astype(np.float32)
    out.vertices["y"] = warped_xyz[:, 1].astype(np.float32)
    out.vertices["z"] = warped_xyz[:, 2].astype(np.float32)

    warped_rot = apply_pose_to_gaussian_rotations(cloud.rot_wxyz, delta)
    out.vertices["rot_0"] = warped_rot[:, 0].astype(np.float32)
    out.vertices["rot_1"] = warped_rot[:, 1].astype(np.float32)
    out.vertices["rot_2"] = warped_rot[:, 2].astype(np.float32)
    out.vertices["rot_3"] = warped_rot[:, 3].astype(np.float32)
    return out
