"""Project world points into PyBullet and 3D Gaussian Splatting cameras."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colmap_export import view_matrix_to_c2w


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float


def load_cameras(cameras_json: Path | str) -> list[dict]:
    """Load PyBullet ``cameras.json`` from dataset export."""

    with Path(cameras_json).open("r", encoding="utf-8") as f:
        blob = json.load(f)
    return list(blob["cameras"])


def load_gs_cameras(cameras_json: Path | str) -> list[dict]:
    """Load trained 3DGS ``cameras.json`` (COLMAP-style entries)."""

    with Path(cameras_json).open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if isinstance(entries, dict) and "cameras" in entries:
        return list(entries["cameras"])
    return list(entries)


def _pybullet_view_proj(camera_record: dict) -> tuple[np.ndarray, np.ndarray]:
    """PyBullet returns column-major 16 floats but stores them row-wise in JSON."""

    view = np.array(camera_record["view_matrix_row_major"], dtype=np.float64).reshape(4, 4)
    proj = np.array(camera_record["projection_matrix_row_major"], dtype=np.float64).reshape(
        4, 4
    )
    return view, proj


def project_pybullet_points(
    points: np.ndarray,
    camera_record: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project PyBullet-world points using stored view/projection matrices.

    Uses row-vector multiplication ``[x,y,z,1] @ view @ proj`` (matches PyBullet).
    Pixel coords: ``u = (ndc_x+1)*0.5*W``, ``v = (1-ndc_y)*0.5*H`` (image y down).
    """

    view, proj = _pybullet_view_proj(camera_record)
    width, height = int(camera_record["image_size"][0]), int(camera_record["image_size"][1])
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1)

    clip = (hom @ view) @ proj
    w = clip[:, 3]
    valid_w = np.abs(w) > 1e-8
    ndc = np.zeros((pts.shape[0], 3), dtype=np.float64)
    ndc[valid_w] = clip[valid_w, :3] / w[valid_w, None]

    u = (ndc[:, 0] * 0.5 + 0.5) * width
    v = (1.0 - ndc[:, 1]) * 0.5 * height
    in_frustum = valid_w & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    in_bounds = (
        in_frustum
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return np.stack([u, v], axis=1), in_bounds


def gs_camera_matrices(entry: dict) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    """World-to-camera rotation/translation from a trained 3DGS cameras.json entry."""

    rot = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
    center = np.asarray(entry["position"], dtype=np.float64).reshape(3)
    # Stored ``rotation`` is R in COLMAP sense with x_cam = R @ x_world + t, t = -R @ C.
    r_w2c = rot.T
    t_w2c = -r_w2c @ center
    intr = CameraIntrinsics(
        width=int(entry["width"]),
        height=int(entry["height"]),
        fx=float(entry["fx"]),
        fy=float(entry["fy"]),
    )
    return r_w2c, t_w2c, intr


def project_gaussian_splatting_points(
    points: np.ndarray,
    gs_camera: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points in the **trained 3DGS world frame** to pixel coordinates.

    Matches graphdeco-inria/gaussian-splatting COLMAP camera convention:
    ``x_cam = R^T @ (x_world - C)`` with image v increasing downward.
    """

    r_w2c, t_w2c, intr = gs_camera_matrices(gs_camera)
    pts = np.asarray(points, dtype=np.float64)
    cam = (r_w2c @ pts.T).T + t_w2c

    z = cam[:, 2]
    visible = z > 1e-4
    u = intr.fx * (cam[:, 0] / z) + intr.width * 0.5
    # COLMAP/3DGS image origin: flip vertical vs camera y-up.
    v = intr.height - (intr.fy * (cam[:, 1] / z) + intr.height * 0.5)

    in_bounds = (
        visible
        & (u >= 0)
        & (u < intr.width)
        & (v >= 0)
        & (v < intr.height)
    )
    return np.stack([u, v], axis=1), in_bounds


def project_transforms_json_points(
    points: np.ndarray,
    transform_matrix: list | np.ndarray,
    camera_angle_x: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project using NeRF ``transform_matrix`` with the 3DGS Blender Y/Z flip."""

    c2w = np.asarray(transform_matrix, dtype=np.float64).reshape(4, 4)
    c2w = c2w.copy()
    c2w[:3, 1:3] *= -1.0
    w2c = np.linalg.inv(c2w)
    fovx = float(camera_angle_x)
    fovy = 2.0 * math.atan(math.tan(fovx * 0.5) * (height / width))
    fx = 0.5 * width / math.tan(fovx * 0.5)
    fy = 0.5 * height / math.tan(fovy * 0.5)
    fake_cam = {
        "rotation": w2c[:3, :3].T.tolist(),
        "position": np.linalg.inv(c2w)[:3, 3].tolist(),  # not used when we pass R,t directly
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
    }
    # Rebuild from w2c directly for consistency.
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    cam = (w2c @ np.concatenate([pts, ones], axis=1).T).T[:, :3]
    z = cam[:, 2]
    visible = z > 1e-4
    u = fx * (cam[:, 0] / z) + width * 0.5
    v = height - (fy * (cam[:, 1] / z) + height * 0.5)
    in_bounds = visible & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=1), in_bounds


def _read_mask(path: Path) -> np.ndarray:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("imageio required: pip install imageio") from exc

    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def filter_points_by_mask_union(
    points: np.ndarray,
    *,
    masks_root: Path,
    frame: int,
    min_camera_hits: int = 1,
    pybullet_cameras: list[dict] | None = None,
    gs_cameras: list[dict] | None = None,
    gs_cam_indices: list[int] | None = None,
) -> np.ndarray:
    """Keep points that project onto foreground masks in >= ``min_camera_hits`` views."""

    pts = np.asarray(points, dtype=np.float64)
    hits = np.zeros(pts.shape[0], dtype=np.int32)
    frame_tag = f"frame{frame:05d}.png"
    masks_root = masks_root.resolve()

    if gs_cameras:
        indices = gs_cam_indices if gs_cam_indices is not None else list(range(len(gs_cameras)))
        for idx in indices:
            cam = gs_cameras[idx]
            cam_id = int(cam.get("id", idx))
            mask_path = masks_root / f"cam{cam_id:02d}" / frame_tag
            if not mask_path.is_file():
                continue
            mask = _read_mask(mask_path)
            h, w = mask.shape[:2]
            uv, visible = project_gaussian_splatting_points(pts, cam)
            vis_idx = np.where(visible)[0]
            u = np.clip(np.round(uv[vis_idx, 0]).astype(np.int32), 0, w - 1)
            v = np.clip(np.round(uv[vis_idx, 1]).astype(np.int32), 0, h - 1)
            hits[vis_idx[mask[v, u] > 0]] += 1

    if pybullet_cameras:
        for cam in pybullet_cameras:
            ci = int(cam["index"])
            mask_path = masks_root / f"cam{ci:02d}" / frame_tag
            if not mask_path.is_file():
                continue
            mask = _read_mask(mask_path)
            h, w = mask.shape[:2]
            uv, visible = project_pybullet_points(pts, cam)
            vis_idx = np.where(visible)[0]
            u = np.clip(np.round(uv[vis_idx, 0]).astype(np.int32), 0, w - 1)
            v = np.clip(np.round(uv[vis_idx, 1]).astype(np.int32), 0, h - 1)
            hits[vis_idx[mask[v, u] > 0]] += 1

    if not gs_cameras and not pybullet_cameras:
        raise ValueError("Provide gs_cameras and/or pybullet_cameras")

    return hits >= int(min_camera_hits)


def mask_coverage_fraction(
    points: np.ndarray,
    mask_path: Path,
    *,
    pybullet_camera: dict | None = None,
    gs_camera: dict | None = None,
) -> float:
    """Fraction of projected points that land on foreground (mask > 0)."""

    pts = np.asarray(points, dtype=np.float64)
    if gs_camera is not None:
        uv, visible = project_gaussian_splatting_points(pts, gs_camera)
    elif pybullet_camera is not None:
        uv, visible = project_pybullet_points(pts, pybullet_camera)
    else:
        raise ValueError("Provide gs_camera or pybullet_camera")

    if not np.any(visible):
        return 0.0

    mask = _read_mask(mask_path)
    vis_idx = np.where(visible)[0]
    u = np.clip(np.round(uv[vis_idx, 0]).astype(np.int32), 0, mask.shape[1] - 1)
    v = np.clip(np.round(uv[vis_idx, 1]).astype(np.int32), 0, mask.shape[0] - 1)
    hits = int(np.sum(mask[v, u] > 0))
    return float(hits) / float(vis_idx.size)


def mask_coverage_fraction_multiview(
    points: np.ndarray,
    masks_root: Path,
    frame: int,
    gs_cameras: list[dict],
    *,
    max_views: int | None = None,
) -> float | None:
    """Mean mask coverage over views that have a mask image for ``frame``."""

    frame_tag = f"frame{frame:05d}.png"
    cams = gs_cameras if max_views is None else gs_cameras[:max_views]
    fracs: list[float] = []
    for ci, cam in enumerate(cams):
        mask_path = masks_root / f"cam{ci:02d}" / frame_tag
        if not mask_path.is_file():
            continue
        fracs.append(mask_coverage_fraction(points, mask_path, gs_camera=cam))
    if not fracs:
        return None
    return float(np.mean(fracs))
