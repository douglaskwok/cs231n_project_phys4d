"""Phase 4 — compose predicted trajectory renders on held-out camera views.

This module provides a lightweight Phase 4 baseline that projects predicted 3D
sphere centers into held-out camera images and overlays a filled disk. It keeps
the same per-frame camera naming convention used by existing 4DGS render tools.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - fallback for lightweight envs
    imageio = None
    from PIL import Image


_CAM_FRAME_RE = re.compile(r"cam(\d+)_(\d+)$")


def _frame_intrinsics(
    frame_rec: dict,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float, float]:
    """Resolve per-frame intrinsics, falling back to the shared top-level block.

    Different rig cameras can have different FoV, so each transforms frame may carry
    its own fl_x/fl_y/cx/cy. Older exports without per-frame keys use the shared set.
    """

    return (
        float(frame_rec.get("fl_x", fl_x)),
        float(frame_rec.get("fl_y", fl_y)),
        float(frame_rec.get("cx", cx)),
        float(frame_rec.get("cy", cy)),
    )


@dataclass(frozen=True)
class Phase4Result:
    """Artifacts produced by phase 4 composition."""

    out_dir: Path
    renders_dir: Path
    num_rendered: int
    num_missing_predictions: int
    num_invalid_projections: int
    calibrated_projection_mode: str
    sample_png: Path | None


def _read_predicted_csv(path: Path) -> dict[int, np.ndarray]:
    """Load phase-3 predictions keyed by sim frame index."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing predicted trajectory CSV: {path}")
    out: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"frame", "x", "y", "z"}
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns {sorted(needed)}")
        for row in reader:
            frame = int(float(row["frame"]))
            xyz = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float64,
            )
            out[frame] = xyz
    if not out:
        raise ValueError(f"No prediction rows in {path}")
    return out


def _parse_cam_frame_from_file_path(file_path: str) -> tuple[int, int]:
    """Extract (cam_idx, sim_frame_idx) from DyNeRF `file_path` stem."""

    stem = Path(file_path).name
    match = _CAM_FRAME_RE.search(stem)
    if not match:
        raise ValueError(f"Unsupported transforms file_path format: {file_path}")
    return int(match.group(1)), int(match.group(2))


def _project_world_to_pixel(
    point_world: np.ndarray,
    c2w: np.ndarray,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """Project world point to pixel coordinates using camera-to-world pose.

    The transforms files store camera-to-world matrices. We invert to world-to-
    camera and use pinhole projection (OpenCV-like sign convention).
    """

    w2c = np.linalg.inv(c2w)
    point_h = np.concatenate([point_world.astype(np.float64), np.array([1.0])], axis=0)
    point_cam = w2c @ point_h
    x_c, y_c, z_c = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
    # DyNeRF camera conventions can differ by axis signs; callers can still reject
    # out-of-bounds projections and try alternate sign choices.
    if z_c <= 1e-6:
        return np.nan, np.nan, z_c
    u = fl_x * (x_c / z_c) + cx
    v = fl_y * (y_c / z_c) + cy
    return float(u), float(v), z_c


def _project_with_mode(
    point_world: np.ndarray,
    c2w: np.ndarray,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    projection_mode: str,
) -> tuple[float, float, float]:
    """Project using one explicit camera-convention mode."""

    w2c = np.linalg.inv(c2w)
    point_h = np.concatenate([point_world.astype(np.float64), np.array([1.0])], axis=0)
    point_cam = w2c @ point_h
    x_c, y_c, z_c = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])

    if projection_mode == "opencv":
        x_v, y_v, z_v = x_c, y_c, z_c
    elif projection_mode == "opencv_zflip":
        x_v, y_v, z_v = x_c, y_c, -z_c
    elif projection_mode == "opencv_yzflip":
        x_v, y_v, z_v = x_c, -y_c, -z_c
    elif projection_mode == "opencv_yflip":
        x_v, y_v, z_v = x_c, -y_c, z_c
    else:
        raise ValueError(f"Unsupported projection mode: {projection_mode}")
    if z_v <= 1e-6:
        return np.nan, np.nan, z_v
    u = fl_x * (x_v / z_v) + cx
    v = fl_y * (y_v / z_v) + cy
    if 0.0 <= u < float(width) and 0.0 <= v < float(height):
        return float(u), float(v), float(z_v)
    return np.nan, np.nan, z_v


def _draw_filled_circle(
    image: np.ndarray,
    *,
    center_u: float,
    center_v: float,
    radius_px: int,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Draw a filled circle without OpenCV dependency."""

    out = image.copy()
    h, w = out.shape[0], out.shape[1]
    if radius_px <= 0:
        return out
    cu = int(round(center_u))
    cv = int(round(center_v))
    u0 = max(0, cu - radius_px)
    u1 = min(w - 1, cu + radius_px)
    v0 = max(0, cv - radius_px)
    v1 = min(h - 1, cv + radius_px)
    if u1 < u0 or v1 < v0:
        return out
    ys, xs = np.ogrid[v0 : v1 + 1, u0 : u1 + 1]
    mask = (xs - cu) ** 2 + (ys - cv) ** 2 <= radius_px**2
    tile = out[v0 : v1 + 1, u0 : u1 + 1]
    tile[mask] = np.array(color_rgb, dtype=np.uint8)
    out[v0 : v1 + 1, u0 : u1 + 1] = tile
    return out


def _draw_gaussian_splat(
    image: np.ndarray,
    *,
    center_u: float,
    center_v: float,
    radius_px: int,
    color_rgb: tuple[int, int, int],
    alpha: float = 0.85,
) -> np.ndarray:
    """Draw a soft 2D Gaussian splat instead of a hard disk."""

    out = image.astype(np.float64).copy()
    h, w = out.shape[0], out.shape[1]
    if radius_px <= 0:
        return image.copy()
    cu = float(center_u)
    cv = float(center_v)
    spread = max(1.0, float(radius_px) * 0.6)
    cutoff = int(max(2, round(3.0 * spread)))
    u0 = max(0, int(round(cu)) - cutoff)
    u1 = min(w - 1, int(round(cu)) + cutoff)
    v0 = max(0, int(round(cv)) - cutoff)
    v1 = min(h - 1, int(round(cv)) + cutoff)
    if u1 < u0 or v1 < v0:
        return image.copy()
    ys, xs = np.mgrid[v0 : v1 + 1, u0 : u1 + 1]
    d2 = (xs - cu) ** 2 + (ys - cv) ** 2
    weight = np.exp(-0.5 * d2 / (spread**2))
    weight = np.clip(weight * float(alpha), 0.0, 1.0)[..., None]
    color = np.array(color_rgb, dtype=np.float64).reshape(1, 1, 3)
    patch = out[v0 : v1 + 1, u0 : u1 + 1, :]
    patch = (1.0 - weight) * patch + weight * color
    out[v0 : v1 + 1, u0 : u1 + 1, :] = patch
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _read_rgb_png(path: Path) -> np.ndarray:
    """Read RGB PNG with imageio fallback to PIL."""

    if imageio is not None:
        arr = imageio.imread(path)
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Write RGB PNG with imageio fallback to PIL."""

    if imageio is not None:
        imageio.imwrite(path, rgb)
    else:
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def _read_mask_png(path: Path) -> np.ndarray:
    """Read binary mask as boolean array."""

    if imageio is not None:
        arr = imageio.imread(path)
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _mask_ball_from_background(
    background: np.ndarray,
    *,
    scene_dir: Path,
    cam_idx: int,
    frame_idx: int,
) -> np.ndarray:
    """Suppress the GT ball by zeroing pixels inside the object mask."""

    mask_path = scene_dir / "masks" / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
    if not mask_path.is_file():
        return background
    mask = _read_mask_png(mask_path)
    out = background.copy()
    out[mask] = 0
    return out


def _mask_centroid(
    *,
    scene_dir: Path,
    cam_idx: int,
    frame_idx: int,
) -> tuple[float, float] | None:
    """Compute GT mask centroid in pixel space for camera-convention calibration."""

    mask_path = scene_dir / "masks" / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
    if not mask_path.is_file():
        return None
    mask = _read_mask_png(mask_path)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _calibrate_projection_mode(
    *,
    frames: list[dict],
    scene_dir: Path,
    predicted: dict[int, np.ndarray],
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> str:
    """Pick one projection convention that best matches GT mask centroids."""

    modes = ["opencv", "opencv_zflip", "opencv_yzflip", "opencv_yflip"]
    best_mode = "opencv"
    best_error = float("inf")
    for mode in modes:
        errors: list[float] = []
        for frame_rec in frames:
            cam_idx, frame_idx = _parse_cam_frame_from_file_path(str(frame_rec["file_path"]))
            pred = predicted.get(frame_idx)
            if pred is None:
                continue
            centroid = _mask_centroid(scene_dir=scene_dir, cam_idx=cam_idx, frame_idx=frame_idx)
            if centroid is None:
                continue
            c2w = np.asarray(frame_rec["transform_matrix"], dtype=np.float64)
            f_fx, f_fy, f_cx, f_cy = _frame_intrinsics(
                frame_rec, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy
            )
            u, v, _ = _project_with_mode(
                pred,
                c2w,
                fl_x=f_fx,
                fl_y=f_fy,
                cx=f_cx,
                cy=f_cy,
                width=width,
                height=height,
                projection_mode=mode,
            )
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            errors.append(float(np.hypot(u - centroid[0], v - centroid[1])))
        if errors:
            err = float(np.mean(errors))
            if err < best_error:
                best_error = err
                best_mode = mode
    # Without GT masks we cannot calibrate from centroids; pick the convention that
    # projects the most predicted points into the image (common when z is flipped).
    if not math.isfinite(best_error):
        best_mode = "opencv"
        best_valid = -1
        for mode in modes:
            valid = 0
            for frame_rec in frames:
                cam_idx, frame_idx = _parse_cam_frame_from_file_path(
                    str(frame_rec["file_path"])
                )
                pred = predicted.get(frame_idx)
                if pred is None:
                    continue
                c2w = np.asarray(frame_rec["transform_matrix"], dtype=np.float64)
                f_fx, f_fy, f_cx, f_cy = _frame_intrinsics(
                    frame_rec, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy
                )
                u, v, _ = _project_with_mode(
                    pred,
                    c2w,
                    fl_x=f_fx,
                    fl_y=f_fy,
                    cx=f_cx,
                    cy=f_cy,
                    width=width,
                    height=height,
                    projection_mode=mode,
                )
                if np.isfinite(u) and np.isfinite(v) and 0.0 <= u < width and 0.0 <= v < height:
                    valid += 1
            if valid > best_valid:
                best_valid = valid
                best_mode = mode
    return best_mode


def run_phase4(
    *,
    predicted_csv: Path,
    dynerf_export: Path,
    scene_dir: Path,
    out_dir: Path,
    sphere_radius_m: float,
    overlay_color_rgb: tuple[int, int, int] = (255, 64, 64),
    mask_gt_ball: bool = True,
    render_style: str = "gaussian_splat",
) -> Phase4Result:
    """Compose held-out test renders by overlaying projected predicted centers."""

    predicted = _read_predicted_csv(predicted_csv)
    dynerf_export = dynerf_export.resolve()
    scene_dir = scene_dir.resolve()
    out_dir = out_dir.resolve()
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    transforms_test = dynerf_export / "transforms_test.json"
    if not transforms_test.is_file():
        raise FileNotFoundError(f"Missing transforms_test.json: {transforms_test}")

    blob = json.loads(transforms_test.read_text(encoding="utf-8"))
    frames = blob.get("frames") or []
    if not frames:
        raise ValueError(f"No test frames in {transforms_test}")
    fl_x = float(blob["fl_x"])
    fl_y = float(blob["fl_y"])
    cx = float(blob["cx"])
    cy = float(blob["cy"])
    img_w = int(blob["w"])
    img_h = int(blob["h"])

    projection_mode = _calibrate_projection_mode(
        frames=frames,
        scene_dir=scene_dir,
        predicted=predicted,
        fl_x=fl_x,
        fl_y=fl_y,
        cx=cx,
        cy=cy,
        width=img_w,
        height=img_h,
    )

    missing_predictions = 0
    invalid_projections = 0
    rendered = 0
    sample_png: Path | None = None

    for idx, frame_rec in enumerate(frames):
        cam_idx, frame_idx = _parse_cam_frame_from_file_path(str(frame_rec["file_path"]))
        pred = predicted.get(frame_idx)
        if pred is None:
            missing_predictions += 1
            continue

        rgb_path = scene_dir / "rgb" / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
        if rgb_path.is_file():
            background = _read_rgb_png(rgb_path)
        else:
            # Fallback keeps output shape deterministic even if source RGB is absent.
            background = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        if mask_gt_ball:
            background = _mask_ball_from_background(
                background,
                scene_dir=scene_dir,
                cam_idx=cam_idx,
                frame_idx=frame_idx,
            )

        c2w = np.asarray(frame_rec["transform_matrix"], dtype=np.float64)
        f_fx, f_fy, f_cx, f_cy = _frame_intrinsics(
            frame_rec, fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy
        )
        u, v, depth = _project_with_mode(
            pred,
            c2w,
            fl_x=f_fx,
            fl_y=f_fy,
            cx=f_cx,
            cy=f_cy,
            width=img_w,
            height=img_h,
            projection_mode=projection_mode,
        )
        if not np.isfinite(u) or not np.isfinite(v) or depth <= 1e-6:
            invalid_projections += 1
            continue

        # Scale the projected marker using depth and known sphere radius.
        radius_px = int(round(max(2.0, (f_fx * float(sphere_radius_m)) / depth)))
        if render_style == "gaussian_splat":
            composed = _draw_gaussian_splat(
                background,
                center_u=u,
                center_v=v,
                radius_px=radius_px,
                color_rgb=overlay_color_rgb,
            )
        else:
            composed = _draw_filled_circle(
                background,
                center_u=u,
                center_v=v,
                radius_px=radius_px,
                color_rgb=overlay_color_rgb,
            )

        out_name = f"{idx:06d}_cam{cam_idx:02d}_{frame_idx:05d}.png"
        out_path = renders_dir / out_name
        _write_rgb_png(out_path, composed)
        if sample_png is None:
            sample_png = out_path
        rendered += 1

    meta = {
        "predicted_csv": str(predicted_csv.resolve()),
        "dynerf_export": str(dynerf_export),
        "scene_dir": str(scene_dir),
        "num_test_views": len(frames),
        "num_rendered": rendered,
        "num_missing_predictions": missing_predictions,
        "num_invalid_projections": invalid_projections,
        "projection_mode": projection_mode,
        "sphere_radius_m": float(sphere_radius_m),
        "overlay_color_rgb": list(map(int, overlay_color_rgb)),
        "mask_gt_ball": bool(mask_gt_ball),
        "render_style": render_style,
    }
    (out_dir / "phase4_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return Phase4Result(
        out_dir=out_dir,
        renders_dir=renders_dir,
        num_rendered=rendered,
        num_missing_predictions=missing_predictions,
        num_invalid_projections=invalid_projections,
        calibrated_projection_mode=projection_mode,
        sample_png=sample_png,
    )
