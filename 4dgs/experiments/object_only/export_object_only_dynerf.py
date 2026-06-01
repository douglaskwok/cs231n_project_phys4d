#!/usr/bin/env python
"""Export masked/full DyNeRF datasets for fudan 4DGS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_FOURDGS_DIR = Path(__file__).resolve().parents[2]
if str(_FOURDGS_DIR) not in sys.path:
    sys.path.insert(0, str(_FOURDGS_DIR))

from _paths import REPO_ROOT  # noqa: E402


def view_matrix_to_c2w(view_col_major_16: list[float]) -> np.ndarray:
    v = np.array(view_col_major_16, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(v)


def camera_angle_x_from_vertical_fov(fov_deg: float, width: int, height: int) -> float:
    v = math.radians(float(fov_deg))
    return float(2.0 * math.atan(math.tan(v / 2.0) * (float(width) / float(height))))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _intrinsics(fov_deg: float, width: int, height: int) -> dict[str, float | int]:
    cam_angle_x = camera_angle_x_from_vertical_fov(fov_deg, width, height)
    return {
        "w": width,
        "h": height,
        "fl_x": float(width / (2.0 * math.tan(cam_angle_x / 2.0))),
        "fl_y": float(height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "camera_angle_x": cam_angle_x,
    }


def _read_rgb(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    mask = imageio.imread(path)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _apply_mask(rgb: np.ndarray, mask: np.ndarray, background: str, mode: str) -> np.ndarray:
    if mode == "full":
        return rgb.copy()

    bg_value = 255 if background == "white" else 0
    out = np.full_like(rgb, bg_value)
    keep = ~mask if mode == "background" else mask
    out[keep] = rgb[keep]
    return out


def _mask_pixels(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask frame: {path}")
    return int(_read_mask(path).sum())


def _read_frame_list(path: Path) -> list[int]:
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        frames.append(int(line))
    if not frames:
        raise ValueError(f"Frame list is empty: {path}")
    return sorted(dict.fromkeys(frames))


def _write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if rgb.shape != xyz.shape:
        raise ValueError("rgb must have shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb, strict=True):
            f.write(
                f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} "
                f"0.0 0.0 0.0 {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def _object_radius_from_config(cfg: dict[str, Any]) -> float | None:
    for obj in cfg.get("scene", {}).get("objects", []):
        if obj.get("shape") == "sphere" and "radius_m" in obj:
            return float(obj["radius_m"])
    return None


def _infer_object_name_from_masks_root(masks_root: Path) -> str | None:
    name = masks_root.name
    if name.startswith("masks_"):
        suffix = name[len("masks_") :]
        if suffix.startswith("block_"):
            return suffix
        if suffix.startswith("object_"):
            return suffix[len("object_") :]
    return None


def _box_size_from_config(cfg: dict[str, Any]) -> np.ndarray | None:
    params = cfg.get("scene", {}).get("scenario_params", {})
    if "block_size_m" in params:
        return np.asarray(params["block_size_m"], dtype=np.float32)
    for obj in cfg.get("scene", {}).get("objects", []):
        if obj.get("shape") == "box" and "size_m" in obj:
            return np.asarray(obj["size_m"], dtype=np.float32)
    return None


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-8
    x, y, z, w = [float(v) for v in q]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _read_pose_records(
    path: Path,
    frame_indices: list[int],
    *,
    object_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    wanted = set(frame_indices)
    centers: list[list[float]] = []
    rotations: list[np.ndarray] = []
    source_rows = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            frame = int(row["frame"])
            if frame not in wanted:
                continue
            if object_name is not None and row.get("object_name") != object_name:
                continue
            if int(row.get("active", "1")) == 0:
                continue
            source_rows += 1
            centers.append([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
            rotations.append(
                _quat_to_matrix(
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                )
            )
    if not centers:
        label = f" for object {object_name}" if object_name else ""
        raise ValueError(f"No active object pose rows matched exported frames{label} in {path}")
    return np.asarray(centers, dtype=np.float32), np.stack(rotations, axis=0), source_rows


def _read_pose_centers(path: Path, frame_indices: list[int]) -> np.ndarray:
    centers, _, _ = _read_pose_records(path, frame_indices)
    return centers


def _sample_box_offsets(
    *,
    rng: np.random.Generator,
    init_points: int,
    size_m: np.ndarray,
    init_surface_ratio: float,
) -> np.ndarray:
    half = np.asarray(size_m, dtype=np.float32) / 2.0
    init_surface_ratio = float(np.clip(init_surface_ratio, 0.0, 1.0))
    surface_count = int(round(init_points * init_surface_ratio))
    interior_count = init_points - surface_count
    offsets = np.empty((init_points, 3), dtype=np.float32)

    if surface_count:
        pts = rng.uniform(-half, half, size=(surface_count, 3)).astype(np.float32)
        face_areas = np.asarray(
            [half[1] * half[2], half[0] * half[2], half[0] * half[1]],
            dtype=np.float32,
        )
        probs = face_areas / (face_areas.sum() + 1e-8)
        axes = rng.choice(3, size=surface_count, p=probs)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=surface_count)
        pts[np.arange(surface_count), axes] = signs * half[axes]
        offsets[:surface_count] = pts
    if interior_count:
        offsets[surface_count:] = rng.uniform(-half, half, size=(interior_count, 3)).astype(np.float32)
    return offsets[rng.permutation(init_points)]


def _mean_masked_color(
    *,
    rgb_root: Path,
    masks_root: Path,
    camera_ids: list[int],
    frame_indices: list[int],
    max_samples: int = 24,
) -> np.ndarray:
    if not camera_ids or not frame_indices:
        return np.array([220, 230, 235], dtype=np.uint8)
    sample_frames = np.linspace(0, len(frame_indices) - 1, min(max_samples, len(frame_indices)), dtype=int)
    colors: list[np.ndarray] = []
    for cam_id in camera_ids:
        for frame_pos in sample_frames:
            frame_idx = frame_indices[int(frame_pos)]
            tag = f"{frame_idx:05d}"
            rgb_path = rgb_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
            mask_path = masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
            if not rgb_path.is_file() or not mask_path.is_file():
                continue
            mask = _read_mask(mask_path)
            if not mask.any():
                continue
            colors.append(_read_rgb(rgb_path)[mask].mean(axis=0))
    if not colors:
        return np.array([220, 230, 235], dtype=np.uint8)
    return np.clip(np.mean(colors, axis=0), 0, 255).astype(np.uint8)


def _write_object_pose_fused_ply(
    *,
    cfg: dict[str, Any],
    scene_dir: Path,
    output: Path,
    rgb_root: Path,
    masks_root: Path,
    camera_ids: list[int],
    frame_indices: list[int],
    init_points: int,
    init_center_mode: str,
    init_surface_ratio: float,
) -> dict[str, Any] | None:
    if init_points <= 0:
        return None
    pose_rel = cfg.get("outputs", {}).get("object_poses")
    if not pose_rel:
        raise ValueError("--init-points requires outputs.object_poses in config.json")
    local_pose_path = scene_dir / "object_poses.csv"
    pose_path = local_pose_path.resolve() if local_pose_path.is_file() else (REPO_ROOT / pose_rel).resolve()
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing object poses for --init-points: {pose_path}")
    radius = _object_radius_from_config(cfg)
    box_size = _box_size_from_config(cfg)
    object_name = _infer_object_name_from_masks_root(masks_root)

    centers, rotations, source_num_centers = _read_pose_records(
        pose_path,
        frame_indices,
        object_name=object_name,
    )
    source_num_centers = int(len(centers))
    if init_center_mode == "first":
        centers = centers[:1]
        rotations = rotations[:1]
    elif init_center_mode == "middle":
        mid = len(centers) // 2
        centers = centers[mid : mid + 1]
        rotations = rotations[mid : mid + 1]
    elif init_center_mode != "all":
        raise ValueError(f"Unknown init_center_mode: {init_center_mode}")

    rng = np.random.default_rng(7)
    center_idx = rng.integers(0, len(centers), size=init_points)

    init_surface_ratio = float(np.clip(init_surface_ratio, 0.0, 1.0))
    shape_meta: dict[str, Any]
    if radius is not None:
        dirs = rng.normal(size=(init_points, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        surface_count = int(round(init_points * init_surface_ratio))
        interior_count = init_points - surface_count
        radii = np.empty((init_points, 1), dtype=np.float32)
        if surface_count:
            radii[:surface_count] = float(radius)
        if interior_count:
            radii[surface_count:] = (
                rng.random(interior_count, dtype=np.float32) ** (1.0 / 3.0)
            )[:, None] * float(radius)
        order = rng.permutation(init_points)
        offsets = dirs[order] * radii[order]
        center_idx = center_idx[order]
        shape_meta = {
            "shape": "sphere",
            "radius_m": float(radius),
            "surface_points": int(surface_count),
            "interior_points": int(interior_count),
        }
    elif box_size is not None:
        offsets = _sample_box_offsets(
            rng=rng,
            init_points=init_points,
            size_m=box_size,
            init_surface_ratio=init_surface_ratio,
        )
        surface_count = int(round(init_points * init_surface_ratio))
        interior_count = init_points - surface_count
        shape_meta = {
            "shape": "box",
            "size_m": [float(v) for v in box_size.tolist()],
            "surface_points": int(surface_count),
            "interior_points": int(interior_count),
            "object_name": object_name,
        }
    else:
        raise ValueError(
            "--init-points requires either a sphere radius_m or a box/block_size_m in config.json"
        )

    xyz = centers[center_idx] + np.einsum("nij,nj->ni", rotations[center_idx], offsets)

    color = _mean_masked_color(
        rgb_root=rgb_root,
        masks_root=masks_root,
        camera_ids=camera_ids,
        frame_indices=frame_indices,
    )
    jitter = rng.normal(0.0, 4.0, size=(init_points, 3))
    rgb = np.clip(color[None, :].astype(np.float32) + jitter, 0, 255).astype(np.uint8)
    fused_path = output / "fused.ply"
    _write_ascii_ply(fused_path, xyz, rgb)
    return {
        "path": str(fused_path),
        "num_points": int(init_points),
        "source": "object_poses",
        "init_center_mode": init_center_mode,
        "init_surface_ratio": init_surface_ratio,
        "num_pose_centers": int(len(centers)),
        "source_num_pose_centers": source_num_centers,
        "mean_color_rgb": [int(v) for v in color.tolist()],
        **shape_meta,
    }


def export_object_only_dynerf(
    *,
    config: Path,
    masks_root: Path,
    output: Path,
    background: str,
    mode: str = "object",
    min_mask_pixels: int = 0,
    min_visible_cameras: int = 1,
    trim_empty_time_ends: bool = False,
    drop_invisible_frames: bool = False,
    drop_invisible_views: bool = False,
    frame_list: Path | None = None,
    all_train: bool = False,
    init_points: int = 0,
    init_center_mode: str = "all",
    init_surface_ratio: float = 0.0,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    cfg = _load_json(config)
    sim = cfg["simulation"]
    cams_cfg = cfg["cameras"]
    train_frames = sim["train_frames"]
    test_frames = sim["test_frames"]

    scene_dir = config.parent
    local_rgb_root = scene_dir / "rgb"
    local_cameras_json = scene_dir / "cameras.json"
    rgb_root = local_rgb_root.resolve() if local_rgb_root.is_dir() else (REPO_ROOT / cfg["outputs"]["rgb_frames"]).resolve()
    cameras_json = (
        local_cameras_json.resolve()
        if local_cameras_json.is_file()
        else (REPO_ROOT / cfg["outputs"]["camera_poses"]).resolve()
    )
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"Missing RGB root: {rgb_root}")
    if not cameras_json.is_file():
        raise FileNotFoundError(f"Missing cameras JSON: {cameras_json}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Missing masks root: {masks_root}")

    cam_blob = _load_json(cameras_json)
    cam_by_index = {int(rec["index"]): rec for rec in cam_blob["cameras"]}

    width = int(cams_cfg["image_size"][0])
    height = int(cams_cfg["image_size"][1])
    fov_deg = float(cams_cfg.get("fov_deg", 60.0))
    fps = 1.0 / float(sim.get("dt_s", 1.0 / 60.0))

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def visible_camera_count(camera_ids: list[int], frame_idx: int) -> int:
        tag = f"{frame_idx:05d}"
        return sum(
            _mask_pixels(masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png")
            > min_mask_pixels
            for cam_id in camera_ids
        )

    def visible_frame_indices(camera_ids: list[int], lo: int, hi: int) -> list[int]:
        visible = []
        for frame_idx in range(lo, hi + 1):
            if visible_camera_count(camera_ids, frame_idx) >= min_visible_cameras:
                visible.append(frame_idx)
        return visible

    def effective_range(camera_ids: list[int], lo: int, hi: int) -> tuple[int, int]:
        if not trim_empty_time_ends:
            return lo, hi
        visible = visible_frame_indices(camera_ids, lo, hi)
        if not visible:
            raise ValueError(
                f"No visible timestamps in frame range {lo}..{hi} "
                f"for min_mask_pixels={min_mask_pixels}"
            )
        return visible[0], visible[-1]

    def frame_entries(
        camera_ids: list[int],
        frame_indices: list[int],
    ) -> tuple[list[dict], dict[str, int]]:
        entries: list[dict] = []
        stats = {
            "empty_masks": 0,
            "below_threshold_masks": 0,
            "dropped_invisible_views": 0,
            "written_views": 0,
        }
        for cam_id in camera_ids:
            rec = cam_by_index[cam_id]
            c2w = view_matrix_to_c2w(rec["view_matrix_row_major"])
            for frame_idx in frame_indices:
                tag = f"{frame_idx:05d}"
                rgb_path = rgb_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
                mask_path = masks_root / f"cam{cam_id:02d}" / f"frame{tag}.png"
                if not rgb_path.is_file():
                    raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask frame: {mask_path}")

                rgb = _read_rgb(rgb_path)
                mask = _read_mask(mask_path)
                mask_area = int(mask.sum())
                if mask_area == 0:
                    stats["empty_masks"] += 1
                if mask_area <= min_mask_pixels:
                    stats["below_threshold_masks"] += 1
                    if drop_invisible_views and mode != "full":
                        stats["dropped_invisible_views"] += 1
                        continue
                masked = _apply_mask(rgb, mask, background, mode)

                stem = f"images/cam{cam_id:02d}_{frame_idx:05d}"
                imageio.imwrite(output / f"{stem}.png", masked)
                frame_intr = _intrinsics(float(rec.get("fov_deg", fov_deg)), width, height)
                entries.append(
                    {
                        **frame_intr,
                        "file_path": stem,
                        "transform_matrix": c2w.tolist(),
                        "time": float(frame_idx) / float(fps),
                        "original_frame": int(frame_idx),
                        "mask_pixels": mask_area,
                    }
                )
                stats["written_views"] += 1
        return entries, stats

    if all_train:
        train_cameras = sorted(cam_by_index)
        test_cameras: list[int] = []
        train_lo = 0
        train_hi = int(sim.get("num_frames", max(int(train_frames[1]), int(test_frames[1])) + 1)) - 1
        test_lo = 0
        test_hi = -1
        fixed_frame_indices = _read_frame_list(frame_list) if frame_list is not None else None
        if fixed_frame_indices is not None:
            out_of_range = [idx for idx in fixed_frame_indices if idx < train_lo or idx > train_hi]
            if out_of_range:
                raise ValueError(
                    f"--frame-list contains frames outside dataset range {train_lo}..{train_hi}: "
                    f"{out_of_range[:10]}"
                )
            train_frame_indices = fixed_frame_indices
        elif drop_invisible_frames and mode != "full":
            train_frame_indices = visible_frame_indices(train_cameras, train_lo, train_hi)
        else:
            train_frame_indices = list(range(train_lo, train_hi + 1))
        test_frame_indices: list[int] = []
    else:
        train_cameras = list(cams_cfg["train_cameras"])
        test_cameras = list(cams_cfg["test_cameras"])
        timeline_cameras = sorted(set(train_cameras + test_cameras))
        train_lo, train_hi = effective_range(timeline_cameras, int(train_frames[0]), int(train_frames[1]))
        test_lo, test_hi = effective_range(timeline_cameras, int(test_frames[0]), int(test_frames[1]))
        fixed_frame_indices = _read_frame_list(frame_list) if frame_list is not None else None
        if fixed_frame_indices is not None:
            all_lo = min(int(train_frames[0]), int(test_frames[0]))
            all_hi = max(int(train_frames[1]), int(test_frames[1]))
            out_of_range = [idx for idx in fixed_frame_indices if idx < all_lo or idx > all_hi]
            if out_of_range:
                raise ValueError(
                    f"--frame-list contains frames outside dataset range {all_lo}..{all_hi}: "
                    f"{out_of_range[:10]}"
                )
            train_frame_indices = [
                idx for idx in fixed_frame_indices if int(train_frames[0]) <= idx <= int(train_frames[1])
            ]
            test_frame_indices = [
                idx for idx in fixed_frame_indices if int(test_frames[0]) <= idx <= int(test_frames[1])
            ]
        elif drop_invisible_frames and mode != "full":
            train_frame_indices = visible_frame_indices(timeline_cameras, train_lo, train_hi)
            test_frame_indices = visible_frame_indices(timeline_cameras, test_lo, test_hi)
        else:
            train_frame_indices = list(range(train_lo, train_hi + 1))
            test_frame_indices = list(range(test_lo, test_hi + 1))

    train, train_empty = frame_entries(train_cameras, train_frame_indices)
    test, test_empty = frame_entries(test_cameras, test_frame_indices)
    init_point_cloud = None
    if mode == "object":
        init_point_cloud = _write_object_pose_fused_ply(
            cfg=cfg,
            scene_dir=scene_dir,
            output=output,
            rgb_root=rgb_root,
            masks_root=masks_root,
            camera_ids=train_cameras,
            frame_indices=train_frame_indices,
            init_points=init_points,
            init_center_mode=init_center_mode,
            init_surface_ratio=init_surface_ratio,
        )

    intr = _intrinsics(fov_deg, width, height)
    with (output / "transforms_train.json").open("w", encoding="utf-8") as f:
        json.dump({**intr, "frames": train}, f, indent=2)
    with (output / "transforms_test.json").open("w", encoding="utf-8") as f:
        json.dump({**intr, "frames": test}, f, indent=2)

    meta = {
        "format": f"dynerf_{mode}",
        "source_config": str(config),
        "rgb_root": str(rgb_root),
        "masks_root": str(masks_root),
        "mode": mode,
        "background": background,
        "fps": fps,
        "train_cameras": train_cameras,
        "test_cameras": test_cameras,
        "train_frame_range": [int(train_frames[0]), int(train_frames[1])],
        "test_frame_range": [int(test_frames[0]), int(test_frames[1])],
        "effective_train_frame_range": [train_lo, train_hi],
        "effective_test_frame_range": [test_lo, test_hi],
        "num_train_views": len(train),
        "num_test_views": len(test),
        "num_train_timestamps": len(train_frame_indices),
        "num_test_timestamps": len(test_frame_indices),
        "train_timestamps": train_frame_indices,
        "test_timestamps": test_frame_indices,
        "empty_train_masks": train_empty["empty_masks"],
        "empty_test_masks": test_empty["empty_masks"],
        "below_threshold_train_masks": train_empty["below_threshold_masks"],
        "below_threshold_test_masks": test_empty["below_threshold_masks"],
        "dropped_train_views": train_empty["dropped_invisible_views"],
        "dropped_test_views": test_empty["dropped_invisible_views"],
        "min_mask_pixels": min_mask_pixels,
        "min_visible_cameras": min_visible_cameras,
        "trim_empty_time_ends": trim_empty_time_ends,
        "drop_invisible_frames": drop_invisible_frames,
        "drop_invisible_views": drop_invisible_views,
        "frame_list": str(frame_list) if frame_list is not None else None,
        "all_train": all_train,
        "init_point_cloud": init_point_cloud,
        "preserves_original_timestamps": True,
        "preserves_synchronized_multiview_frames": not drop_invisible_views,
        "image_size": [width, height],
        "fov_deg": fov_deg,
        "time_duration_suggested": [
            float(train_frame_indices[0]) / fps if train_frame_indices else 0.0,
            float(train_frame_indices[-1]) / fps if train_frame_indices else 0.0,
        ],
        "effective_time_range_s": [
            float(train_frame_indices[0]) / fps if train_frame_indices else 0.0,
            float(train_frame_indices[-1]) / fps if train_frame_indices else 0.0,
        ],
    }
    with (output / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    frame_map = {
        "fps": fps,
        "train": [
            {
                "kept_index": kept_idx,
                "original_frame": int(frame_idx),
                "original_time_s": float(frame_idx) / fps,
            }
            for kept_idx, frame_idx in enumerate(train_frame_indices)
        ],
        "test": [
            {
                "kept_index": kept_idx,
                "original_frame": int(frame_idx),
                "original_time_s": float(frame_idx) / fps,
            }
            for kept_idx, frame_idx in enumerate(test_frame_indices)
        ],
    }
    with (output / "frame_map.json").open("w", encoding="utf-8") as f:
        json.dump(frame_map, f, indent=2)
    return meta


def run_smoke_test() -> int:
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rgb = root / "rgb" / "cam00"
        masks = root / "masks" / "cam00"
        rgb.mkdir(parents=True)
        masks.mkdir(parents=True)
        image = np.full((32, 32, 3), 50, dtype=np.uint8)
        image[12:20, 12:20] = [30, 90, 220]
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        imageio.imwrite(rgb / "frame00000.png", image)
        imageio.imwrite(masks / "frame00000.png", mask)

        cameras = {
            "cameras": [
                {
                    "index": 0,
                    "view_matrix_row_major": np.eye(4).reshape(-1).tolist(),
                }
            ]
        }
        (root / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
        cfg = {
            "simulation": {"train_frames": [0, 0], "test_frames": [0, 0], "dt_s": 1 / 60},
            "cameras": {
                "train_cameras": [0],
                "test_cameras": [0],
                "image_size": [32, 32],
                "fov_deg": 60.0,
            },
            "outputs": {
                "rgb_frames": str(rgb.parent),
                "camera_poses": str(root / "cameras.json"),
            },
        }
        cfg_path = root / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        expectations = {
            "object": ((0, 0, 0), (30, 90, 220)),
            "background": ((50, 50, 50), (0, 0, 0)),
            "full": ((50, 50, 50), (30, 90, 220)),
        }
        for mode, (bg_expected, obj_expected) in expectations.items():
            out_dir = root / f"out_{mode}"
            meta = export_object_only_dynerf(
                config=cfg_path,
                masks_root=masks.parent,
                output=out_dir,
                background="black",
                mode=mode,
            )
            print(json.dumps(meta, indent=2))
            out_img = imageio.imread(out_dir / "images" / "cam00_00000.png")
            if tuple(out_img[0, 0]) != bg_expected or tuple(out_img[15, 15]) != obj_expected:
                raise RuntimeError(f"Smoke masked image failed for mode={mode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sphere_bounce_m2.json")
    parser.add_argument("--masks-root", type=Path, default=REPO_ROOT / "outputs" / "sphere_bounce_m2" / "masks")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "4dgs" / "experiments" / "object_only" / "runs" / "dynerf_ball_only",
    )
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=0,
        help="Masks with <= this many foreground pixels are treated as invisible.",
    )
    parser.add_argument(
        "--min-visible-cameras",
        type=int,
        default=1,
        help=(
            "For --drop-invisible-frames/--trim-empty-time-ends, keep a timestamp "
            "only if at least this many cameras have mask area > --min-mask-pixels."
        ),
    )
    parser.add_argument(
        "--trim-empty-time-ends",
        action="store_true",
        help="Trim leading/trailing timestamps where the mask is invisible in every selected camera.",
    )
    parser.add_argument(
        "--drop-invisible-frames",
        action="store_true",
        help=(
            "Drop whole timestamps that fail --min-visible-cameras. This preserves "
            "synchronized multi-view frame packets for 4DGS."
        ),
    )
    parser.add_argument(
        "--drop-invisible-views",
        action="store_true",
        help=(
            "Drop individual camera-time entries whose mask area is <= --min-mask-pixels. "
            "This breaks synchronized multi-view frame packets; use --drop-invisible-frames for 4DGS."
        ),
    )
    parser.add_argument(
        "--frame-list",
        type=Path,
        default=None,
        help=(
            "Optional text file containing one original frame index per line. "
            "Use this to force multiple per-object exports onto the exact same timeline."
        ),
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help=(
            "Ignore config train/test split: put all selected cameras and all scene frames "
            "into transforms_train.json, leaving transforms_test.json empty. Use this for "
            "final reconstruction/viewing rather than held-out evaluation."
        ),
    )
    parser.add_argument(
        "--init-points",
        type=int,
        default=0,
        help=(
            "Write a Wu/HUSTVL fused.ply initialization with this many points, sampled "
            "from PyBullet object_poses.csv. Use for small object-only Wu 4DGS runs."
        ),
    )
    parser.add_argument(
        "--init-center-mode",
        choices=("all", "first", "middle"),
        default="all",
        help=(
            "Which object pose centers to use for --init-points. 'all' preserves the old "
            "trajectory-shaped initialization; 'first'/'middle' initialize one canonical sphere."
        ),
    )
    parser.add_argument(
        "--init-surface-ratio",
        type=float,
        default=0.0,
        help="Fraction of --init-points placed on the sphere surface instead of inside it.",
    )
    parser.add_argument(
        "--mode",
        choices=("object", "background", "full"),
        default="object",
        help="object keeps mask pixels, background removes mask pixels, full keeps original RGB.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        return run_smoke_test()

    meta = export_object_only_dynerf(
        config=args.config.resolve(),
        masks_root=args.masks_root.resolve(),
        output=args.output.resolve(),
        background=args.background,
        mode=args.mode,
        min_mask_pixels=args.min_mask_pixels,
        min_visible_cameras=args.min_visible_cameras,
        trim_empty_time_ends=args.trim_empty_time_ends,
        drop_invisible_frames=args.drop_invisible_frames,
        drop_invisible_views=args.drop_invisible_views,
        frame_list=args.frame_list.resolve() if args.frame_list is not None else None,
        all_train=args.all_train,
        init_points=args.init_points,
        init_center_mode=args.init_center_mode,
        init_surface_ratio=args.init_surface_ratio,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.mode} DyNeRF dataset: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
