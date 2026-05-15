"""Export PyBullet multi-view video to DyNeRF / fudan 4D Gaussian Splatting layout.

Target repo: https://github.com/fudan-zvg/4d-gaussian-splatting
Expects ``transforms_train.json``, ``transforms_test.json``, and ``images/`` with
per-frame ``time`` fields (see ``scene/dataset_readers.py``).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from phys4d.colmap_export import (
    camera_angle_x_from_vertical_fov,
    view_matrix_to_c2w,
)


def _intrinsics_block(
    fov_deg: float, width: int, height: int
) -> dict[str, float | int]:
    cam_angle_x = camera_angle_x_from_vertical_fov(fov_deg, width, height)
    fl_x = float(width / (2.0 * math.tan(cam_angle_x / 2.0)))
    fl_y = float(height / (2.0 * math.tan(math.radians(fov_deg) / 2.0)))
    return {
        "w": width,
        "h": height,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "camera_angle_x": cam_angle_x,
    }


def _image_stem(cam_index: int, frame_index: int) -> str:
    return f"images/cam{cam_index:02d}_{frame_index:05d}"


def export_dynerf_dataset(
    *,
    rgb_root: Path,
    cameras_json: Path,
    out_dir: Path,
    train_cameras: list[int],
    test_cameras: list[int],
    train_frame_range: tuple[int, int],
    test_frame_range: tuple[int, int],
    fps: float = 60.0,
    fov_deg: float = 60.0,
    image_size: tuple[int, int] = (256, 256),
    copy_images: bool = True,
) -> dict[str, Any]:
    """Build a DyNeRF-style folder for fudan-zvg/4d-gaussian-splatting."""
    rgb_root = rgb_root.resolve()
    out_dir = out_dir.resolve()
    cameras_json = cameras_json.resolve()
    w, h = int(image_size[0]), int(image_size[1])

    with cameras_json.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    cam_by_index = {int(rec["index"]): rec for rec in blob["cameras"]}
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def _frame_entries(
        camera_ids: list[int], frame_lo: int, frame_hi: int
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for cam_id in camera_ids:
            if cam_id not in cam_by_index:
                raise KeyError(f"Camera {cam_id} missing from {cameras_json}")
            rec = cam_by_index[cam_id]
            c2w = view_matrix_to_c2w(rec["view_matrix_row_major"])
            for frame_idx in range(frame_lo, frame_hi + 1):
                frame_tag = f"{frame_idx:05d}"
                src = rgb_root / f"cam{cam_id:02d}" / f"frame{frame_tag}.png"
                if not src.is_file():
                    raise FileNotFoundError(f"Missing RGB frame: {src}")

                stem = _image_stem(cam_id, frame_idx)
                dst = out_dir / f"{stem}.png"
                if copy_images:
                    shutil.copy2(src, dst)
                elif not dst.exists():
                    shutil.copy2(src, dst)

                entries.append(
                    {
                        "file_path": stem,
                        "transform_matrix": c2w.tolist(),
                        "time": float(frame_idx) / float(fps),
                    }
                )
        return entries

    train_lo, train_hi = train_frame_range
    test_lo, test_hi = test_frame_range
    train_frames = _frame_entries(train_cameras, train_lo, train_hi)
    test_frames = _frame_entries(test_cameras, test_lo, test_hi)

    intrinsics = _intrinsics_block(fov_deg, w, h)
    train_transforms = {**intrinsics, "frames": train_frames}
    test_transforms = {**intrinsics, "frames": test_frames}

    with (out_dir / "transforms_train.json").open("w", encoding="utf-8") as f:
        json.dump(train_transforms, f, indent=2)
    with (out_dir / "transforms_test.json").open("w", encoding="utf-8") as f:
        json.dump(test_transforms, f, indent=2)

    t_train_end = train_hi / fps
    t_test_end = test_hi / fps
    meta = {
        "format": "dynerf",
        "fps": fps,
        "train_cameras": train_cameras,
        "test_cameras": test_cameras,
        "train_frame_range": [train_lo, train_hi],
        "test_frame_range": [test_lo, test_hi],
        "num_train_views": len(train_frames),
        "num_test_views": len(test_frames),
        "time_duration_suggested": [0.0, t_train_end],
        "test_time_range_s": [test_lo / fps, t_test_end],
        "image_size": [w, h],
        "fov_deg": fov_deg,
    }
    with (out_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
