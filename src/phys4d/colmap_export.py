"""Export PyBullet multi-view RGB to a 3DGS Blender-style scene folder.

Official graphdeco-inria/gaussian-splatting loads ``transforms_train.json`` when
``sparse/`` is absent (see ``scene/__init__.py``). This avoids COLMAP SfM for
our simulator-calibrated cameras.

Uses one timestep (default frame 0): all cameras, static scene — not full video.
For temporal dynamics use a 4DGS fork later.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def view_matrix_to_c2w(view_col_major_16: list[float]) -> np.ndarray:
    """Invert PyBullet OpenGL view matrix (column-major 4x4) -> camera-to-world."""
    v = np.array(view_col_major_16, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(v)


def camera_angle_x_from_vertical_fov(fov_deg: float, width: int, height: int) -> float:
    v = math.radians(float(fov_deg))
    return float(2.0 * math.atan(math.tan(v / 2.0) * (float(width) / float(height))))


def export_blender_scene_for_3dgs(
    *,
    rgb_root: Path,
    cameras_json: Path,
    out_dir: Path,
    frame: int = 0,
    fov_deg: float = 60.0,
    image_size: tuple[int, int] = (256, 256),
    copy_images: bool = True,
) -> dict[str, Any]:
    """Build ``out_dir`` with ``transforms_train.json`` and ``train/*.png``.

    Images are read from ``rgb_root/camXX/frameYYYYY.png`` and written as
    ``train/camXX.png`` (paths in JSON omit the ``.png`` extension, per 3DGS).
    """
    rgb_root = rgb_root.resolve()
    out_dir = out_dir.resolve()
    cameras_json = cameras_json.resolve()
    w, h = int(image_size[0]), int(image_size[1])
    frame_tag = f"{frame:05d}"

    with cameras_json.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    for rec in blob["cameras"]:
        ci = int(rec["index"])
        src = rgb_root / f"cam{ci:02d}" / f"frame{frame_tag}.png"
        if not src.is_file():
            raise FileNotFoundError(f"Missing RGB frame: {src}")

        stem = f"train/cam{ci:02d}"
        dst = out_dir / f"{stem}.png"
        if copy_images:
            shutil.copy2(src, dst)
        else:
            if not dst.exists():
                shutil.copy2(src, dst)

        c2w = view_matrix_to_c2w(rec["view_matrix_row_major"])
        frames.append({"file_path": stem, "transform_matrix": c2w.tolist()})

    cam_angle_x = camera_angle_x_from_vertical_fov(fov_deg, w, h)
    transforms = {
        "camera_angle_x": cam_angle_x,
        "frames": frames,
    }
    with (out_dir / "transforms_train.json").open("w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    # 3DGS also looks for transforms_test.json when eval=True; duplicate for simplicity.
    with (out_dir / "transforms_test.json").open("w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    meta = {
        "frame_index": frame,
        "num_views": len(frames),
        "image_size": [w, h],
        "fov_deg": fov_deg,
        "note": "Static multi-view slice from PyBullet export; not full video sequence.",
    }
    with (out_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
