#!/usr/bin/env python
"""Create a derived object-appearance scene from existing RGB + masks.

This is useful for controlled 4DGS ablations: keep the same cameras, masks,
poses, and physics, but rewrite object pixels/background colors.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB color must be formatted as R,G,B")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB color values must be integers") from exc
    if any(v < 0 or v > 255 for v in values):
        raise argparse.ArgumentTypeError("RGB color values must be in 0..255")
    return values


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(imageio.imread(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(imageio.imread(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _copy_tree_frames(src_root: Path, dst_root: Path, max_frame: int) -> int:
    count = 0
    for src_cam in sorted(src_root.glob("cam*")):
        if not src_cam.is_dir():
            continue
        dst_cam = dst_root / src_cam.name
        dst_cam.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_cam.glob("frame*.png")):
            frame = int(src.stem.replace("frame", ""))
            if frame > max_frame:
                continue
            shutil.copy2(src, dst_cam / src.name)
            count += 1
    return count


def filter_scene(
    *,
    source_scene: Path,
    output_scene: Path,
    object_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    duration_s: float | None,
    max_frame: int | None,
    mask_subdir: str,
) -> dict:
    source_scene = source_scene.resolve()
    output_scene = output_scene.resolve()
    cfg_path = source_scene / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing source config: {cfg_path}")
    cfg = _load_json(cfg_path)
    sim = cfg["simulation"]
    fps = 1.0 / float(sim["dt_s"])

    if duration_s is not None and max_frame is not None:
        raise ValueError("Pass only one of --duration-s or --max-frame")
    if max_frame is None:
        if duration_s is None:
            max_frame = int(sim.get("num_frames", 1)) - 1
        else:
            max_frame = int(round(duration_s * fps))
    max_frame = min(max_frame, int(sim.get("num_frames", max_frame + 1)) - 1)
    if max_frame < 0:
        raise ValueError("max_frame must be nonnegative")

    rgb_src = _resolve(cfg["outputs"]["rgb_frames"])
    masks_src = source_scene / mask_subdir
    if not masks_src.is_dir():
        masks_src = _resolve(cfg["outputs"].get(mask_subdir, str(masks_src)))
    if not rgb_src.is_dir():
        raise FileNotFoundError(f"Missing RGB root: {rgb_src}")
    if not masks_src.is_dir():
        raise FileNotFoundError(f"Missing mask root: {masks_src}")

    if output_scene.exists():
        raise FileExistsError(f"Output already exists: {output_scene}")
    rgb_dst = output_scene / "rgb"
    masks_dst = output_scene / mask_subdir
    rgb_dst.mkdir(parents=True)
    masks_dst.mkdir(parents=True)

    frame_count = 0
    written_rgb = 0
    for src_cam in sorted(rgb_src.glob("cam*")):
        if not src_cam.is_dir():
            continue
        cam_name = src_cam.name
        dst_cam = rgb_dst / cam_name
        dst_cam.mkdir(parents=True, exist_ok=True)
        mask_cam = masks_src / cam_name
        if not mask_cam.is_dir():
            raise FileNotFoundError(f"Missing mask camera folder: {mask_cam}")
        for rgb_path in sorted(src_cam.glob("frame*.png")):
            frame = int(rgb_path.stem.replace("frame", ""))
            if frame > max_frame:
                continue
            mask_path = mask_cam / rgb_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            rgb = np.empty_like(_read_rgb(rgb_path))
            rgb[...] = np.asarray(background_color, dtype=np.uint8)
            mask = _read_mask(mask_path)
            rgb[mask] = np.asarray(object_color, dtype=np.uint8)
            imageio.imwrite(dst_cam / rgb_path.name, rgb)
            written_rgb += 1
        frame_count = max(frame_count, len(list(dst_cam.glob("frame*.png"))))

    copied_masks = _copy_tree_frames(masks_src, masks_dst, max_frame)
    table_masks_src = source_scene / "masks_table"
    if table_masks_src.is_dir():
        _copy_tree_frames(table_masks_src, output_scene / "masks_table", max_frame)

    for filename in ("cameras.json", "object_poses.csv"):
        src = source_scene / filename
        if src.is_file():
            shutil.copy2(src, output_scene / filename)
    assets_src = source_scene / "assets"
    if assets_src.is_dir():
        shutil.copytree(assets_src, output_scene / "assets")

    rel_scene = output_scene.relative_to(REPO_ROOT)
    cfg["simulation"]["num_frames"] = max_frame + 1
    cfg["simulation"]["duration_s"] = float(max_frame / fps)
    cfg["simulation"]["train_frames"] = [0, max_frame]
    cfg["simulation"]["test_frames"] = [0, -1]
    cfg["outputs"]["rgb_frames"] = str(rel_scene / "rgb")
    cfg["outputs"]["masks"] = str(rel_scene / mask_subdir)
    if (output_scene / "masks_table").is_dir():
        cfg["outputs"]["table_masks"] = str(rel_scene / "masks_table")
    cfg["outputs"]["camera_poses"] = str(rel_scene / "cameras.json")
    if (output_scene / "object_poses.csv").is_file():
        cfg["outputs"]["object_poses"] = str(rel_scene / "object_poses.csv")
    cfg["outputs"]["metadata"] = str(rel_scene / "metadata.json")
    cfg.setdefault("derived_from", {})["source_scene"] = str(source_scene.relative_to(REPO_ROOT))
    cfg["derived_from"]["appearance_filter"] = {
        "object_color_rgb": list(object_color),
        "background_color_rgb": list(background_color),
        "max_frame": max_frame,
        "duration_s": float(max_frame / fps),
        "mask_subdir": mask_subdir,
    }
    with (output_scene / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    metadata = {
        "source_scene": str(source_scene),
        "output_scene": str(output_scene),
        "fps": fps,
        "max_frame": max_frame,
        "duration_s": float(max_frame / fps),
        "object_color_rgb": list(object_color),
        "background_color_rgb": list(background_color),
        "rgb_frames_written": written_rgb,
        "mask_frames_copied": copied_masks,
    }
    with (output_scene / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_scene", type=Path)
    parser.add_argument("output_scene", type=Path)
    parser.add_argument("--object-color", type=_parse_rgb, default=(255, 255, 255))
    parser.add_argument("--background-color", type=_parse_rgb, default=(0, 0, 0))
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--max-frame", type=int)
    parser.add_argument("--mask-subdir", default="masks")
    args = parser.parse_args()

    meta = filter_scene(
        source_scene=args.source_scene,
        output_scene=args.output_scene,
        object_color=args.object_color,
        background_color=args.background_color,
        duration_s=args.duration_s,
        max_frame=args.max_frame,
        mask_subdir=args.mask_subdir,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
