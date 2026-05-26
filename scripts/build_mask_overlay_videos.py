#!/usr/bin/env python
"""Build RGB+mask overlay MP4s for a generated ping-pong scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _read_rgb(path: Path) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr) > 0


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    red = np.array([255, 40, 40], dtype=np.uint8)
    out[mask] = (0.55 * out[mask] + 0.45 * red).astype(np.uint8)
    return out


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat(mask[..., None], 3, axis=2).astype(np.uint8) * 255


def _parse_cameras(value: str) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(12))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_frames(value: str | None, scene_root: Path, camera: int) -> list[int]:
    if value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    rgb_dir = scene_root / "rgb" / f"cam{camera:02d}"
    frames = []
    for path in sorted(rgb_dir.glob("frame*.png")):
        frames.append(int(path.stem.replace("frame", "")))
    return frames


def _write_camera_video(
    *,
    scene_root: Path,
    mask_subdir: str,
    camera: int,
    frames: list[int],
    out_path: Path,
    fps: int,
    mode: str,
) -> int:
    written = 0
    with imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            tag = f"{frame:05d}"
            rgb_path = scene_root / "rgb" / f"cam{camera:02d}" / f"frame{tag}.png"
            mask_path = scene_root / mask_subdir / f"cam{camera:02d}" / f"frame{tag}.png"
            if not rgb_path.is_file() or not mask_path.is_file():
                continue
            mask = _read_mask(mask_path)
            frame_out = _mask_rgb(mask) if mode == "mask" else _overlay(_read_rgb(rgb_path), mask)
            writer.append_data(frame_out)
            written += 1
    return written


def _write_grid_video(
    *,
    scene_root: Path,
    mask_subdir: str,
    cameras: list[int],
    frames: list[int],
    out_path: Path,
    fps: int,
    cols: int,
    mode: str,
) -> int:
    if not cameras or not frames:
        return 0
    first = _read_rgb(scene_root / "rgb" / f"cam{cameras[0]:02d}" / f"frame{frames[0]:05d}.png")
    black = np.zeros_like(first)
    rows = (len(cameras) + cols - 1) // cols
    written = 0
    with imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            tag = f"{frame:05d}"
            tiles = []
            for camera in cameras:
                rgb_path = scene_root / "rgb" / f"cam{camera:02d}" / f"frame{tag}.png"
                mask_path = scene_root / mask_subdir / f"cam{camera:02d}" / f"frame{tag}.png"
                if rgb_path.is_file() and mask_path.is_file():
                    mask = _read_mask(mask_path)
                    tiles.append(_mask_rgb(mask) if mode == "mask" else _overlay(_read_rgb(rgb_path), mask))
                else:
                    tiles.append(black)
            while len(tiles) < rows * cols:
                tiles.append(black)
            grid_rows = [
                np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1)
                for row in range(rows)
            ]
            writer.append_data(np.concatenate(grid_rows, axis=0))
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cameras", default="all")
    parser.add_argument("--frames", default=None, help="Comma-separated frames; default uses all frames from first camera")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--mask-subdir", default="masks")
    parser.add_argument("--mode", choices=["overlay", "mask"], default="overlay")
    args = parser.parse_args()

    scene_root = args.scene_root.resolve()
    cameras = _parse_cameras(args.cameras)
    frames = _parse_frames(args.frames, scene_root, cameras[0])
    out_dir = (args.out_dir or scene_root / "videos" / "overlays").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "mask" if args.mode == "mask" else "overlay"

    for camera in cameras:
        count = _write_camera_video(
            scene_root=scene_root,
            mask_subdir=args.mask_subdir,
            camera=camera,
            frames=frames,
            out_path=out_dir / f"cam{camera:02d}_{suffix}.mp4",
            fps=args.fps,
            mode=args.mode,
        )
        print(f"wrote {out_dir / f'cam{camera:02d}_{suffix}.mp4'} ({count} frames)")
    count = _write_grid_video(
        scene_root=scene_root,
        mask_subdir=args.mask_subdir,
        cameras=cameras,
        frames=frames,
        out_path=out_dir / f"all_cams_{suffix}_grid.mp4",
        fps=args.fps,
        cols=args.grid_cols,
        mode=args.mode,
    )
    print(f"wrote {out_dir / f'all_cams_{suffix}_grid.mp4'} ({count} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
