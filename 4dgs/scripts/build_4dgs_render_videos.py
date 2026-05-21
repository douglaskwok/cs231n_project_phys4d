#!/usr/bin/env python
"""Build MP4 previews from downloaded 4DGS render PNGs.

Modal render downloads can contain only files like:

    000000_cam00_00000.png
    000001_cam01_00000.png

This script makes one MP4 per camera and optional grid previews.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


PNG_RE = re.compile(r"^(\d+)_cam(\d+)_(\d+)\.png$")


def _load_render_index(render_dir: Path) -> tuple[dict[int, list[tuple[int, Path]]], dict[int, dict[int, Path]]]:
    by_cam: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    by_frame: dict[int, dict[int, Path]] = defaultdict(dict)
    for path in sorted(render_dir.glob("*.png")):
        match = PNG_RE.match(path.name)
        if not match:
            continue
        _seq, cam, frame = map(int, match.groups())
        by_cam[cam].append((frame, path))
        by_frame[frame][cam] = path
    return by_cam, by_frame


def _write_camera_videos(by_cam: dict[int, list[tuple[int, Path]]], out_dir: Path, fps: int) -> None:
    for cam, rows in sorted(by_cam.items()):
        rows = sorted(rows)
        dst = out_dir / f"cam{cam:02d}.mp4"
        with imageio.get_writer(
            dst,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for _frame, path in rows:
                writer.append_data(imageio.imread(path)[..., :3])
        print(f"wrote {dst} ({len(rows)} frames)")


def _write_grid_video(
    *,
    by_frame: dict[int, dict[int, Path]],
    frames: list[int],
    cams: list[int],
    dst: Path,
    cols: int,
    fps: int,
) -> None:
    if not frames or not cams:
        return
    first_path = next((by_frame[frames[0]].get(cam) for cam in cams if cam in by_frame[frames[0]]), None)
    if first_path is None:
        return
    first = imageio.imread(first_path)[..., :3]
    rows = (len(cams) + cols - 1) // cols
    black = np.zeros_like(first)
    with imageio.get_writer(
        dst,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            tiles = [
                imageio.imread(by_frame[frame][cam])[..., :3]
                if cam in by_frame.get(frame, {})
                else black
                for cam in cams
            ]
            while len(tiles) < rows * cols:
                tiles.append(black)
            grid_rows = [
                np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1)
                for row in range(rows)
            ]
            writer.append_data(np.concatenate(grid_rows, axis=0))
    print(f"wrote {dst} ({len(frames)} frames)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path, nargs="?", default=Path("latest"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--train-cams", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--test-cams", default="10,11")
    args = parser.parse_args()

    render_dir = args.render_dir.resolve()
    out_dir = (args.out_dir or render_dir / "videos").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cam, by_frame = _load_render_index(render_dir)
    if not by_cam:
        raise FileNotFoundError(f"No 4DGS render PNGs found in {render_dir}")

    print(f"found {sum(len(v) for v in by_cam.values())} PNGs across cameras {sorted(by_cam)}")
    _write_camera_videos(by_cam, out_dir, args.fps)

    train_cams = [int(x) for x in args.train_cams.split(",") if x.strip()]
    test_cams = [int(x) for x in args.test_cams.split(",") if x.strip()]
    train_frames = sorted(
        frame for frame, cams in by_frame.items() if any(cam in cams for cam in train_cams)
    )
    test_frames = sorted(
        frame for frame, cams in by_frame.items() if any(cam in cams for cam in test_cams)
    )
    _write_grid_video(
        by_frame=by_frame,
        frames=train_frames,
        cams=train_cams,
        dst=out_dir / "train_cams_grid.mp4",
        cols=5,
        fps=args.fps,
    )
    _write_grid_video(
        by_frame=by_frame,
        frames=test_frames,
        cams=test_cams,
        dst=out_dir / "test_cams_grid.mp4",
        cols=max(1, len(test_cams)),
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
