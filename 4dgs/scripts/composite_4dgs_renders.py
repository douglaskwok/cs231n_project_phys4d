#!/usr/bin/env python
"""Composite multiple object-only 4DGS render folders into one render folder.

Object-only renders use a black background, so this script treats non-black
pixels as foreground and combines matching PNG names across render directories.
The output filenames stay compatible with ``build_4dgs_render_videos.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PNG_RE = re.compile(r"^\d+_cam(\d+)_(\d+)\.png$")
SEQUENTIAL_PNG_RE = re.compile(r"^(\d+)\.png$")


def _read_rgb(path: Path) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def _foreground_mask(rgb: np.ndarray, threshold: int) -> np.ndarray:
    return np.max(rgb, axis=2) > threshold


def _composite_images(images: list[np.ndarray], threshold: int, mode: str) -> np.ndarray:
    if not images:
        raise ValueError("No images to composite.")
    out = np.zeros_like(images[0])
    if mode == "thresholded_max":
        for img in images:
            masked = img.copy()
            masked[~_foreground_mask(masked, threshold)] = 0
            out = np.maximum(out, masked)
        return out
    if mode == "max":
        for img in images:
            out = np.maximum(out, img)
        return out

    # Painter's algorithm: later render dirs are drawn over earlier ones.
    for img in images:
        mask = _foreground_mask(img, threshold)
        out[mask] = img[mask]
    return out


def _render_index(render_dir: Path) -> dict[tuple[int, int], Path]:
    out: dict[tuple[int, int], Path] = {}
    for path in sorted(render_dir.glob("*.png")):
        match = PNG_RE.match(path.name)
        if not match:
            seq_match = SEQUENTIAL_PNG_RE.match(path.name)
            if not seq_match:
                continue
            seq = int(seq_match.group(1))
            out[(0, seq)] = path
            continue
        cam, frame = map(int, match.groups())
        out[(cam, frame)] = path
    return out


def _black_like(reference: Path) -> np.ndarray:
    return np.zeros_like(_read_rgb(reference))


def composite_render_dirs(
    *,
    render_dirs: list[Path],
    out_dir: Path,
    threshold: int,
    mode: str,
    frame_policy: str,
) -> dict:
    if len(render_dirs) < 2:
        raise ValueError("Pass at least two render directories.")
    for render_dir in render_dirs:
        if not render_dir.is_dir():
            raise FileNotFoundError(render_dir)

    indexes = [_render_index(render_dir) for render_dir in render_dirs]
    if any(not idx for idx in indexes):
        empty = [str(path) for path, idx in zip(render_dirs, indexes) if not idx]
        raise FileNotFoundError(f"No render PNGs found in: {empty}")

    if frame_policy == "common":
        keys = set(indexes[0])
        for idx in indexes[1:]:
            keys &= set(idx)
    elif frame_policy == "reference":
        keys = set(indexes[0])
    else:
        keys = set()
        for idx in indexes:
            keys |= set(idx)

    sorted_keys = sorted(keys)
    if not sorted_keys:
        raise FileNotFoundError("No matching camera/frame keys across render directories.")

    input_png_counts = {}
    for render_dir in render_dirs:
        input_png_counts[str(render_dir)] = len(list(render_dir.glob("*.png")))

    out_dir.mkdir(parents=True, exist_ok=True)
    missing_by_input = {str(path): 0 for path in render_dirs}
    for out_seq, key in enumerate(sorted_keys):
        reference = next((idx[key] for idx in indexes if key in idx), None)
        if reference is None:
            continue
        images = []
        for render_dir, idx in zip(render_dirs, indexes):
            if key in idx:
                images.append(_read_rgb(idx[key]))
            else:
                missing_by_input[str(render_dir)] += 1
                images.append(_black_like(reference))
        cam, frame = key
        if all(SEQUENTIAL_PNG_RE.match(idx[key].name) for idx in indexes if key in idx):
            name = f"{frame:05d}.png"
        else:
            name = f"{out_seq:06d}_cam{cam:02d}_{frame:05d}.png"
        imageio.imwrite(out_dir / name, _composite_images(images, threshold, mode))

    meta = {
        "num_frames": len(sorted_keys),
        "mode": mode,
        "frame_policy": frame_policy,
        "threshold": threshold,
        "inputs": [str(p) for p in render_dirs],
        "input_png_counts": input_png_counts,
        "missing_frames_by_input": missing_by_input,
    }
    (out_dir / "composite_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("max", "over", "thresholded_max"),
        default="max",
        help=(
            "max preserves colored objects on black; over draws later dirs over earlier dirs; "
            "thresholded_max zeros sub-threshold pixels before max compositing."
        ),
    )
    parser.add_argument(
        "--frame-policy",
        choices=("common", "reference", "union"),
        default="reference",
        help="reference keeps the first render dir's timeline and fills missing object renders with black.",
    )
    parser.add_argument("--threshold", type=int, default=8)
    args = parser.parse_args()

    meta = composite_render_dirs(
        render_dirs=[p.resolve() for p in args.render_dirs],
        out_dir=args.out_dir.resolve(),
        threshold=args.threshold,
        mode=args.mode,
        frame_policy=args.frame_policy,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote composite renders: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
