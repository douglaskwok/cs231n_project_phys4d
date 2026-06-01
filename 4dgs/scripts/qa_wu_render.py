#!/usr/bin/env python3
"""Quick QA for Wu 4DGS object-only renders.

Compares sorted native Wu render PNGs against their GT PNGs, computes a simple
foreground area ratio, and writes a small render-vs-GT contact sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _fg_area(rgb: np.ndarray, threshold: int) -> int:
    return int((rgb.max(axis=2) > threshold).sum())


def _resize(img: Image.Image, width: int) -> Image.Image:
    h = max(1, round(img.height * width / img.width))
    return img.resize((width, h), Image.Resampling.BILINEAR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tile-width", type=int, default=220)
    args = parser.parse_args()

    render_paths = sorted(args.renders.glob("*.png"))
    gt_paths = sorted(args.gt.glob("*.png"))
    if not render_paths:
        raise FileNotFoundError(f"No render PNGs under {args.renders}")
    if len(render_paths) != len(gt_paths):
        raise ValueError(
            f"Render/GT count mismatch: {len(render_paths)} vs {len(gt_paths)}"
        )

    render_areas: list[int] = []
    gt_areas: list[int] = []
    ratios: list[float] = []
    max_values: list[int] = []
    for render_path, gt_path in zip(render_paths, gt_paths):
        render = _load_rgb(render_path)
        gt = _load_rgb(gt_path)
        r_area = _fg_area(render, args.threshold)
        g_area = _fg_area(gt, args.threshold)
        render_areas.append(r_area)
        gt_areas.append(g_area)
        max_values.append(int(render.max()))
        if g_area > 0:
            ratios.append(r_area / g_area)

    valid = np.array(ratios, dtype=np.float64)
    summary = {
        "num_frames": len(render_paths),
        "threshold": args.threshold,
        "render_area_mean": float(np.mean(render_areas)),
        "render_area_max": int(np.max(render_areas)),
        "gt_area_mean": float(np.mean(gt_areas)),
        "gt_area_max": int(np.max(gt_areas)),
        "ratio_mean": float(valid.mean()) if valid.size else 0.0,
        "ratio_median": float(np.median(valid)) if valid.size else 0.0,
        "ratio_p90": float(np.percentile(valid, 90)) if valid.size else 0.0,
        "render_max_mean": float(np.mean(max_values)),
        "render_max_max": int(np.max(max_values)),
        "zero_render_frames": int(sum(area == 0 for area in render_areas)),
        "nonzero_gt_frames": int(sum(area > 0 for area in gt_areas)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    sample_count = min(args.samples, len(render_paths))
    indices = np.linspace(0, len(render_paths) - 1, sample_count, dtype=int).tolist()
    tiles: list[Image.Image] = []
    for index in indices:
        render_img = _resize(Image.open(render_paths[index]).convert("RGB"), args.tile_width)
        gt_img = _resize(Image.open(gt_paths[index]).convert("RGB"), args.tile_width)
        label_h = 24
        tile = Image.new("RGB", (args.tile_width * 2, render_img.height + label_h), "white")
        tile.paste(render_img, (0, label_h))
        tile.paste(gt_img, (args.tile_width, label_h))
        draw = ImageDraw.Draw(tile)
        ratio = render_areas[index] / gt_areas[index] if gt_areas[index] else 0.0
        draw.text((4, 4), f"{render_paths[index].stem} R/GT {ratio:.2f}", fill=(0, 0, 0))
        tiles.append(tile)

    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    cell_w = args.tile_width * 2
    cell_h = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * cell_w, (i // cols) * cell_h))
    sheet.save(args.out)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
