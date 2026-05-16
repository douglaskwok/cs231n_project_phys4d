#!/usr/bin/env python
"""Phase 2: extract per-frame states + t=0 visual features for all batch scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.perception.extract import extract_scene_perception  # noqa: E402
from phys4d.visual_dynamics.scenes import discover_batch_scenes, load_manifest_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/visual_dynamics.json",
    )
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    batch_root = args.batch_root or (REPO_ROOT / cfg["data"]["batch_root"])
    scenes = discover_batch_scenes(batch_root)
    if args.limit:
        scenes = scenes[: args.limit]

    for i, scene in enumerate(scenes):
        print(f"[{i + 1}/{len(scenes)}] {scene.scene_id}")
        extract_scene_perception(
            scene.scene_dir,
            dt_s=float(cfg["temporal"]["dt_s"]),
            visual_frame=int(cfg["perception"]["visual_feature_frame"]),
            image_size=int(cfg["perception"]["image_size"]),
            max_views=int(cfg["perception"]["max_views"]),
            device=args.device,
        )
    print(f"Done. Per-scene outputs under {batch_root}/*/perception/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
