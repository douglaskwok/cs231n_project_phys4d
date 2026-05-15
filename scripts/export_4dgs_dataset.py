#!/usr/bin/env python
"""Export full bounce video to DyNeRF layout for fudan-zvg/4d-gaussian-splatting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.dynerf_export import export_dynerf_dataset  # noqa: E402


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sphere_bounce_m2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_m2" / "dynerf_sphere_bounce",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (local only; Modal upload needs copies)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    sim = cfg["simulation"]
    cams_cfg = cfg["cameras"]
    train_frames = sim["train_frames"]
    test_frames = sim["test_frames"]

    rgb_root = REPO_ROOT / cfg["outputs"]["rgb_frames"]
    cameras_json = REPO_ROOT / cfg["outputs"]["camera_poses"]

    if not rgb_root.is_dir():
        print(f"Missing RGB tree: {rgb_root}", file=sys.stderr)
        print("Run: python scripts/generate_sphere_bounce_dataset.py", file=sys.stderr)
        return 1

    dt = float(sim.get("dt_s", 1.0 / 60.0))
    fps = 1.0 / dt

    meta = export_dynerf_dataset(
        rgb_root=rgb_root,
        cameras_json=cameras_json,
        out_dir=args.output.resolve(),
        train_cameras=list(cams_cfg["train_cameras"]),
        test_cameras=list(cams_cfg["test_cameras"]),
        train_frame_range=(int(train_frames[0]), int(train_frames[1])),
        test_frame_range=(int(test_frames[0]), int(test_frames[1])),
        fps=fps,
        fov_deg=float(cams_cfg.get("fov_deg", 60.0)),
        image_size=(
            int(cams_cfg["image_size"][0]),
            int(cams_cfg["image_size"][1]),
        ),
        copy_images=not args.symlink,
    )

    out = args.output.resolve()
    print(f"Wrote DyNeRF dataset: {out}")
    print(
        f"  train: {meta['num_train_views']} views "
        f"(cams {meta['train_cameras']}, frames {meta['train_frame_range']})"
    )
    print(
        f"  test:  {meta['num_test_views']} views "
        f"(cams {meta['test_cameras']}, frames {meta['test_frame_range']})"
    )
    print(f"  suggested time_duration: {meta['time_duration_suggested']}")
    print("Modal: modal run modal_app.py --upload-4d && modal run modal_app.py --train-4d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
