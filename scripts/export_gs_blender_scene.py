#!/usr/bin/env python
"""Package PyBullet RGB + cameras into a 3D Gaussian Splatting Blender-style folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.colmap_export import export_blender_scene_for_3dgs  # noqa: E402


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
        "--frame",
        type=int,
        default=0,
        help="Which exported timestep to use for all cameras (static 3DGS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_m2" / "gs_blender_frame00000",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    rgb_root = REPO_ROOT / cfg["outputs"]["rgb_frames"]
    cameras_json = REPO_ROOT / cfg["outputs"]["camera_poses"]
    cams_cfg = cfg["cameras"]

    if not rgb_root.is_dir():
        print(f"Missing RGB tree: {rgb_root}", file=sys.stderr)
        print("Run: python scripts/generate_sphere_bounce_dataset.py", file=sys.stderr)
        return 1

    meta = export_blender_scene_for_3dgs(
        rgb_root=rgb_root,
        cameras_json=cameras_json,
        out_dir=args.output.resolve(),
        frame=args.frame,
        fov_deg=float(cams_cfg.get("fov_deg", 60.0)),
        image_size=(int(cams_cfg["image_size"][0]), int(cams_cfg["image_size"][1])),
        copy_images=True,
    )

    print(f"Wrote 3DGS Blender scene: {args.output.resolve()}")
    print(f"  views: {meta['num_views']}  frame: {meta['frame_index']}")
    print("Train on Modal: modal run modal_app.py --train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
