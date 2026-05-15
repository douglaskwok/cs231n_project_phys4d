#!/usr/bin/env python
"""Project warped sphere Gaussians into masks; measure foreground hit rate per camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.camera_project import load_gs_cameras, mask_coverage_fraction  # noqa: E402
from phys4d.gaussian_ply import (  # noqa: E402
    estimate_pb_to_gs_z_scale,
    filter_sphere_gaussians_by_opacity,
    load_gaussian_ply,
    warp_gaussians_to_frame,
)
from phys4d.poses import load_object_poses_csv  # noqa: E402


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
        "--ply",
        type=Path,
        default=REPO_ROOT / "gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply",
    )
    parser.add_argument("--ref-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=30)
    parser.add_argument(
        "--cameras",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated camera indices (matches gs cameras.json id)",
    )
    parser.add_argument(
        "--gs-cameras",
        type=Path,
        default=REPO_ROOT / "gs_sphere_bounce/cameras.json",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Subsample for speed (0 = use all)",
    )
    parser.add_argument("--z-scale", type=float, default=None)
    parser.add_argument("--z-align-frame", type=int, default=30)
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    poses_path = REPO_ROOT / cfg["outputs"]["object_poses"]
    masks_root = REPO_ROOT / cfg["outputs"]["masks"]
    if not args.gs_cameras.is_file():
        print(f"Missing --gs-cameras: {args.gs_cameras}", file=sys.stderr)
        return 1

    if not args.ply.is_file():
        print(f"Missing PLY: {args.ply}", file=sys.stderr)
        return 1

    trajectory = load_object_poses_csv(poses_path)
    gs_cams = {int(c.get("id", i)): c for i, c in enumerate(load_gs_cameras(args.gs_cameras)[:6])}
    import numpy as np

    cloud = filter_sphere_gaussians_by_opacity(load_gaussian_ply(args.ply))
    print(f"Using {cloud.xyz.shape[0]} sphere Gaussians (opacity filter)")

    z_scale = args.z_scale
    if z_scale is None:
        gs_z_ref = float(np.mean(cloud.xyz[:, 2]))
        z_scale = estimate_pb_to_gs_z_scale(
            trajectory,
            gs_z_ref,
            ref_frame=args.ref_frame,
            align_frame=args.z_align_frame,
            gs_z_at_align=0.70,
        )

    warped = warp_gaussians_to_frame(
        cloud,
        trajectory,
        ref_frame=args.ref_frame,
        target_frame=args.target_frame,
        z_scale=z_scale,
    )

    pts = warped.xyz
    if args.max_points > 0 and pts.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=args.max_points, replace=False)
        pts = pts[idx]

    cam_indices = [int(x.strip()) for x in args.cameras.split(",") if x.strip()]
    frame_tag = f"frame{args.target_frame:05d}.png"

    print(
        f"Mask coverage (warped sphere centers, frame {args.target_frame}, "
        f"ref {args.ref_frame})"
    )
    for ci in cam_indices:
        if ci not in gs_cams:
            print(f"  cam{ci:02d}: missing in gs cameras.json")
            continue
        cam = gs_cams[ci]
        mask_path = masks_root / f"cam{ci:02d}" / frame_tag
        if not mask_path.is_file():
            print(f"  cam{ci:02d}: missing mask {mask_path}")
            continue
        frac = mask_coverage_fraction(pts, mask_path, gs_camera=cam)
        print(f"  cam{ci:02d}: {frac * 100:.1f}% of projected points on foreground mask")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
