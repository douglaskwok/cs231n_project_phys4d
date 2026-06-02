#!/usr/bin/env python
"""Warp a trained 3DGS PLY to a target simulation frame using object_poses.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d.camera_project import load_gs_cameras  # noqa: E402
from phys4d.gaussian_ply import (  # noqa: E402
    estimate_pb_to_gs_z_scale,
    filter_sphere_gaussians_by_masks,
    filter_sphere_gaussians_by_opacity,
    load_gaussian_ply,
    save_gaussian_ply,
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
        help="Trained 3DGS checkpoint (canonical / ref frame)",
    )
    parser.add_argument("--ref-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PLY (default: outputs/.../warped_frameXXXXX.ply)",
    )
    parser.add_argument(
        "--sphere-only",
        action="store_true",
        help="Keep sphere Gaussians (opacity filter at ref; use --mask-filter for masks)",
    )
    parser.add_argument(
        "--mask-filter",
        action="store_true",
        help="Select sphere via mask projection (needs --gs-cameras; best when frame>0)",
    )
    parser.add_argument(
        "--gs-cameras",
        type=Path,
        default=REPO_ROOT / "gs_sphere_bounce/cameras.json",
        help="Trained 3DGS cameras.json for projection",
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=None,
        help="PyBullet-to-GS z scale (default: fit from ref vs frame 30)",
    )
    parser.add_argument(
        "--z-align-frame",
        type=int,
        default=30,
        help="Second timestep used to fit default z scale",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    poses_path = REPO_ROOT / cfg["outputs"]["object_poses"]
    if not args.ply.is_file():
        print(f"Missing PLY: {args.ply}", file=sys.stderr)
        return 1
    if not poses_path.is_file():
        print(f"Missing poses: {poses_path}", file=sys.stderr)
        return 1

    trajectory = load_object_poses_csv(poses_path)
    cloud = load_gaussian_ply(args.ply)

    if args.sphere_only or args.mask_filter:
        if args.mask_filter and args.gs_cameras.is_file():
            gs_cams = load_gs_cameras(args.gs_cameras)
            masks_root = REPO_ROOT / cfg["outputs"]["masks"]
            cloud = filter_sphere_gaussians_by_masks(
                cloud,
                masks_root,
                args.ref_frame,
                min_camera_hits=1,
                gs_cameras=gs_cams[:6],
            )
            print(
                f"Sphere mask filter (3DGS cams) @ frame {args.ref_frame}: "
                f"{cloud.xyz.shape[0]} Gaussians"
            )
        else:
            cloud = filter_sphere_gaussians_by_opacity(cloud)
            print(f"Sphere opacity filter: {cloud.xyz.shape[0]} Gaussians")

    z_scale = args.z_scale
    if z_scale is None:
        # Empirical alignment: high-opacity centroid z vs PyBullet z at two frames.
        ref_cloud = filter_sphere_gaussians_by_opacity(cloud) if not args.mask_filter else cloud
        gs_z_ref = float(np.mean(ref_cloud.xyz[:, 2]))
        z_scale = estimate_pb_to_gs_z_scale(
            trajectory,
            gs_z_ref,
            ref_frame=args.ref_frame,
            align_frame=args.z_align_frame,
            gs_z_at_align=0.70,  # cam00 frame30 mask-aligned centroid (~v=151)
        )
        print(f"Auto z_scale={z_scale:.4f} (pb_z -> gs_z for vertical bounce)")

    warped = warp_gaussians_to_frame(
        cloud,
        trajectory,
        ref_frame=args.ref_frame,
        target_frame=args.target_frame,
        z_scale=z_scale,
    )

    out = args.output
    if out is None:
        out = (
            REPO_ROOT
            / "outputs"
            / "sphere_bounce_m2"
            / f"warped_frame{args.target_frame:05d}.ply"
        )
    out = out.resolve()
    save_gaussian_ply(warped, out)

    target_pose = trajectory.by_frame(args.target_frame)
    mean_z = float(np.mean(warped.xyz[:, 2]))
    print(f"Wrote warped PLY: {out}")
    print(f"  ref_frame={args.ref_frame} -> target_frame={args.target_frame}")
    print(f"  pose z_m={target_pose.position[2]:.4f}  mean gaussian z={mean_z:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
