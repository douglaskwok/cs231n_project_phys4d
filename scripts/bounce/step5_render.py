#!/usr/bin/env python
"""Step 5 — composite translated canonical object Gaussians with background and render test views.

Example::

  python scripts/bounce/step5_render.py \\
    --canonical dataset/.../4dgs_wu/.../point_cloud.ply \\
    --bg-ply dataset/.../background_3dgs/background.ply \\
    --predicted outputs/bounce_pipeline/<RUN>/step4c/trajectory_predicted.csv \\
    --ref-traj outputs/bounce_pipeline/<RUN>/step4a/trajectory_smoothed.csv \\
    --dynerf-export 4dgs/experiments/object_only/runs/<RUN> \\
    --cfg-args dataset/.../4dgs_wu/.../cfg_args \\
    --scene-dir dataset/.../scene_<id> \\
    --out outputs/bounce_pipeline/<RUN>/step5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.render_compose import run_step5  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True, help="Wu object point_cloud.ply")
    parser.add_argument("--bg-ply", type=Path, required=True, help="Static background 3DGS PLY")
    parser.add_argument("--predicted", type=Path, required=True, help="step4c/trajectory_predicted.csv")
    parser.add_argument(
        "--ref-traj",
        type=Path,
        required=True,
        help="step4a/trajectory_smoothed.csv (defines p_ref = first row)",
    )
    parser.add_argument(
        "--dynerf-export",
        type=Path,
        required=True,
        help="Folder with transforms_test.json (camera poses)",
    )
    parser.add_argument(
        "--cfg-args",
        type=Path,
        default=None,
        help="Wu cfg_args (required for CUDA 3DGS rasterization)",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="PyBullet scene root (rgb/) for overlay fallback backgrounds",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output step5/ directory")
    parser.add_argument(
        "--wu-root",
        type=Path,
        default=Path(os.environ.get("WU_4DGS_ROOT", _REPO_ROOT / "third_party" / "4DGaussians")),
    )
    parser.add_argument(
        "--force-overlay",
        action="store_true",
        help="Skip Wu 3DGS rasterizer; draw projected ball on GT RGB (milestone3 fallback)",
    )
    parser.add_argument(
        "--sphere-radius-m",
        type=float,
        default=0.1,
        help="Ball radius for overlay fallback projection",
    )
    parser.add_argument(
        "--black-background",
        action="store_true",
        help="Use black clear color (default: white, matching object-only training)",
    )
    parser.add_argument(
        "--object-scale",
        type=float,
        default=None,
        help=(
            "Isotropic scale (phase1 align_scale) to bring the 4DGS-world object into the "
            "metric background frame. Omit when the object is already metric (e.g. big ball)."
        ),
    )
    parser.add_argument(
        "--object-crop-radius",
        type=float,
        default=None,
        help=(
            "Crop object Gaussians to this radius (in source 4DGS units) around the ball "
            "centroid before scaling, to drop the diffuse floater halo."
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated camera indices to render (default: all test cameras).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PNGs that already exist in the output renders/ folder (resume support).",
    )
    args = parser.parse_args()

    cam_list = None
    if args.cameras:
        cam_list = [int(x.strip()) for x in args.cameras.split(",") if x.strip()]

    if not args.force_overlay and args.cfg_args is None:
        print(
            "WARNING: --cfg-args not set; will use overlay fallback if Wu rasterizer unavailable.",
            file=sys.stderr,
        )

    result = run_step5(
        canonical_ply=args.canonical,
        bg_ply=args.bg_ply,
        predicted_csv=args.predicted,
        ref_traj_csv=args.ref_traj,
        dynerf_export=args.dynerf_export,
        out_dir=args.out,
        cfg_args=args.cfg_args,
        scene_dir=args.scene_dir,
        wu_root=args.wu_root,
        white_background=not args.black_background,
        force_overlay=args.force_overlay,
        sphere_radius_m=args.sphere_radius_m,
        object_scale=args.object_scale,
        object_crop_radius=args.object_crop_radius,
        cameras=cam_list,
        skip_existing=args.skip_existing,
    )

    print(f"Step 5 complete → {result.out_dir}")
    print(f"  mode:     {result.render_mode}")
    print(f"  rendered: {result.num_rendered} PNGs in {result.renders_dir}")
    print(f"  meta:     {result.meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
