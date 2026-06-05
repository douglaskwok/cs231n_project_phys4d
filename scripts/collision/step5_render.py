#!/usr/bin/env python
"""Step 5 (collision) — composite N translated objects with background, render test views.

Multi-object analogue of ``scripts/bounce/step5_render.py``. Per-object flags take
comma-separated lists (one value per object); a single value is broadcast to all.

Example (two objects)::

  python scripts/collision/step5_render.py \\
    --canonical objA.ply,objB.ply \\
    --bg-ply background.ply \\
    --predicted step4c/trajectory_predicted_obj0.csv,step4c/trajectory_predicted_obj1.csv \\
    --ref-traj step4a/obj0/trajectory_smoothed.csv,step4a/obj1/trajectory_smoothed.csv \\
    --dynerf-export runs/objA \\
    --cfg-args objA/cfg_args \\
    --object-scale 0.05,0.05 --object-crop-radius 0.33,0.33 \\
    --out outputs/collision_pipeline/<RUN>/step5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.collision.render_compose import run_step5  # noqa: E402


def _split_paths(value: str) -> list[Path]:
    return [Path(v.strip()) for v in value.split(",") if v.strip()]


def _split_floats(value: str | None, n: int) -> list[float | None]:
    if value is None:
        return [None] * n
    parts = [v.strip() for v in value.split(",")]
    vals = [float(p) if p not in ("", "none", "None") else None for p in parts]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise SystemExit(f"expected 1 or {n} values, got {len(vals)}: {value}")
    return vals


def _split_vec3s(value: str | None, n: int) -> list[np.ndarray | None]:
    if value is None:
        return [None] * n
    chunks = [c.strip() for c in value.split(";") if c.strip()]
    if len(chunks) == 1:
        parts = [float(p) for p in chunks[0].split(",") if p.strip()]
        if len(parts) != 3:
            raise SystemExit(f"expected 3 comma-separated values for box half-extents, got {parts}")
        vec = np.array(parts, dtype=np.float64)
        return [vec.copy()] * n
    if len(chunks) != n:
        raise SystemExit(f"expected 1 or {n} box half-extent triples, got {len(chunks)}: {value}")
    out: list[np.ndarray | None] = []
    for chunk in chunks:
        parts = [float(p) for p in chunk.split(",") if p.strip()]
        if len(parts) != 3:
            raise SystemExit(f"expected 3 values per box half-extents triple, got {parts}")
        out.append(np.array(parts, dtype=np.float64))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, help="Comma-separated object point_cloud.ply per object")
    parser.add_argument("--bg-ply", type=Path, required=True, help="Static background 3DGS PLY")
    parser.add_argument("--predicted", required=True, help="Comma-separated per-object predicted CSV")
    parser.add_argument("--ref-traj", required=True, help="Comma-separated per-object step4a trajectory CSV")
    parser.add_argument("--dynerf-export", type=Path, required=True, help="Folder with transforms_test.json")
    parser.add_argument("--cfg-args", type=Path, default=None, help="Wu cfg_args for CUDA rasterization")
    parser.add_argument("--scene-dir", type=Path, default=None, help="PyBullet scene root (overlay fallback)")
    parser.add_argument("--out", type=Path, required=True, help="Output step5/ directory")
    parser.add_argument(
        "--wu-root",
        type=Path,
        default=Path(os.environ.get("WU_4DGS_ROOT", _REPO_ROOT / "third_party" / "4DGaussians")),
    )
    parser.add_argument("--force-overlay", action="store_true")
    parser.add_argument("--sphere-radius-m", type=float, default=0.1)
    parser.add_argument("--black-background", action="store_true")
    parser.add_argument("--object-scale", default=None, help="Comma-separated isotropic scale per object")
    parser.add_argument("--object-crop-radius", default=None, help="Comma-separated spherical crop radius per object")
    parser.add_argument(
        "--object-crop-box-half-extents",
        default="0.21,0.21,0.13125",
        help="Metric half-extents hx,hy,hz for axis-aligned box crop (default 0.42x0.42x0.2625 m box).",
    )
    parser.add_argument(
        "--object-opacity-boost",
        type=float,
        default=4.0,
        help="Multiply object splat alpha by this factor (default 4.0).",
    )
    parser.add_argument(
        "--object-max-scale",
        default="0.04",
        help="Comma-separated max activated splat scale per object (metric units, after --object-scale).",
    )
    parser.add_argument(
        "--object-min-opacity",
        default="0.05",
        help="Drop splats below this activated opacity before render (default 0.05).",
    )
    parser.add_argument(
        "--composite-mode",
        choices=("alpha", "depth", "threshold", "merge3d"),
        default="alpha",
        help="Layered composite: alpha (default), depth, threshold, or merge3d single-pass.",
    )
    parser.add_argument(
        "--composite-2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use layered compositing (default on). --no-composite-2d forces merge3d.",
    )
    parser.add_argument(
        "--composite-alpha-gamma",
        type=float,
        default=1.0,
        help="Alpha matte edge sharpness in alpha composite mode (default 1.0).",
    )
    parser.add_argument(
        "--composite-threshold",
        type=int,
        default=32,
        help="Foreground mask threshold for layered composite (default 32).",
    )
    parser.add_argument("--cameras", type=str, default=None, help="Comma-separated camera indices")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    canonical = _split_paths(args.canonical)
    predicted = _split_paths(args.predicted)
    ref_traj = _split_paths(args.ref_traj)
    cfg_args_list = _split_paths(str(args.cfg_args)) if args.cfg_args else None
    n = len(canonical)
    if not (len(predicted) == len(ref_traj) == n):
        print("ERROR: --canonical/--predicted/--ref-traj must have equal counts", file=sys.stderr)
        return 2

    object_scales = _split_floats(args.object_scale, n)
    object_crop_radii = _split_floats(args.object_crop_radius, n)
    object_crop_boxes = _split_vec3s(args.object_crop_box_half_extents, n)
    object_max_scales = _split_floats(args.object_max_scale, n)
    object_min_opacities = _split_floats(args.object_min_opacity, n)

    cam_list = None
    if args.cameras:
        cam_list = [int(x.strip()) for x in args.cameras.split(",") if x.strip()]

    if cfg_args_list and len(cfg_args_list) not in (1, n):
        print(f"ERROR: --cfg-args must have 1 or {n} paths", file=sys.stderr)
        return 2

    if not args.force_overlay and cfg_args_list is None:
        print("WARNING: --cfg-args not set; overlay fallback will be used.", file=sys.stderr)

    result = run_step5(
        canonical_plys=canonical,
        bg_ply=args.bg_ply,
        predicted_csvs=predicted,
        ref_traj_csvs=ref_traj,
        dynerf_export=args.dynerf_export,
        out_dir=args.out,
        cfg_args=cfg_args_list[0] if cfg_args_list else None,
        cfg_args_list=cfg_args_list,
        scene_dir=args.scene_dir,
        wu_root=args.wu_root,
        white_background=not args.black_background,
        force_overlay=args.force_overlay,
        sphere_radius_m=args.sphere_radius_m,
        object_scales=object_scales,
        object_crop_radii=object_crop_radii,
        object_crop_box_half_extents_m=object_crop_boxes,
        object_max_scales=object_max_scales,
        object_min_opacities=object_min_opacities,
        object_opacity_boost=args.object_opacity_boost,
        composite_2d=args.composite_2d,
        composite_mode=args.composite_mode,
        composite_threshold=args.composite_threshold,
        composite_alpha_gamma=args.composite_alpha_gamma,
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
