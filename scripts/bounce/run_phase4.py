#!/usr/bin/env python
"""Phase 4 — compose held-out renders from predicted trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.render_compose import run_phase4  # noqa: E402


def _load_sphere_radius_m(scene_dir: Path) -> float:
    """Read sphere radius from scene config for projected marker sizing."""

    config_path = scene_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing scene config: {config_path}")
    blob = json.loads(config_path.read_text(encoding="utf-8"))
    objects = blob.get("scene", {}).get("objects", [])
    # Prefer an explicitly named sphere if present.
    for obj in objects:
        if obj.get("name") == "sphere" and "radius_m" in obj:
            return float(obj["radius_m"])
    # Fallback: first spherical object in the scene.
    for obj in objects:
        if obj.get("shape") == "sphere" and "radius_m" in obj:
            return float(obj["radius_m"])
    raise ValueError(f"Could not find sphere radius_m in {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predicted",
        type=Path,
        required=True,
        help="Phase-3 trajectory_predicted.csv path.",
    )
    parser.add_argument(
        "--dynerf-export",
        type=Path,
        required=True,
        help="Object-only DyNeRF export folder with transforms_test.json.",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Scene root with rgb/ and config.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output phase4 directory.",
    )
    parser.add_argument(
        "--sphere-radius-m",
        type=float,
        default=None,
        help="Override sphere radius in meters (default: read from scene config).",
    )
    parser.add_argument(
        "--no-mask-gt-ball",
        action="store_true",
        help="Disable masking out the GT ball from background RGB.",
    )
    parser.add_argument(
        "--render-style",
        choices=("gaussian_splat", "disk"),
        default="gaussian_splat",
        help="Rendered prediction appearance style.",
    )
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    sphere_radius_m = (
        float(args.sphere_radius_m)
        if args.sphere_radius_m is not None
        else _load_sphere_radius_m(scene_dir)
    )

    result = run_phase4(
        predicted_csv=args.predicted,
        dynerf_export=args.dynerf_export,
        scene_dir=scene_dir,
        out_dir=args.out,
        sphere_radius_m=sphere_radius_m,
        mask_gt_ball=not args.no_mask_gt_ball,
        render_style=args.render_style,
    )
    print(f"Phase 4 complete -> {result.out_dir}")
    print(f"  renders: {result.renders_dir}")
    print(f"  num_rendered: {result.num_rendered}")
    print(f"  missing_predictions: {result.num_missing_predictions}")
    print(f"  invalid_projections: {result.num_invalid_projections}")
    print(f"  projection_mode: {result.calibrated_projection_mode}")
    if result.sample_png is not None:
        print(f"  sample_png: {result.sample_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
