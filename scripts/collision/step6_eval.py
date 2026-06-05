#!/usr/bin/env python
"""Step 6 (collision) — per-object trajectory metrics + shared render metrics.

Multi-object analogue of ``scripts/bounce/step6_eval.py``. Trajectory metrics
(RMSE / velocity / acceleration R²) run per object; render metrics (PSNR / SSIM /
MAE) operate on the full composited frame and are computed once. Per-object
results land under ``<out>/obj{k}/`` and a combined summary in ``<out>/metrics.json``.

Example::

  python scripts/collision/step6_eval.py \\
    --predicted step4c/trajectory_predicted_obj0.csv,step4c/trajectory_predicted_obj1.csv \\
    --gt-poses scene/object_poses_obj0.csv,scene/object_poses_obj1.csv \\
    --rendered step5/renders --gt-rgb-root scene/rgb \\
    --out outputs/collision_pipeline/<RUN>/step6 --fps 120 \\
    --extracted-traj step4a/obj0/trajectory_smoothed.csv,step4a/obj1/trajectory_smoothed.csv \\
    --refit-metric step4b_metric/refit_metric.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.metrics import run_step6  # noqa: E402


def _split(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, help="Comma-separated per-object predicted CSV")
    parser.add_argument("--gt-poses", required=True, help="Comma-separated per-object object_poses.csv")
    parser.add_argument("--rendered", type=Path, required=True, help="step5/renders directory")
    parser.add_argument("--gt-rgb-root", type=Path, required=True, help="Scene rgb/ root")
    parser.add_argument("--out", type=Path, required=True, help="Output step6/ directory")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--trajectory-only", action="store_true", help="Skip PSNR/SSIM render metrics")
    parser.add_argument("--extracted-traj", default=None, help="Comma-separated per-object step4a CSV")
    parser.add_argument("--refit-metric", type=Path, default=None, help="step4b_metric/refit_metric.json")
    args = parser.parse_args()

    predicted = [Path(p) for p in _split(args.predicted)]
    gt_poses = [Path(p) for p in _split(args.gt_poses)]
    n_obj = len(predicted)
    if len(gt_poses) != n_obj:
        raise SystemExit("--predicted and --gt-poses must list the same number of objects")
    extracted = [Path(p) for p in _split(args.extracted_traj)]

    # The collision refit stores a per-object similarity list; the bounce fused plot
    # expects a single similarity dict, so per-object overlays are computed by passing
    # the matching extracted traj only when a per-object refit is available. We pass the
    # split for the test-marker line.
    split_blob = None
    if args.refit_metric is not None and args.refit_metric.is_file():
        split_blob = json.loads(args.refit_metric.read_text(encoding="utf-8")).get("split")

    out_base = args.out.resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"n_objects": n_obj, "objects": []}
    render_block: dict | None = None

    for k in range(n_obj):
        # Render metrics computed once (object 0); other objects trajectory-only.
        skip_render = bool(args.trajectory_only) or (k > 0)
        result = run_step6(
            predicted_csv=predicted[k],
            gt_poses_csv=gt_poses[k],
            rendered_dir=args.rendered,
            gt_rgb_root=args.gt_rgb_root,
            out_dir=out_base / f"obj{k}",
            fps=args.fps,
            skip_render_metrics=skip_render,
            extracted_traj_csv=extracted[k] if k < len(extracted) else None,
            refit_metric_json=None,  # per-object similarity not directly compatible
            split=split_blob,
        )
        blob = json.loads(result.metrics_json.read_text(encoding="utf-8"))
        if k == 0 and not args.trajectory_only:
            render_block = blob.get("render")
        summary["objects"].append(
            {
                "object": k,
                "pos_rmse_m": result.pos_rmse_m,
                "vel_r2": result.vel_r2,
                "acc_r2": result.acc_r2,
                "metrics_json": str(result.metrics_json),
            }
        )
        print(f"obj{k}: pos_rmse={result.pos_rmse_m*100:.2f} cm  vel_r2={result.vel_r2:.3f}")

    rmses = [o["pos_rmse_m"] for o in summary["objects"]]
    summary["mean_pos_rmse_m"] = float(sum(rmses) / len(rmses))
    summary["render"] = render_block
    summary_json = out_base / "metrics.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nStep 6 complete → {out_base}")
    print(f"  mean pos RMSE: {summary['mean_pos_rmse_m']*100:.2f} cm")
    if render_block:
        print(f"  render PSNR: {render_block.get('psnr_db_mean')}")
        print(f"  render SSIM: {render_block.get('ssim_mean')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
