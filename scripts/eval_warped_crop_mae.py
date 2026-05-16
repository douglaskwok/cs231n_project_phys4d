#!/usr/bin/env python
"""Render proxy: 2D projection error (px) of GT vs predicted sphere center on held-out frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phys4d.camera_project import load_cameras, project_pybullet_points  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402


def _centroid_px_error(
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    camera: dict,
) -> float | None:
    gt_uv, gt_ok = project_pybullet_points(gt_pos.reshape(1, 3), camera)
    pr_uv, pr_ok = project_pybullet_points(pred_pos.reshape(1, 3), camera)
    if not (gt_ok[0] and pr_ok[0]):
        return None
    du = float(gt_uv[0, 0] - pr_uv[0, 0])
    dv = float(gt_uv[0, 1] - pr_uv[0, 1])
    return float(np.hypot(du, dv))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=REPO_ROOT / "outputs/sphere_bounce_m2")
    parser.add_argument("--pred-poses", type=Path, required=True)
    parser.add_argument("--frames", type=str, default="60,70,80")
    parser.add_argument("--cameras", type=str, default="0,1,2,3")
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "outputs/param_id_pipeline/projection_error.json")
    args = parser.parse_args()

    scene = args.scene_dir.resolve()
    gt_traj = load_object_poses_csv(scene / "object_poses.csv")
    pred_traj = load_object_poses_csv(args.pred_poses)
    cams = load_cameras(scene / "cameras.json")
    cam_ids = [int(x) for x in args.cameras.split(",") if x.strip()]
    frames = [int(x) for x in args.frames.split(",") if x.strip()]

    rows = []
    for frame in frames:
        gt_p = gt_traj.by_frame(frame).position
        pr_p = pred_traj.by_frame(frame).position
        for ci in cam_ids:
            err = _centroid_px_error(gt_p, pr_p, cams[ci])
            if err is not None:
                rows.append({"frame": frame, "camera": ci, "projection_l2_px": err})

    mean_px = float(np.mean([r["projection_l2_px"] for r in rows])) if rows else float("nan")
    report = {
        "mean_projection_l2_px": mean_px,
        "rows": rows,
        "note": "2D reprojection of sphere center; full RGB PSNR needs graphdeco render on Modal.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
