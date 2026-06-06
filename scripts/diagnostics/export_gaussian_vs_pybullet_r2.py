#!/usr/bin/env python3
"""Compute direct R2 between extracted 4DGS points and PyBullet GT."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


BOUNCE_SUMMARY = Path("outputs/bounce_pipeline/wu75k_z_summary/step6_metrics_summary.csv")
BOUNCE_OUT = Path("outputs/bounce_pipeline/wu75k_diagnostics")
COLLISION_SUMMARY = Path("outputs/collision_pipeline/summary/step6_metrics_summary_compact.csv")
COLLISION_OUT = Path("outputs/collision_pipeline/summary/diagnostics")

BOUNCE_RUN_DIR_OVERRIDES = {
    "wu75k_scene_0003_e0p93_am5p0_pred": "scene_0003_e0p93_am5p0",
}


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def read_xyz(path: Path, cols: tuple[str, str, str] = ("x", "y", "z")) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    xyz: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(float(row["frame"])))
            xyz.append([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
    order = np.argsort(frames)
    return np.asarray(frames, dtype=np.int32)[order], np.asarray(xyz, dtype=np.float64)[order]


def apply_similarity(sim: dict[str, object], pts: np.ndarray) -> np.ndarray:
    scale = float(sim["scale"])
    rotation = np.asarray(sim["R"], dtype=np.float64)
    translation = np.asarray(sim["t"], dtype=np.float64)
    return (scale * (rotation @ pts.T)).T + translation[None, :]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def matched_gt(gt_frames: np.ndarray, gt_xyz: np.ndarray, frames: np.ndarray) -> np.ndarray:
    by_frame = {int(frame): xyz for frame, xyz in zip(gt_frames.tolist(), gt_xyz)}
    return np.stack([by_frame[int(frame)] for frame in frames.tolist()], axis=0)


def export(rows: list[dict[str, object]], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar(value) for key, value in row.items()})
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return csv_path


def bounce_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in csv.DictReader(BOUNCE_SUMMARY.open(newline="", encoding="utf-8")):
        display_run = summary["run"]
        run_dir = Path("outputs/bounce_pipeline") / BOUNCE_RUN_DIR_OVERRIDES.get(display_run, display_run)
        metrics = json.loads((run_dir / "step6" / "metrics.json").read_text(encoding="utf-8"))
        refit = json.loads((run_dir / "step4b_metric" / "refit_metric.json").read_text(encoding="utf-8"))
        test_start = int(metrics["split"]["test"][0])
        gt_frames, gt_xyz = read_xyz(Path(metrics["sources"]["gt_poses_csv"]), ("x_m", "y_m", "z_m"))
        ext_frames, ext_xyz = read_xyz(run_dir / "step4a" / "trajectory_smoothed.csv")
        ext_metric = apply_similarity(refit["similarity"], ext_xyz)
        keep = ext_frames < test_start
        frames = ext_frames[keep]
        ext = ext_metric[keep]
        gt = matched_gt(gt_frames, gt_xyz, frames)
        rows.append(
            {
                "domain": "bounce",
                "run": display_run,
                "object": "",
                "scope": "train_before_split",
                "num_points": int(len(frames)),
                "frame_start": int(np.min(frames)),
                "frame_end": int(np.max(frames)),
                "r2_xyz": r2_score(gt.reshape(-1), ext.reshape(-1)),
                "rmse_xyz_m": rmse(gt.reshape(-1), ext.reshape(-1)),
                "r2_x": r2_score(gt[:, 0], ext[:, 0]),
                "rmse_x_m": rmse(gt[:, 0], ext[:, 0]),
                "r2_y": r2_score(gt[:, 1], ext[:, 1]),
                "rmse_y_m": rmse(gt[:, 1], ext[:, 1]),
                "r2_z": r2_score(gt[:, 2], ext[:, 2]),
                "rmse_z_m": rmse(gt[:, 2], ext[:, 2]),
                "extracted_csv": str(run_dir / "step4a" / "trajectory_smoothed.csv"),
                "gt_csv": str(metrics["sources"]["gt_poses_csv"]),
            }
        )
    return rows


def collision_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in csv.DictReader(COLLISION_SUMMARY.open(newline="", encoding="utf-8")):
        run = summary["run"]
        run_dir = Path("outputs/collision_pipeline") / run
        refit = json.loads((run_dir / "step4b_metric" / "refit_metric.json").read_text(encoding="utf-8"))
        test_start = int(refit["split"]["test"][0])
        for obj_idx, obj_name in [(0, "Object A"), (1, "Object B")]:
            gt_csv = run_dir / "_poses" / f"object_poses_obj{obj_idx}.csv"
            extracted_csv = run_dir / "step4a" / f"obj{obj_idx}" / "trajectory_smoothed.csv"
            gt_frames, gt_xyz = read_xyz(gt_csv, ("x_m", "y_m", "z_m"))
            ext_frames, ext_xyz = read_xyz(extracted_csv)
            ext_metric = apply_similarity(refit["similarity"][obj_idx], ext_xyz)
            keep = ext_frames < test_start
            frames = ext_frames[keep]
            ext = ext_metric[keep]
            gt = matched_gt(gt_frames, gt_xyz, frames)
            rows.append(
                {
                    "domain": "collision",
                    "run": run,
                    "object": obj_name,
                    "scope": "train_before_split",
                    "num_points": int(len(frames)),
                    "frame_start": int(np.min(frames)),
                    "frame_end": int(np.max(frames)),
                    "r2_xyz": r2_score(gt.reshape(-1), ext.reshape(-1)),
                    "rmse_xyz_m": rmse(gt.reshape(-1), ext.reshape(-1)),
                    "r2_x": r2_score(gt[:, 0], ext[:, 0]),
                    "rmse_x_m": rmse(gt[:, 0], ext[:, 0]),
                    "r2_y": r2_score(gt[:, 1], ext[:, 1]),
                    "rmse_y_m": rmse(gt[:, 1], ext[:, 1]),
                    "r2_z": r2_score(gt[:, 2], ext[:, 2]),
                    "rmse_z_m": rmse(gt[:, 2], ext[:, 2]),
                    "extracted_csv": str(extracted_csv),
                    "gt_csv": str(gt_csv),
                }
            )
    return rows


def main() -> None:
    bounce = bounce_rows()
    collision = collision_rows()
    print(f"Wrote bounce: {export(bounce, BOUNCE_OUT, 'gaussian_vs_pybullet_train_r2')}")
    print(f"Wrote collision: {export(collision, COLLISION_OUT, 'gaussian_vs_pybullet_train_r2')}")
    print(f"Wrote combined: {export(bounce + collision, Path('outputs/diagnostics'), 'gaussian_vs_pybullet_train_r2_all')}")


if __name__ == "__main__":
    main()
