#!/usr/bin/env python3
"""Export axis-wise Gaussian-train and prediction-pipeline metric tables."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


BOUNCE_GAUSSIAN = Path("outputs/bounce_pipeline/wu75k_diagnostics/gaussian_vs_pybullet_train_r2.csv")
BOUNCE_PREDICTION = Path("outputs/bounce_pipeline/wu75k_velocity_refit_summary/best_velocity_metrics.csv")
BOUNCE_SUMMARY = Path("outputs/bounce_pipeline/wu75k_z_summary/step6_metrics_summary.csv")
COLLISION_GAUSSIAN = Path("outputs/collision_pipeline/summary/diagnostics/gaussian_vs_pybullet_train_r2.csv")
COLLISION_SUMMARY = Path("outputs/collision_pipeline/summary/step6_metrics_summary_compact.csv")
OUT_DIR = Path("outputs/diagnostics")

BOUNCE_RUN_DIR_OVERRIDES = {
    "wu75k_scene_0003_e0p93_am5p0_pred": "scene_0003_e0p93_am5p0",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_xyz(path: Path, cols: tuple[str, str, str] = ("x", "y", "z")) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    xyz: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frames.append(int(float(row["frame"])))
            xyz.append([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
    order = np.argsort(frames)
    return np.asarray(frames, dtype=np.int32)[order], np.asarray(xyz, dtype=np.float64)[order]


def matched_gt(gt_frames: np.ndarray, gt_xyz: np.ndarray, frames: np.ndarray) -> np.ndarray:
    by_frame = {int(frame): xyz for frame, xyz in zip(gt_frames.tolist(), gt_xyz)}
    return np.stack([by_frame[int(frame)] for frame in frames.tolist()], axis=0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def r2_pearson(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_flat = pred.reshape(-1).astype(np.float64)
    gt_flat = gt.reshape(-1).astype(np.float64)
    pred_flat = pred_flat - float(np.mean(pred_flat))
    gt_flat = gt_flat - float(np.mean(gt_flat))
    denom = float(np.linalg.norm(pred_flat) * np.linalg.norm(gt_flat))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.dot(pred_flat, gt_flat) / denom)
    return float(np.clip(corr * corr, 0.0, 1.0))


def safe_gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def scene_number(run: str) -> int:
    match = re.search(r"scene_(\d{4})", run)
    return int(match.group(1)) if match else 9999


def bounce_run_dir(run: str) -> Path:
    return Path("outputs/bounce_pipeline") / BOUNCE_RUN_DIR_OVERRIDES.get(run, run)


def bounce_metadata() -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in read_csv(BOUNCE_PREDICTION):
        meta[row["run"]] = {
            "restitution": row.get("restitution", ""),
            "angle_deg": row.get("angle_deg", ""),
            "prediction_csv": row["predicted_csv"],
        }
    return meta


def prediction_axis_metrics(pred_csv: Path, gt_csv: Path, axis_idx: int) -> tuple[int, int, int, float, float]:
    pred_frames, pred_xyz = read_xyz(pred_csv)
    gt_frames, gt_xyz = read_xyz(gt_csv, ("x_m", "y_m", "z_m"))
    gt = matched_gt(gt_frames, gt_xyz, pred_frames)
    y_true = gt[:, axis_idx]
    y_pred = pred_xyz[:, axis_idx]
    return (
        int(len(pred_frames)),
        int(np.min(pred_frames)),
        int(np.max(pred_frames)),
        rmse(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


def export_bounce() -> list[dict[str, object]]:
    meta = bounce_metadata()
    rows: list[dict[str, object]] = []
    for gaussian in sorted(read_csv(BOUNCE_GAUSSIAN), key=lambda row: scene_number(row["run"])):
        run = gaussian["run"]
        run_dir = bounce_run_dir(run)
        metrics = json.loads((run_dir / "step6" / "metrics.json").read_text(encoding="utf-8"))
        pred_csv = Path(meta[run]["prediction_csv"])
        gt_csv = Path(metrics["sources"]["gt_poses_csv"])
        n_pred, pred_start, pred_end, pred_rmse, pred_r2 = prediction_axis_metrics(pred_csv, gt_csv, 2)
        rows.append(
            {
                "domain": "ball_bounce",
                "run": run,
                "object": "",
                "axis": "z",
                "restitution": meta[run]["restitution"],
                "angle_deg": meta[run]["angle_deg"],
                "gaussian_train_n": gaussian["num_points"],
                "gaussian_train_frame_start": gaussian["frame_start"],
                "gaussian_train_frame_end": gaussian["frame_end"],
                "gaussian_train_rmse_m": float(gaussian["rmse_z_m"]),
                "gaussian_train_r2": float(gaussian["r2_z"]),
                "prediction_n": n_pred,
                "prediction_frame_start": pred_start,
                "prediction_frame_end": pred_end,
                "prediction_rmse_m": pred_rmse,
                "prediction_r2": pred_r2,
            }
        )
    return rows


def collision_prediction_sources(run: str, object_name: str) -> tuple[Path, Path]:
    obj_idx = 0 if object_name == "Object A" else 1
    metrics_path = Path("outputs/collision_pipeline") / run / "step6" / f"obj{obj_idx}" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Path(metrics["sources"]["predicted_csv"]), Path(metrics["sources"]["gt_poses_csv"])


def collision_step6_metrics(run: str, object_name: str) -> dict:
    obj_idx = 0 if object_name == "Object A" else 1
    metrics_path = Path("outputs/collision_pipeline") / run / "step6" / f"obj{obj_idx}" / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def collision_metadata() -> dict[str, dict[str, str]]:
    return {row["run"]: row for row in read_csv(COLLISION_SUMMARY)}


def apply_similarity(sim: dict[str, object], pts: np.ndarray) -> np.ndarray:
    scale = float(sim["scale"])
    rotation = np.asarray(sim["R"], dtype=np.float64)
    translation = np.asarray(sim["t"], dtype=np.float64)
    return (scale * (rotation @ pts.T)).T + translation[None, :]


def collision_gaussian_train_metrics(run: str, object_name: str, fps: float) -> tuple[int, int, int, float, float]:
    obj_idx = 0 if object_name == "Object A" else 1
    run_dir = Path("outputs/collision_pipeline") / run
    refit = json.loads((run_dir / "step4b_metric" / "refit_metric.json").read_text(encoding="utf-8"))
    test_start = int(refit["split"]["test"][0])
    gt_frames, gt_xyz = read_xyz(run_dir / "_poses" / f"object_poses_obj{obj_idx}.csv", ("x_m", "y_m", "z_m"))
    ext_frames, ext_xyz = read_xyz(run_dir / "step4a" / f"obj{obj_idx}" / "trajectory_smoothed.csv")
    ext_metric = apply_similarity(refit["similarity"][obj_idx], ext_xyz)
    keep = ext_frames < test_start
    frames = ext_frames[keep]
    ext = ext_metric[keep]
    gt = matched_gt(gt_frames, gt_xyz, frames)
    dt = 1.0 / max(float(fps), 1e-9)
    return (
        int(len(frames)),
        int(np.min(frames)),
        int(np.max(frames)),
        rmse(gt[:, 0], ext[:, 0]),
        r2_pearson(safe_gradient(ext, dt), safe_gradient(gt, dt)),
    )


def export_collision() -> list[dict[str, object]]:
    meta = collision_metadata()
    rows: list[dict[str, object]] = []
    for gaussian in sorted(read_csv(COLLISION_GAUSSIAN), key=lambda row: (scene_number(row["run"]), row["object"])):
        run = gaussian["run"]
        step6 = collision_step6_metrics(run, gaussian["object"])
        pred_csv, gt_csv = collision_prediction_sources(run, gaussian["object"])
        n_pred, pred_start, pred_end, pred_rmse, _pred_x_r2 = prediction_axis_metrics(pred_csv, gt_csv, 0)
        fit_n, fit_start, fit_end, fit_x_rmse, fit_vel_r2 = collision_gaussian_train_metrics(
            run,
            gaussian["object"],
            fps=float(step6["fps"]),
        )
        rows.append(
            {
                "domain": "collision",
                "run": run,
                "object": gaussian["object"],
                "axis": "x_rmse__velocity_r2",
                "restitution": meta[run].get("gt_object_restitution", ""),
                "angle_deg": "",
                "gaussian_train_n": fit_n,
                "gaussian_train_frame_start": fit_start,
                "gaussian_train_frame_end": fit_end,
                "gaussian_train_rmse_m": fit_x_rmse,
                "gaussian_train_r2": fit_vel_r2,
                "prediction_n": n_pred,
                "prediction_frame_start": pred_start,
                "prediction_frame_end": pred_end,
                "prediction_rmse_m": pred_rmse,
                "prediction_r2": float(step6["trajectory"]["vel_r2"]),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    bounce = export_bounce()
    collision = export_collision()
    write_rows(OUT_DIR / "ball_bounce_z_gaussian_train_vs_prediction.csv", bounce)
    write_rows(OUT_DIR / "collision_x_gaussian_train_vs_prediction.csv", collision)
    write_rows(OUT_DIR / "axis_gaussian_train_vs_prediction.csv", bounce + collision)
    print(f"Wrote {len(bounce)} bounce rows")
    print(f"Wrote {len(collision)} collision rows")
    print(f"Wrote combined rows: {OUT_DIR / 'axis_gaussian_train_vs_prediction.csv'}")


if __name__ == "__main__":
    main()
