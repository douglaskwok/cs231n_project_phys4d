#!/usr/bin/env python3
"""Fit train-side Gaussian trajectory diagnostics and compare to predictions.

This is a visual-diagnostic companion to the bounce/collision summary charts.
It fits a low-order polynomial in time to the extracted 4DGS train points only,
then evaluates that curve over the heldout/test prediction frames.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


FIT_DEGREE = 2
ROOT = Path(".")
BOUNCE_SUMMARY = Path("outputs/bounce_pipeline/wu75k_velocity_refit_summary/best_velocity_metrics.csv")
BOUNCE_OUT = Path("outputs/bounce_pipeline/wu75k_diagnostics")
COLLISION_SUMMARY = Path("outputs/collision_pipeline/summary/step6_metrics_summary_compact.csv")
COLLISION_OUT = Path("outputs/collision_pipeline/summary/diagnostics")


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


def fit_poly(t: np.ndarray, y: np.ndarray, degree: int = FIT_DEGREE) -> tuple[np.ndarray, float, float]:
    degree = min(int(degree), max(1, len(t) - 1))
    t0 = float(np.mean(t))
    ts = float(np.std(t))
    if ts <= 1e-12:
        ts = 1.0
    coeff = np.polyfit((t - t0) / ts, y, degree)
    return coeff, t0, ts


def eval_poly(coeff: np.ndarray, t0: float, ts: float, t: np.ndarray) -> np.ndarray:
    return np.polyval(coeff, (t - t0) / ts)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def candidate_run_from_refit(path: Path) -> str:
    parts = path.parts
    if "candidates" in parts:
        idx = parts.index("candidates")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def export_csv_json(rows: list[dict[str, object]], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: scalar(v) for k, v in row.items()})
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return csv_path


def bounce_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not BOUNCE_SUMMARY.is_file():
        return rows
    for row in csv.DictReader(BOUNCE_SUMMARY.open(newline="", encoding="utf-8")):
        pred_csv = Path(row["predicted_csv"])
        refit_json = Path(row["refit_json"])
        refit = json.loads(refit_json.read_text(encoding="utf-8"))
        split = refit["split"]
        test_start, test_end = [int(v) for v in split["test"]]
        fps = 120.0
        candidate_run = candidate_run_from_refit(refit_json)
        extracted_csv = Path("outputs/bounce_pipeline") / candidate_run / "step4a" / "trajectory_smoothed.csv"
        if not extracted_csv.is_file():
            continue

        ext_frames, ext_xyz = read_xyz(extracted_csv)
        ext_metric = apply_similarity(refit["similarity"], ext_xyz)
        heldout_len = test_end - test_start + 1
        train_start = max(0, test_start - heldout_len)
        train_mask = (ext_frames >= train_start) & (ext_frames < test_start)
        train_t = ext_frames[train_mask] / fps
        train_y = ext_metric[train_mask, 2]

        pred_frames, pred_xyz = read_xyz(pred_csv)
        pred_mask = (pred_frames >= test_start) & (pred_frames <= test_end)
        pred_t = pred_frames[pred_mask] / fps
        pred_y = pred_xyz[pred_mask, 2]

        coeff, t0, ts = fit_poly(train_t, train_y)
        train_fit_y = eval_poly(coeff, t0, ts, train_t)
        diag_y = eval_poly(coeff, t0, ts, pred_t)
        rows.append(
            {
                "domain": "bounce",
                "run": row["run"],
                "axis": "z",
                "object": "",
                "fit_model": f"poly{len(coeff)-1}",
                "train_points": int(len(train_y)),
                "test_points": int(len(pred_y)),
                "train_start_frame": train_start,
                "train_end_frame": test_start - 1,
                "test_start_frame": test_start,
                "test_end_frame": test_end,
                "train_gaussian_fit_r2": r2_score(train_y, train_fit_y),
                "train_gaussian_fit_rmse_m": rmse(train_y, train_fit_y),
                "diagnostic_vs_prediction_r2": r2_score(pred_y, diag_y),
                "diagnostic_vs_prediction_rmse_m": rmse(pred_y, diag_y),
                "predicted_csv": str(pred_csv),
                "extracted_csv": str(extracted_csv),
                "refit_json": str(refit_json),
            }
        )
    return rows


def collision_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not COLLISION_SUMMARY.is_file():
        return rows
    for summary_row in csv.DictReader(COLLISION_SUMMARY.open(newline="", encoding="utf-8")):
        run = summary_row["run"]
        run_dir = Path("outputs/collision_pipeline") / run
        refit_json = run_dir / "step4b_metric" / "refit_metric.json"
        if not refit_json.is_file():
            continue
        refit = json.loads(refit_json.read_text(encoding="utf-8"))
        test_start, test_end = [int(v) for v in refit["split"]["test"]]
        heldout_len = test_end - test_start + 1
        train_start = max(0, test_start - heldout_len)
        fps = 60.0
        for obj_idx, obj_name in [(0, "Object A"), (1, "Object B")]:
            extracted_csv = run_dir / "step4a" / f"obj{obj_idx}" / "trajectory_smoothed.csv"
            pred_csv = run_dir / "step4c" / f"trajectory_predicted_obj{obj_idx}.csv"
            if not extracted_csv.is_file() or not pred_csv.is_file():
                continue
            ext_frames, ext_xyz = read_xyz(extracted_csv)
            ext_metric = apply_similarity(refit["similarity"][obj_idx], ext_xyz)
            train_mask = (ext_frames >= train_start) & (ext_frames < test_start)
            train_t = ext_frames[train_mask] / fps
            train_y = ext_metric[train_mask, 0]

            pred_frames, pred_xyz = read_xyz(pred_csv)
            pred_mask = (pred_frames >= test_start) & (pred_frames <= test_end)
            pred_t = pred_frames[pred_mask] / fps
            pred_y = pred_xyz[pred_mask, 0]

            coeff, t0, ts = fit_poly(train_t, train_y)
            train_fit_y = eval_poly(coeff, t0, ts, train_t)
            diag_y = eval_poly(coeff, t0, ts, pred_t)
            rows.append(
                {
                    "domain": "collision",
                    "run": run,
                    "axis": "x",
                    "object": obj_name,
                    "fit_model": f"poly{len(coeff)-1}",
                    "train_points": int(len(train_y)),
                    "test_points": int(len(pred_y)),
                    "train_start_frame": train_start,
                    "train_end_frame": test_start - 1,
                    "test_start_frame": test_start,
                    "test_end_frame": test_end,
                    "train_gaussian_fit_r2": r2_score(train_y, train_fit_y),
                    "train_gaussian_fit_rmse_m": rmse(train_y, train_fit_y),
                    "diagnostic_vs_prediction_r2": r2_score(pred_y, diag_y),
                    "diagnostic_vs_prediction_rmse_m": rmse(pred_y, diag_y),
                    "predicted_csv": str(pred_csv),
                    "extracted_csv": str(extracted_csv),
                    "refit_json": str(refit_json),
                }
            )
    return rows


def main() -> None:
    bounce = bounce_rows()
    collision = collision_rows()
    if bounce:
        path = export_csv_json(bounce, BOUNCE_OUT, "train_gaussian_curve_vs_prediction")
        print(f"Wrote bounce diagnostics: {path}")
    if collision:
        path = export_csv_json(collision, COLLISION_OUT, "train_gaussian_curve_vs_prediction")
        print(f"Wrote collision diagnostics: {path}")
    combined = bounce + collision
    if combined:
        path = export_csv_json(combined, Path("outputs/diagnostics"), "train_gaussian_curve_vs_prediction_all")
        print(f"Wrote combined diagnostics: {path}")


if __name__ == "__main__":
    main()
