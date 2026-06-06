#!/usr/bin/env python3
"""Export consolidated collision step6/refit metrics."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path("outputs/collision_pipeline")
OUT = ROOT / "summary"
RUN_RE = re.compile(
    r"scene_(?P<scene>\d+)_col_m(?P<mass>\d+)_(?P<mass_mode>asym|equal)(?P<split>\d+)_r(?P<rest>\d+)_event(?P<event>\d+)(?P<variant>.*)$"
)
EXCLUDED_RUN_SUFFIXES = ("_toend", "_fixed")


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_config_path(metrics: dict[str, object]) -> Path | None:
    objects = metrics.get("objects")
    if not isinstance(objects, list) or not objects:
        return None
    metrics_path = objects[0].get("metrics_json") if isinstance(objects[0], dict) else None
    if metrics_path is None:
        return None
    obj_metrics = load_json(Path(str(metrics_path)))
    sources = obj_metrics.get("sources") if isinstance(obj_metrics.get("sources"), dict) else {}
    gt_rgb_root = sources.get("gt_rgb_root")
    if gt_rgb_root is None:
        return None
    path = Path(str(gt_rgb_root)).parent / "config.json"
    return path if path.is_file() else None


def run_params(run: str) -> dict[str, object]:
    match = RUN_RE.match(run)
    if match is None:
        return {
            "scene_index": None,
            "mass_token": None,
            "mass_mode": None,
            "velocity_split_token": None,
            "restitution_token": None,
            "event_frame_token": None,
            "variant": "toend" if run.endswith("_toend") else "",
        }
    groups = match.groupdict()
    return {
        "scene_index": int(groups["scene"]),
        "mass_token": int(groups["mass"]),
        "mass_mode": groups["mass_mode"],
        "velocity_split_token": int(groups["split"]),
        "restitution_token": int(groups["rest"]),
        "event_frame_token": int(groups["event"]),
        "variant": groups["variant"].lstrip("_"),
    }


def config_fields(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}
    cfg = load_json(config_path)
    scene = cfg.get("scene") if isinstance(cfg.get("scene"), dict) else {}
    params = scene.get("scenario_params") if isinstance(scene.get("scenario_params"), dict) else {}
    sim = cfg.get("simulation") if isinstance(cfg.get("simulation"), dict) else {}
    mass_a = params.get("object_a_mass_kg")
    mass_b = params.get("object_b_mass_kg")
    return {
        "dataset_scene": config_path.parent.name,
        "gt_object_restitution": params.get("object_restitution"),
        "gt_mass_obj0_kg": mass_a,
        "gt_mass_obj1_kg": mass_b,
        "gt_mass_ratio_obj1_to_obj0": (float(mass_b) / float(mass_a)) if mass_a and mass_b else None,
        "gt_velocity_scale": params.get("velocity_scale"),
        "gt_velocity_split": params.get("velocity_split"),
        "gt_closing_speed_m_s": params.get("object_closing_speed_m_s"),
        "gt_obj0_initial_velocity_m_s": params.get("object_a_initial_velocity_m_s"),
        "gt_obj1_initial_velocity_m_s": params.get("object_b_initial_velocity_m_s"),
        "gt_num_frames": sim.get("num_frames"),
        "gt_dt_s": sim.get("dt_s"),
    }


def refit_fields(run_dir: Path) -> dict[str, object]:
    path = run_dir / "step4b_metric" / "refit_metric.json"
    if not path.is_file():
        return {}
    refit = load_json(path)
    bodies = refit.get("bodies") if isinstance(refit.get("bodies"), list) else []
    mass_ratio = refit.get("mass_ratio_to_obj0")
    return {
        "predicted_restitution_pair": refit.get("restitution_pair"),
        "predicted_restitution_wall": refit.get("restitution_wall"),
        "predicted_mass_ratio_obj1_to_obj0": mass_ratio[1] if isinstance(mass_ratio, list) and len(mass_ratio) > 1 else None,
        "predicted_restitution_floor_obj0": bodies[0].get("restitution_floor") if len(bodies) > 0 and isinstance(bodies[0], dict) else None,
        "predicted_restitution_floor_obj1": bodies[1].get("restitution_floor") if len(bodies) > 1 and isinstance(bodies[1], dict) else None,
        "train_fit_mse_m2": refit.get("train_fit_mse_m2"),
        "heldout_test_rmse_m": refit.get("heldout_test_rmse_m"),
        "heldout_test_rmse_obj0_m": (
            refit.get("heldout_test_rmse_per_object_m", [None, None])[0]
            if isinstance(refit.get("heldout_test_rmse_per_object_m"), list)
            else None
        ),
        "heldout_test_rmse_obj1_m": (
            refit.get("heldout_test_rmse_per_object_m", [None, None])[1]
            if isinstance(refit.get("heldout_test_rmse_per_object_m"), list)
            and len(refit.get("heldout_test_rmse_per_object_m", [])) > 1
            else None
        ),
        "gt_collisions_full": (refit.get("collisions") or {}).get("gt_full") if isinstance(refit.get("collisions"), dict) else None,
        "predicted_collisions_full": (refit.get("collisions") or {}).get("predicted_full") if isinstance(refit.get("collisions"), dict) else None,
    }


def object_metric_fields(metrics: dict[str, object], prefix: str, object_index: int) -> dict[str, object]:
    objects = metrics.get("objects") if isinstance(metrics.get("objects"), list) else []
    obj = next((o for o in objects if isinstance(o, dict) and o.get("object") == object_index), {})
    result = {
        f"{prefix}_pos_rmse_m": obj.get("pos_rmse_m"),
        f"{prefix}_vel_r2": obj.get("vel_r2"),
        f"{prefix}_acc_r2": obj.get("acc_r2"),
    }
    metrics_json = obj.get("metrics_json")
    if metrics_json:
        obj_metrics = load_json(Path(str(metrics_json)))
        split = obj_metrics.get("split") if isinstance(obj_metrics.get("split"), dict) else {}
        result.update(
            {
                f"{prefix}_num_pred_frames": obj_metrics.get("num_pred_frames"),
                f"{prefix}_frame_min": obj_metrics.get("frame_min"),
                f"{prefix}_frame_max": obj_metrics.get("frame_max"),
                f"{prefix}_train": split.get("train"),
                f"{prefix}_test": split.get("test"),
            }
        )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(ROOT.glob("*/step6/metrics.json")):
        run_dir = metrics_path.parents[1]
        if run_dir.name.endswith(EXCLUDED_RUN_SUFFIXES):
            continue
        metrics = load_json(metrics_path)
        render = metrics.get("render") if isinstance(metrics.get("render"), dict) else {}
        row = {
            "run": run_dir.name,
            **run_params(run_dir.name),
            **config_fields(dataset_config_path(metrics)),
            **refit_fields(run_dir),
            "n_objects": metrics.get("n_objects"),
            "mean_pos_rmse_m": metrics.get("mean_pos_rmse_m"),
            **object_metric_fields(metrics, "obj0", 0),
            **object_metric_fields(metrics, "obj1", 1),
            "psnr_db_mean": render.get("psnr_db_mean"),
            "mae_mean": render.get("mae_mean"),
            "ssim_mean": render.get("ssim_mean"),
            "num_render_matched": render.get("num_matched"),
        }
        rows.append(row)

    csv_path = OUT / "step6_metrics_summary.csv"
    compact_csv_path = OUT / "step6_metrics_summary_compact.csv"
    json_path = OUT / "step6_metrics_summary.json"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: scalar(v) for k, v in row.items()})
    compact_fieldnames = [
        "run",
        "dataset_scene",
        "variant",
        "gt_object_restitution",
        "predicted_restitution_pair",
        "predicted_restitution_wall",
        "gt_mass_ratio_obj1_to_obj0",
        "predicted_mass_ratio_obj1_to_obj0",
        "mean_pos_rmse_m",
        "obj0_pos_rmse_m",
        "obj0_vel_r2",
        "obj0_acc_r2",
        "obj1_pos_rmse_m",
        "obj1_vel_r2",
        "obj1_acc_r2",
        "psnr_db_mean",
        "mae_mean",
        "ssim_mean",
        "gt_collisions_full",
        "predicted_collisions_full",
        "obj0_test",
        "obj1_test",
    ]
    with compact_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compact_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: scalar(row.get(k)) for k in compact_fieldnames})
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} collision runs to {OUT}")
    print(f"Metrics: {csv_path}")
    print(f"Compact metrics: {compact_csv_path}")


if __name__ == "__main__":
    main()
