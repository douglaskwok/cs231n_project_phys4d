"""Step 4c — roll fitted bounce physics forward onto held-out test frames."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phys4d.poses import load_object_poses_csv

from .physics import PhysicsParams, load_physics_params_json, load_trajectory_csv, simulate_trajectory


@dataclass
class Step4cResult:
    """Artifacts written under ``step4c/``."""

    out_dir: Path
    predicted_csv: Path
    plot_png: Path
    meta_json: Path
    num_test_frames: int
    boundary_gap_m: float
    min_clearance_m: float


def _read_frame_map(frame_map_path: Path) -> tuple[list[dict], list[dict]]:
    """Load train/test rows from ``frame_map.json``."""

    frame_map_path = frame_map_path.resolve()
    if not frame_map_path.is_file():
        raise FileNotFoundError(f"Missing frame_map.json: {frame_map_path}")
    blob = json.loads(frame_map_path.read_text(encoding="utf-8"))
    train = list(blob.get("train") or [])
    test = list(blob.get("test") or [])
    if not train:
        raise ValueError(f"No train entries in {frame_map_path}")
    if not test:
        raise ValueError(
            f"No test entries in {frame_map_path}. Export may be all-train; "
            "retrain with a held-out split before extrapolation."
        )
    return train, test


def _fps_from_sources(
    frame_map_path: Path,
    scene_config: Path | None,
    default_fps: float,
) -> float:
    """Resolve FPS from scene config, frame_map, or default."""

    if scene_config is not None and scene_config.is_file():
        blob = json.loads(scene_config.read_text(encoding="utf-8"))
        fps = float(blob.get("fps", 0.0))
        if fps > 0.0:
            return fps
    blob = json.loads(frame_map_path.read_text(encoding="utf-8"))
    fps = float(blob.get("fps", 0.0))
    if fps > 0.0:
        return fps
    return float(default_fps)


def _validate_scene_config(
    scene_config: Path | None,
    *,
    num_train: int,
    num_test: int,
) -> None:
    """Optional consistency check against PyBullet scene ``config.json``."""

    if scene_config is None or not scene_config.is_file():
        return
    blob = json.loads(scene_config.read_text(encoding="utf-8"))
    expected_train = blob.get("train_frames")
    expected_test = blob.get("test_frames")
    if expected_train is not None and int(expected_train) != num_train:
        raise ValueError(
            f"frame_map train count {num_train} != scene config train_frames {expected_train}"
        )
    if expected_test is not None and int(expected_test) != num_test:
        raise ValueError(
            f"frame_map test count {num_test} != scene config test_frames {expected_test}"
        )


def _build_rollout_timeline(
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    frames_traj: np.ndarray | None,
    times_traj: np.ndarray | None,
    obs_traj: np.ndarray | None,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge train observations (from Step 4a CSV) with held-out test times.

    Test rows carry NaN observations; the physics integrator continues from the
    simulator state reached at the end of the train window (single continuous rollout).
    """

    by_frame: dict[int, tuple[float, np.ndarray]] = {}
    if frames_traj is not None and times_traj is not None and obs_traj is not None:
        for i, frame in enumerate(frames_traj.tolist()):
            by_frame[int(frame)] = (float(times_traj[i]), obs_traj[i].copy())

    train_set = {int(r["original_frame"]) for r in train_rows}
    test_set = {int(r["original_frame"]) for r in test_rows}
    all_rows = sorted(
        [*train_rows, *test_rows],
        key=lambda r: int(r["original_frame"]),
    )

    frames_out: list[int] = []
    times_out: list[float] = []
    obs_out: list[np.ndarray] = []

    for row in all_rows:
        frame = int(row["original_frame"])
        t_sec = float(row.get("original_time_s", frame / fps))
        if frame in by_frame:
            t_sec, xyz = by_frame[frame]
        elif frame in test_set:
            xyz = np.full(3, np.nan, dtype=np.float64)
        else:
            xyz = np.full(3, np.nan, dtype=np.float64)
        frames_out.append(frame)
        times_out.append(t_sec)
        obs_out.append(xyz)

    frames_arr = np.asarray(frames_out, dtype=np.int32)
    times_arr = np.asarray(times_out, dtype=np.float64)
    obs_arr = np.stack(obs_out, axis=0)

    train_mask = np.asarray([f in train_set for f in frames_arr.tolist()], dtype=bool)
    test_mask = np.asarray([f in test_set for f in frames_arr.tolist()], dtype=bool)

    if not np.any(test_mask):
        raise ValueError("Rollout timeline has no test frames.")
    finite_train = train_mask & np.isfinite(obs_arr).all(axis=1)
    if frames_traj is not None and not np.any(finite_train):
        raise ValueError(
            "No overlap between Step 4a trajectory frames and frame_map train entries."
        )
    if np.any(np.diff(times_arr) <= 0.0):
        raise ValueError("Merged timeline t_sec must be strictly increasing.")

    return frames_arr, times_arr, obs_arr, train_mask, test_mask


def _write_predicted_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "t_sec", "x", "y", "z"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def _plot_full(
    path: Path,
    times: np.ndarray,
    obs: np.ndarray,
    pred: np.ndarray,
    train_mask: np.ndarray,
    *,
    gt: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, axis, name in zip(axes, range(3), ("x", "y", "z")):
        finite_train = train_mask & np.isfinite(obs).all(axis=1)
        if np.any(finite_train):
            ax.plot(
                times[finite_train],
                obs[finite_train, axis],
                "o",
                ms=2,
                alpha=0.5,
                label="train observed",
            )
        ax.plot(times, pred[:, axis], "-", lw=2.0, label="simulated")
        if gt is not None:
            ax.plot(times, gt[:, axis], "--", lw=1.2, alpha=0.75, label="GT")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    test_idx = np.where(~train_mask)[0]
    if test_idx.size > 0:
        split_t = float(times[test_idx[0]])
        for ax in axes:
            ax.axvline(split_t, color="k", linestyle="--", linewidth=1.0, alpha=0.8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Step 4c — train observed + held-out prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_step4c(
    *,
    params_json: Path,
    frame_map_path: Path,
    out_dir: Path,
    trajectory_csv: Path | None = None,
    scene_config: Path | None = None,
    gt_poses: Path | None = None,
    fps: float = 60.0,
    ground_epsilon_m: float = 1e-4,
    strict: bool = False,
) -> Step4cResult:
    """Extrapolate test frames via continuous physics rollout after the train window."""

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params: PhysicsParams = load_physics_params_json(params_json)
    train_rows, test_rows = _read_frame_map(frame_map_path)
    _validate_scene_config(scene_config, num_train=len(train_rows), num_test=len(test_rows))

    export_fps = _fps_from_sources(frame_map_path, scene_config, default_fps=fps)

    frames_traj = times_traj = obs_traj = None
    if trajectory_csv is not None:
        frames_traj, times_traj, obs_traj = load_trajectory_csv(trajectory_csv)

    frames, times, obs, train_mask, test_mask = _build_rollout_timeline(
        train_rows,
        test_rows,
        frames_traj=frames_traj,
        times_traj=times_traj,
        obs_traj=obs_traj,
        fps=export_fps,
    )

    # Single rollout from fitted initial conditions through train, then test.
    # Test predictions are NOT re-simulated from p0 (step 4c spec).
    pred = simulate_trajectory(
        times,
        p0=params.p0,
        v0=params.v0,
        gravity_z=params.gravity_z,
        restitution=params.restitution,
        ground_z=params.ground_z,
    )

    test_indices = np.where(test_mask)[0]
    rows: list[dict[str, float | int]] = []
    for i in test_indices.tolist():
        rows.append(
            {
                "frame": int(frames[i]),
                "t_sec": float(times[i]),
                "x": float(pred[i, 0]),
                "y": float(pred[i, 1]),
                "z": float(pred[i, 2]),
            }
        )

    boundary_gap = float("nan")
    if frames_traj is not None:
        train_obs_idx = np.where(train_mask & np.isfinite(obs).all(axis=1))[0]
        if train_obs_idx.size > 0:
            prev_train = int(train_obs_idx[-1])
            first_test = int(test_indices[0])
            boundary_gap = float(np.linalg.norm(pred[first_test] - obs[prev_train]))

    min_clearance = float(np.min(pred[:, 2] - params.ground_z))

    gt_arr: np.ndarray | None = None
    if gt_poses is not None and gt_poses.is_file():
        trajectory = load_object_poses_csv(gt_poses)
        gt_arr = np.empty_like(pred)
        for i, frame in enumerate(frames.tolist()):
            gt_arr[i] = trajectory.by_frame(int(frame)).position

    pred_csv = out_dir / "trajectory_predicted.csv"
    plot_png = out_dir / "trajectory_full_plot.png"
    meta_json = out_dir / "step4c_meta.json"

    _write_predicted_csv(pred_csv, rows)
    _plot_full(plot_png, times, obs, pred, train_mask, gt=gt_arr)

    meta = {
        "fps": float(export_fps),
        "num_test_frames": len(rows),
        "num_test_frames_expected": len(test_rows),
        "boundary_gap_m": boundary_gap,
        "min_clearance_to_ground_m": min_clearance,
        "ground_z_m": float(params.ground_z),
        "ground_epsilon_m": float(ground_epsilon_m),
        "source_params": str(params_json.resolve()),
        "source_frame_map": str(frame_map_path.resolve()),
        "source_traj": str(trajectory_csv.resolve()) if trajectory_csv else None,
        "source_scene_config": str(scene_config.resolve()) if scene_config else None,
        "test_frame_min": int(min(r["frame"] for r in rows)),
        "test_frame_max": int(max(r["frame"] for r in rows)),
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if len(rows) != len(test_rows):
        raise ValueError(
            f"Predicted {len(rows)} test rows but frame_map lists {len(test_rows)}"
        )

    if strict:
        if np.isfinite(boundary_gap) and boundary_gap >= 0.01:
            raise ValueError(
                f"Boundary gap {boundary_gap:.4f} m >= 1 cm; check fit or train trajectory."
            )
        if min_clearance < -ground_epsilon_m:
            raise ValueError(
                f"Floor penetration: min(z - z_g) = {min_clearance:.6f} m"
            )

    return Step4cResult(
        out_dir=out_dir,
        predicted_csv=pred_csv,
        plot_png=plot_png,
        meta_json=meta_json,
        num_test_frames=len(rows),
        boundary_gap_m=boundary_gap,
        min_clearance_m=min_clearance,
    )
