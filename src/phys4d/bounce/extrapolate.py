"""Phase 3 — extrapolate fitted bounce physics onto held-out timestamps."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physics import PhysicsParams, _simulate_positions, load_trajectory_csv


@dataclass(frozen=True)
class Phase3Result:
    """Artifacts produced by phase 3 rollout."""

    out_dir: Path
    predicted_csv: Path
    plot_png: Path
    num_test_frames: int
    boundary_gap_m: float
    min_clearance_m: float


def _load_phase2_params(path: Path) -> PhysicsParams:
    """Read `physics_params.json` written by phase 2."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing phase2 params JSON: {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    params = blob.get("params") or {}
    needed = {"p0", "v0", "gravity_z_m_s2", "restitution", "ground_z_m"}
    if not needed.issubset(set(params.keys())):
        raise ValueError(f"{path} missing keys in params: {sorted(needed)}")
    return PhysicsParams(
        p0=np.asarray(params["p0"], dtype=np.float64).reshape(3),
        v0=np.asarray(params["v0"], dtype=np.float64).reshape(3),
        gravity_z=float(params["gravity_z_m_s2"]),
        restitution=float(params["restitution"]),
        ground_z=float(params["ground_z_m"]),
    )


def _read_split_timestamps(dynerf_export: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load train/test original frame indices from frame_map or export_meta."""

    frame_map = dynerf_export / "frame_map.json"
    if frame_map.is_file():
        blob = json.loads(frame_map.read_text(encoding="utf-8"))
        train = np.asarray([int(r["original_frame"]) for r in blob.get("train", [])], dtype=np.int32)
        test = np.asarray([int(r["original_frame"]) for r in blob.get("test", [])], dtype=np.int32)
        return train, test

    export_meta = dynerf_export / "export_meta.json"
    if export_meta.is_file():
        blob = json.loads(export_meta.read_text(encoding="utf-8"))
        train = np.asarray(blob.get("train_timestamps") or [], dtype=np.int32)
        test = np.asarray(blob.get("test_timestamps") or [], dtype=np.int32)
        return train, test

    raise FileNotFoundError(
        f"Need frame_map.json or export_meta.json under {dynerf_export}"
    )


def _export_fps(dynerf_export: Path, default_fps: float) -> float:
    """Read nominal FPS from export metadata when available."""

    export_meta = dynerf_export / "export_meta.json"
    if export_meta.is_file():
        blob = json.loads(export_meta.read_text(encoding="utf-8"))
        fps = float(blob.get("fps", default_fps))
        if fps > 0.0:
            return fps
    frame_map = dynerf_export / "frame_map.json"
    if frame_map.is_file():
        blob = json.loads(frame_map.read_text(encoding="utf-8"))
        fps = float(blob.get("fps", default_fps))
        if fps > 0.0:
            return fps
    return default_fps


def _build_rollout_timeline(
    frames_traj: np.ndarray,
    times_traj: np.ndarray,
    obs_traj: np.ndarray,
    train_frames: np.ndarray,
    test_frames: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge Phase-1 train observations with held-out test times for physics rollout.

    Phase 1 only extracts train-window centroids. Phase 3 still needs a time grid that
    covers test frames so the fitted simulator can extrapolate; test rows carry NaN obs.
    """

    train_set = {int(f) for f in train_frames.tolist()}
    test_set = {int(f) for f in test_frames.tolist()}
    if not train_set:
        raise ValueError("Export split has no train timestamps.")
    if not test_set:
        raise ValueError("Export split has no test timestamps.")

    # Index observed train centroids by sim frame id.
    by_frame: dict[int, tuple[float, np.ndarray]] = {}
    for i, frame in enumerate(frames_traj.tolist()):
        by_frame[int(frame)] = (float(times_traj[i]), obs_traj[i].copy())

    traj_train = train_set & set(by_frame.keys())
    if not traj_train:
        raise ValueError(
            "No overlap between phase1 trajectory frames and export train timestamps. "
            "Check --dynerf-export matches the export used for phase1."
        )

    rollout_frames = sorted(train_set | test_set)
    frames_out: list[int] = []
    times_out: list[float] = []
    obs_out: list[np.ndarray] = []
    for frame in rollout_frames:
        if frame in by_frame:
            t_sec, xyz = by_frame[frame]
        elif frame in test_set:
            # Held-out times: use export FPS (same convention as DyNeRF export).
            t_sec = float(frame) / float(fps)
            xyz = np.full(3, np.nan, dtype=np.float64)
        else:
            # Train frame missing from phase1 CSV — still include for a contiguous grid.
            t_sec = float(frame) / float(fps)
            xyz = np.full(3, np.nan, dtype=np.float64)
        frames_out.append(int(frame))
        times_out.append(t_sec)
        obs_out.append(xyz)

    frames_arr = np.asarray(frames_out, dtype=np.int32)
    times_arr = np.asarray(times_out, dtype=np.float64)
    obs_arr = np.stack(obs_out, axis=0)
    frame_list = frames_arr.tolist()
    train_mask = np.asarray([f in train_set for f in frame_list], dtype=bool)
    test_mask = np.asarray([f in test_set for f in frame_list], dtype=bool)
    if not np.any(test_mask):
        raise ValueError("Rollout timeline has no test frames after merge.")
    finite_train = train_mask & np.isfinite(obs_arr).all(axis=1)
    if not np.any(finite_train):
        raise ValueError("Rollout timeline has no finite train observations from phase1.")

    return frames_arr, times_arr, obs_arr, train_mask, test_mask


def _write_predicted_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write test-only prediction CSV with fixed column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "t_sec", "x", "y", "z"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def _plot_full(path: Path, times: np.ndarray, obs: np.ndarray, pred: np.ndarray, train_mask: np.ndarray) -> None:
    """Plot observed train, predicted full trajectory, and train/test split boundary."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, name in enumerate(labels):
        axes[i].plot(times[train_mask], obs[train_mask, i], "o", ms=2, alpha=0.5, label="train observed")
        axes[i].plot(times, pred[:, i], "-", lw=2.0, label="simulated full")
        axes[i].set_ylabel(f"{name} (m)")
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="best", fontsize=8)
    # Draw split marker at first test timestamp.
    test_idx = np.where(~train_mask)[0]
    if test_idx.size > 0:
        split_t = float(times[test_idx[0]])
        for ax in axes:
            ax.axvline(split_t, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Phase 3 — full rollout with held-out split")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase3(
    *,
    params_json: Path,
    trajectory_csv: Path,
    dynerf_export: Path,
    out_dir: Path,
    fps: float = 60.0,
) -> Phase3Result:
    """Roll fitted physics forward and emit held-out predictions."""

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    params = _load_phase2_params(params_json)

    frames_traj, times_traj, obs_traj = load_trajectory_csv(trajectory_csv)
    dynerf_export = dynerf_export.resolve()
    train_frames, test_frames = _read_split_timestamps(dynerf_export)

    if test_frames.size == 0:
        raise ValueError(
            "No held-out test timestamps in export. This run looks like --all-train; "
            "re-export/re-train with a split-respecting object-only export."
        )

    export_fps = _export_fps(dynerf_export, default_fps=fps)
    frames, times, obs, train_mask, test_mask = _build_rollout_timeline(
        frames_traj,
        times_traj,
        obs_traj,
        train_frames,
        test_frames,
        fps=export_fps,
    )

    pred = _simulate_positions(
        times,
        p0=params.p0,
        v0=params.v0,
        gravity_z=params.gravity_z,
        restitution=params.restitution,
        ground_z=params.ground_z,
    )

    # Continuity check at split boundary: first test sim point vs last observed train centroid.
    test_indices = np.where(test_mask)[0]
    first_test = int(test_indices[0])
    train_obs_idx = np.where(train_mask & np.isfinite(obs).all(axis=1))[0]
    if train_obs_idx.size == 0:
        raise ValueError("No finite train observations for boundary check.")
    prev_train = int(train_obs_idx[-1])
    boundary_gap = float(np.linalg.norm(pred[first_test] - obs[prev_train]))
    min_clearance = float(np.min(pred[:, 2] - params.ground_z))

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

    pred_csv = out_dir / "trajectory_predicted.csv"
    plot_png = out_dir / "trajectory_full_plot.png"
    _write_predicted_csv(pred_csv, rows)
    _plot_full(plot_png, times, obs, pred, train_mask=train_mask)

    meta = {
        "fps": float(export_fps),
        "num_phase1_frames": int(frames_traj.shape[0]),
        "num_total_frames": int(frames.shape[0]),
        "num_test_frames": int(len(rows)),
        "boundary_gap_m": boundary_gap,
        "min_clearance_to_ground_m": min_clearance,
        "ground_z_m": float(params.ground_z),
        "source_params": str(params_json.resolve()),
        "source_traj": str(trajectory_csv.resolve()),
        "source_export": str(dynerf_export.resolve()),
        "test_frame_min": int(min(r["frame"] for r in rows)),
        "test_frame_max": int(max(r["frame"] for r in rows)),
    }
    (out_dir / "phase3_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return Phase3Result(
        out_dir=out_dir,
        predicted_csv=pred_csv,
        plot_png=plot_png,
        num_test_frames=len(rows),
        boundary_gap_m=boundary_gap,
        min_clearance_m=min_clearance,
    )
