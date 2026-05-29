"""Phase 1 — opacity-weighted 4DGS centroids, smoothing, and GT validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phys4d.poses import ObjectPoseTrajectory, load_object_poses_csv

from .load_4dgs import Loaded4DGS, deformed_gaussians_at_time, load_4dgs_checkpoint


@dataclass(frozen=True)
class TrainTimestamp:
    """One kept training timestep from ``frame_map.json``."""

    kept_index: int
    original_frame: int
    original_time_s: float


@dataclass
class Phase1Result:
    """Paths and summary metrics written by :func:`run_phase1`."""

    out_dir: Path
    raw_csv: Path
    smoothed_csv: Path
    plot_png: Path
    num_frames: int
    train_mse_m2: float | None
    train_mse_aligned_m2: float | None


def load_export_fps(dynerf_export: Path) -> float:
    meta_path = dynerf_export / "export_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return float(meta.get("fps", 60.0))
    frame_map_path = dynerf_export / "frame_map.json"
    if frame_map_path.is_file():
        return float(json.loads(frame_map_path.read_text(encoding="utf-8")).get("fps", 60.0))
    return 60.0


def load_train_timestamps(dynerf_export: Path) -> list[TrainTimestamp]:
    """Read kept train timestamps; fall back to unique ``time`` in transforms_train."""

    frame_map_path = dynerf_export / "frame_map.json"
    if frame_map_path.is_file():
        blob = json.loads(frame_map_path.read_text(encoding="utf-8"))
        rows = blob.get("train") or []
        if not rows:
            raise ValueError(f"No train entries in {frame_map_path}")
        return [
            TrainTimestamp(
                kept_index=int(r["kept_index"]),
                original_frame=int(r["original_frame"]),
                original_time_s=float(r["original_time_s"]),
            )
            for r in rows
        ]

    tf_path = dynerf_export / "transforms_train.json"
    if not tf_path.is_file():
        raise FileNotFoundError(
            f"Need frame_map.json or transforms_train.json under {dynerf_export}"
        )
    frames = json.loads(tf_path.read_text(encoding="utf-8")).get("frames") or []
    seen: dict[float, int] = {}
    out: list[TrainTimestamp] = []
    for fr in frames:
        t = float(fr["time"])
        if t in seen:
            continue
        stem = Path(fr["file_path"]).name
        # images/cam00_00016 -> frame 16
        frame_idx = int(stem.split("_")[-1])
        seen[t] = frame_idx
        out.append(
            TrainTimestamp(
                kept_index=len(out),
                original_frame=frame_idx,
                original_time_s=t,
            )
        )
    out.sort(key=lambda row: row.original_time_s)
    for i, row in enumerate(out):
        out[i] = TrainTimestamp(
            kept_index=i,
            original_frame=row.original_frame,
            original_time_s=row.original_time_s,
        )
    if not out:
        raise ValueError(f"No frames in {tf_path}")
    return out


def opacity_weighted_centroid(xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """3D center of mass; weights are already activated opacities (× marginal_t)."""

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("No Gaussians passed marginal/opacity filter")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total < 1e-12:
        return pts.mean(axis=0)
    return (pts * w[:, None]).sum(axis=0) / total


def smooth_trajectory_savgol(
    positions: np.ndarray,
    *,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky–Golay per axis; shrink window if the series is short."""

    from scipy.signal import savgol_filter

    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 3:
        return pos.copy()
    wl = min(int(window_length), n if n % 2 == 1 else n - 1)
    if wl < 3:
        return pos.copy()
    po = min(int(polyorder), wl - 1)
    out = np.empty_like(pos)
    for axis in range(3):
        out[:, axis] = savgol_filter(pos[:, axis], wl, po, mode="interp")
    return out


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid alignment (rotation + translation) mapping pred → target; returns scale."""

    from scipy.spatial import procrustes

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("pred and target must both be (N, 3)")
    mtx1, mtx2, disparity = procrustes(p, t)
    return mtx1, float(disparity)


def _gt_positions_for_frames(
    trajectory: ObjectPoseTrajectory, frames: list[int]
) -> np.ndarray:
    out = np.empty((len(frames), 3), dtype=np.float64)
    for i, frame in enumerate(frames):
        out[i] = trajectory.by_frame(frame).position
    return out


def write_trajectory_csv(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame", "t_sec", "x", "y", "z"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def plot_trajectory(
    path: Path,
    *,
    raw: np.ndarray,
    smoothed: np.ndarray,
    times_s: np.ndarray,
    gt: np.ndarray | None = None,
    gt_aligned: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(times_s, dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("x", "y", "z")
    for ax, axis, name in zip(axes, range(3), labels):
        ax.plot(t, raw[:, axis], "o-", ms=2, lw=1, alpha=0.45, label="raw 4DGS")
        ax.plot(t, smoothed[:, axis], "-", lw=2, label="smoothed")
        if gt is not None:
            ax.plot(t, gt[:, axis], "--", lw=1.5, alpha=0.8, label="GT PyBullet")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Phase 1 — ball centroid trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase1(
    *,
    checkpoint: Path,
    config: Path,
    dynerf_export: Path,
    out_dir: Path,
    gt_poses: Path | None = None,
    fourd_root: Path | None = None,
    device: str = "cuda",
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
    marginal_threshold: float = 0.05,
    include_test_timestamps: bool = False,
) -> Phase1Result:
    """Extract train-window trajectory CSVs and validation plot."""

    dynerf_export = dynerf_export.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = load_train_timestamps(dynerf_export)
    if include_test_timestamps:
        frame_map_path = dynerf_export / "frame_map.json"
        if frame_map_path.is_file():
            test_rows = json.loads(frame_map_path.read_text(encoding="utf-8")).get("test") or []
            base = len(timestamps)
            for j, r in enumerate(test_rows):
                timestamps.append(
                    TrainTimestamp(
                        kept_index=base + j,
                        original_frame=int(r["original_frame"]),
                        original_time_s=float(r["original_time_s"]),
                    )
                )

    loaded = load_4dgs_checkpoint(
        checkpoint=checkpoint,
        config=config,
        fourd_root=fourd_root,
        device=device,
    )

    raw_rows: list[dict[str, float | int]] = []
    raw_positions: list[np.ndarray] = []
    times: list[float] = []
    frames: list[int] = []

    for ts_row in timestamps:
        xyz, weights = deformed_gaussians_at_time(
            loaded,
            ts_row.original_time_s,
            marginal_threshold=marginal_threshold,
        )
        center = opacity_weighted_centroid(xyz, weights)
        raw_positions.append(center)
        times.append(ts_row.original_time_s)
        frames.append(ts_row.original_frame)
        raw_rows.append(
            {
                "frame": ts_row.original_frame,
                "t_sec": ts_row.original_time_s,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            }
        )

    raw_arr = np.stack(raw_positions, axis=0)
    smoothed_arr = smooth_trajectory_savgol(
        raw_arr,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )

    smoothed_rows: list[dict[str, float | int]] = []
    for i, row in enumerate(raw_rows):
        smoothed_rows.append(
            {
                "frame": row["frame"],
                "t_sec": row["t_sec"],
                "x": float(smoothed_arr[i, 0]),
                "y": float(smoothed_arr[i, 1]),
                "z": float(smoothed_arr[i, 2]),
            }
        )

    raw_csv = out_dir / "trajectory_raw.csv"
    smoothed_csv = out_dir / "trajectory_smoothed.csv"
    plot_png = out_dir / "trajectory_plot.png"
    write_trajectory_csv(raw_csv, raw_rows)
    write_trajectory_csv(smoothed_csv, smoothed_rows)

    gt_arr: np.ndarray | None = None
    gt_aligned: np.ndarray | None = None
    train_mse: float | None = None
    train_mse_aligned: float | None = None

    if gt_poses is not None and gt_poses.is_file():
        trajectory = load_object_poses_csv(gt_poses)
        gt_arr = _gt_positions_for_frames(trajectory, frames)
        train_mse = float(np.mean(np.sum((raw_arr - gt_arr) ** 2, axis=1)))
        gt_aligned, _ = procrustes_align(raw_arr, gt_arr)
        train_mse_aligned = float(np.mean(np.sum((raw_arr - gt_aligned) ** 2, axis=1)))

    plot_trajectory(
        plot_png,
        raw=raw_arr,
        smoothed=smoothed_arr,
        times_s=np.asarray(times),
        gt=gt_arr,
        gt_aligned=gt_aligned,
    )

    meta = {
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config.resolve()),
        "dynerf_export": str(dynerf_export),
        "time_duration": list(loaded.time_duration),
        "num_frames": len(raw_rows),
        "fps": load_export_fps(dynerf_export),
        "train_mse_m2": train_mse,
        "train_mse_aligned_m2": train_mse_aligned,
        "savgol_window": savgol_window,
        "savgol_polyorder": savgol_polyorder,
    }
    (out_dir / "phase1_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    return Phase1Result(
        out_dir=out_dir,
        raw_csv=raw_csv,
        smoothed_csv=smoothed_csv,
        plot_png=plot_png,
        num_frames=len(raw_rows),
        train_mse_m2=train_mse,
        train_mse_aligned_m2=train_mse_aligned,
    )
