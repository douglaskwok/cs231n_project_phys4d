"""Step 4a — opacity-weighted Wu 4DGS centroids, Savitzky–Golay smoothing, GT validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phys4d.poses import ObjectPoseTrajectory, load_object_poses_csv

from .load_4dgs import (
    LoadedWu4DGS,
    deform_positions_at_norm_time,
    geometric_median,
    load_wu_4dgs,
    opacity_weighted_centroid,
    robust_opacity_weighted_centroid,
    select_object_gaussian_set,
)


@dataclass(frozen=True)
class TrainTimestamp:
    """One training timestep: normalized time, seconds, and PyBullet frame index."""

    kept_index: int
    original_frame: int
    t_sec: float
    t_norm: float


@dataclass
class Step4aResult:
    """Artifacts written under ``step4a/``."""

    out_dir: Path
    raw_csv: Path
    smoothed_csv: Path
    plot_png: Path
    meta_json: Path
    num_frames: int
    train_mse_m2: float | None
    train_mse_aligned_m2: float | None


def load_train_timestamps(dynerf_export: Path) -> list[TrainTimestamp]:
    """Build per-frame normalized times matching Wu training (Gotcha #2 in milestone3)."""

    dynerf_export = dynerf_export.resolve()
    frame_map_path = dynerf_export / "frame_map.json"

    # Prefer frame_map train rows — preserves original PyBullet frame indices for GT.
    if frame_map_path.is_file():
        blob = json.loads(frame_map_path.read_text(encoding="utf-8"))
        rows = blob.get("train") or []
        if not rows:
            raise ValueError(f"No train entries in {frame_map_path}")
        times_s = sorted({float(r["original_time_s"]) for r in rows})
        t_min, t_max = times_s[0], times_s[-1]
        span = t_max - t_min
        if span < 1e-9:
            span = 1.0

        out: list[TrainTimestamp] = []
        for i, r in enumerate(sorted(rows, key=lambda x: float(x["original_time_s"]))):
            t_sec = float(r["original_time_s"])
            out.append(
                TrainTimestamp(
                    kept_index=i,
                    original_frame=int(r["original_frame"]),
                    t_sec=t_sec,
                    t_norm=(t_sec - t_min) / span,
                )
            )
        return out

    # Fallback: unique ``time`` values from transforms_train.json (milestone3 pseudocode).
    tf_path = dynerf_export / "transforms_train.json"
    if not tf_path.is_file():
        raise FileNotFoundError(
            f"Need frame_map.json or transforms_train.json under {dynerf_export}"
        )

    frames = json.loads(tf_path.read_text(encoding="utf-8")).get("frames") or []
    train_times = sorted(float(fr["time"]) for fr in frames)
    unique_times = sorted(set(train_times))
    t_min, t_max = unique_times[0], unique_times[-1]
    span = t_max - t_min
    if span < 1e-9:
        span = 1.0

    out = []
    for i, t_sec in enumerate(unique_times):
        frame_idx = i
        for fr in frames:
            if float(fr["time"]) == t_sec:
                stem = Path(fr["file_path"]).name
                frame_idx = int(stem.split("_")[-1])
                break
        out.append(
            TrainTimestamp(
                kept_index=i,
                original_frame=frame_idx,
                t_sec=t_sec,
                t_norm=(t_sec - t_min) / span,
            )
        )
    if not out:
        raise ValueError(f"No frames in {tf_path}")
    return out


def smooth_trajectory_savgol(
    positions: np.ndarray,
    *,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky–Golay per axis; shrink window on short series."""

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
    """Rigid alignment mapping pred → target (constant offset + rotation)."""

    from scipy.spatial import procrustes

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    mtx1, _, disparity = procrustes(p, t)
    return mtx1, float(disparity)


def write_trajectory_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
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
    for ax, axis, name in zip(axes, range(3), ("x", "y", "z")):
        ax.plot(t, raw[:, axis], "o-", ms=2, lw=1, alpha=0.45, label="raw 4DGS")
        ax.plot(t, smoothed[:, axis], "-", lw=2, label="smoothed")
        if gt is not None:
            ax.plot(t, gt[:, axis], "--", lw=1.5, alpha=0.8, label="GT PyBullet")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Step 4a — ball centroid trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gt_positions_for_frames(trajectory: ObjectPoseTrajectory, frames: list[int]) -> np.ndarray:
    out = np.empty((len(frames), 3), dtype=np.float64)
    for i, frame in enumerate(frames):
        out[i] = trajectory.by_frame(frame).position
    return out


def run_step4a(
    *,
    canonical_ply: Path,
    deform_path: Path,
    cfg_args_path: Path,
    dynerf_export: Path,
    out_dir: Path,
    gt_poses: Path | None = None,
    wu_root: Path | None = None,
    device: str = "cuda",
    savgol_window: int = 9,
    savgol_polyorder: int = 2,
    robust_centroid: bool = True,
    robust_k_mad: float = 3.0,
    robust_iter: int = 8,
    track_fixed_set: bool = True,
) -> Step4aResult:
    """Extract train-window trajectory CSVs and validation plot.

    When ``track_fixed_set`` (default), the object's Gaussian set is selected ONCE via
    density clustering and tracked through time with a weighted geometric median. This
    avoids the frame-to-frame membership flicker (and thus the centroid jitter) of
    per-frame MAD trimming, with no scene-specific thresholds.
    """

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = load_train_timestamps(dynerf_export)
    loaded: LoadedWu4DGS = load_wu_4dgs(
        canonical_ply=canonical_ply,
        deform_path=deform_path,
        cfg_args_path=cfg_args_path,
        wu_root=wu_root,
        device=device,
    )

    raw_rows: list[dict[str, float | int]] = []
    raw_positions: list[np.ndarray] = []
    times: list[float] = []
    frames: list[int] = []
    inlier_fracs: list[float] = []

    # Select the object Gaussian set once (fixed index set tracked across all frames).
    fixed_mask: np.ndarray | None = None
    ball_set_size: int | None = None
    if track_fixed_set:
        ref_xyz, ref_w = deform_positions_at_norm_time(loaded, timestamps[0].t_norm)
        fixed_mask = select_object_gaussian_set(ref_xyz, ref_w)
        ball_set_size = int(fixed_mask.sum())

    for ts in timestamps:
        xyz, weights = deform_positions_at_norm_time(loaded, ts.t_norm)
        if fixed_mask is not None:
            center = geometric_median(xyz[fixed_mask], weights[fixed_mask])
            inlier_fracs.append(float(fixed_mask.mean()))
        elif robust_centroid:
            center, mask = robust_opacity_weighted_centroid(
                xyz, weights, n_iter=robust_iter, k_mad=robust_k_mad
            )
            inlier_fracs.append(float(mask.mean()))
        else:
            center = opacity_weighted_centroid(xyz, weights)
        raw_positions.append(center)
        times.append(ts.t_sec)
        frames.append(ts.original_frame)
        raw_rows.append(
            {
                "frame": ts.original_frame,
                "t_sec": ts.t_sec,
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
    meta_json = out_dir / "step4a_meta.json"

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

    if np.any(~np.isfinite(raw_arr)) or np.any(~np.isfinite(smoothed_arr)):
        raise ValueError("Trajectory contains NaN or Inf values")
    t_arr = np.asarray(times)
    if np.any(np.diff(t_arr) < -1e-9):
        raise ValueError("t_sec is not monotonic")

    meta = {
        "canonical_ply": str(canonical_ply.resolve()),
        "deform_dir": str(loaded.deform_dir),
        "cfg_args": str(cfg_args_path.resolve()),
        "dynerf_export": str(dynerf_export.resolve()),
        "num_frames": len(raw_rows),
        "train_mse_m2": train_mse,
        "train_mse_aligned_m2": train_mse_aligned,
        "savgol_window": savgol_window,
        "savgol_polyorder": savgol_polyorder,
        "track_fixed_set": bool(track_fixed_set),
        "ball_set_size": ball_set_size,
        "centroid_method": "geometric_median_fixed_set" if track_fixed_set else (
            "robust_mad" if robust_centroid else "opacity_mean"
        ),
        "robust_centroid": robust_centroid,
        "robust_k_mad": robust_k_mad if robust_centroid else None,
        "robust_iter": robust_iter if robust_centroid else None,
        "robust_inlier_frac_mean": (
            float(np.mean(inlier_fracs)) if inlier_fracs else None
        ),
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return Step4aResult(
        out_dir=out_dir,
        raw_csv=raw_csv,
        smoothed_csv=smoothed_csv,
        plot_png=plot_png,
        meta_json=meta_json,
        num_frames=len(raw_rows),
        train_mse_m2=train_mse,
        train_mse_aligned_m2=train_mse_aligned,
    )
