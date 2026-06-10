"""Step 4b — fit discrete bounce physics to an observed 3D trajectory (CSV only)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from phys4d.poses import load_object_poses_csv


@dataclass(frozen=True)
class PhysicsParams:
    """Fitted simulator parameters: translation + vertical floor bounce."""

    p0: np.ndarray
    v0: np.ndarray
    gravity_z: float
    restitution: float
    ground_z: float


@dataclass(frozen=True)
class BounceEvent:
    """Approximate bounce timing and pre/post vertical velocity."""

    index: int
    t_sec: float
    v_before: float
    v_after: float


@dataclass
class Step4bResult:
    """Artifacts written under ``step4b/``."""

    out_dir: Path
    params_json: Path
    fit_plot_png: Path
    fit_mse_m2: float
    bounce_count: int


def load_trajectory_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory CSV as frame ids, times (s), and xyz positions."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory CSV: {path}")

    frames: list[int] = []
    times: list[float] = []
    positions: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"frame", "t_sec", "x", "y", "z"}
        if reader.fieldnames is None or not needed.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns {sorted(needed)}")
        for row in reader:
            frames.append(int(float(row["frame"])))
            times.append(float(row["t_sec"]))
            positions.append([float(row["x"]), float(row["y"]), float(row["z"])])

    if not frames:
        raise ValueError(f"No rows in trajectory CSV: {path}")

    frame_arr = np.asarray(frames, dtype=np.int32)
    time_arr = np.asarray(times, dtype=np.float64)
    pos_arr = np.asarray(positions, dtype=np.float64)
    if np.any(~np.isfinite(pos_arr)):
        raise ValueError(f"Trajectory has non-finite values: {path}")
    if np.any(np.diff(time_arr) <= 0.0):
        raise ValueError("Trajectory t_sec must be strictly increasing.")
    return frame_arr, time_arr, pos_arr


def load_physics_params_json(path: Path) -> PhysicsParams:
    """Read ``physics_params.json`` produced by :func:`run_step4b`."""

    path = path.resolve()
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


def detect_bounces(
    times: np.ndarray,
    z: np.ndarray,
    *,
    min_speed: float = 0.35,
    near_ground_margin: float = 0.06,
    max_restitution_like: float = 1.35,
) -> list[BounceEvent]:
    """Detect floor impacts from vz sign flips near local z minima."""

    dt = np.diff(times)
    vz = np.diff(z) / np.maximum(dt, 1e-9)
    z_floor = float(np.min(z))
    events: list[BounceEvent] = []

    for i in range(1, vz.shape[0]):
        v_before = float(vz[i - 1])
        v_after = float(vz[i])
        if not (v_before < -min_speed and v_after > min_speed):
            continue

        zi = float(z[i])
        z_prev = float(z[i - 1]) if i - 1 >= 0 else zi
        z_next = float(z[i + 1]) if i + 1 < z.shape[0] else zi
        is_local_min = zi <= z_prev and zi <= z_next
        is_near_floor = zi <= z_floor + float(near_ground_margin)
        if not (is_local_min and is_near_floor):
            continue

        e_like = (-v_after / v_before) if v_before < -1e-9 else np.inf
        if e_like > float(max_restitution_like):
            continue

        events.append(
            BounceEvent(
                index=i,
                t_sec=float(times[i]),
                v_before=v_before,
                v_after=v_after,
            )
        )
    return events


def simulate_trajectory(
    times: np.ndarray,
    *,
    p0: np.ndarray,
    v0: np.ndarray,
    gravity_z: float,
    restitution: float,
    ground_z: float,
) -> np.ndarray:
    """Forward simulate piecewise motion (discrete bounce integrator)."""

    times = np.asarray(times, dtype=np.float64)
    n = int(times.shape[0])
    out = np.empty((n, 3), dtype=np.float64)
    p = np.asarray(p0, dtype=np.float64).copy()
    v = np.asarray(v0, dtype=np.float64).copy()

    for i in range(n):
        out[i] = p.copy()
        if i + 1 >= n:
            break
        dt = float(times[i + 1] - times[i])
        if dt <= 0.0:
            raise ValueError("times must be strictly increasing")
        v[2] += float(gravity_z) * dt
        p = p + v * dt
        if p[2] < ground_z and v[2] < 0.0:
            v[2] = -float(restitution) * v[2]
            p[2] = ground_z

    return out


def _initial_guess(times: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Heuristic initialization from the first few observed points."""

    p0 = obs[0].copy()
    k = min(6, obs.shape[0])
    dt = times[1:k] - times[: k - 1]
    vel = (obs[1:k] - obs[: k - 1]) / np.maximum(dt[:, None], 1e-9)
    v0 = np.median(vel, axis=0)
    g0 = -9.81
    z_ground0 = float(np.min(obs[:, 2]))
    bounces = detect_bounces(times, obs[:, 2])
    if bounces:
        vals = [-ev.v_after / ev.v_before for ev in bounces if ev.v_before < -1e-6]
        e0 = float(np.clip(np.median(vals), 0.35, 0.95)) if vals else 0.8
    else:
        e0 = 0.8
    return p0, v0, g0, e0, z_ground0


def fit_physics_params(
    times: np.ndarray,
    obs: np.ndarray,
    *,
    fix_gravity: bool = True,
    robust_loss: str = "soft_l1",
    f_scale: float = 0.05,
    max_nfev: int = 1200,
) -> tuple[PhysicsParams, np.ndarray, float]:
    """Nonlinear least-squares fit for (p0, v0, g, e, z_g)."""

    p0, v0, g0, e0, z_ground0 = _initial_guess(times, obs)

    if fix_gravity:
        x0 = np.array([*p0, *v0, e0, z_ground0], dtype=np.float64)
    else:
        x0 = np.array([*p0, *v0, g0, e0, z_ground0], dtype=np.float64)

    p_span = float(np.max(np.linalg.norm(obs - obs[0], axis=1)) + 1.0)
    lo_common = [-p_span, -p_span, -p_span, -30.0, -30.0, -30.0]
    hi_common = [p_span, p_span, p_span * 2.0, 30.0, 30.0, 30.0]
    z_lo = float(np.min(obs[:, 2]) - 0.2)
    z_hi = float(np.min(obs[:, 2]) + 0.2)

    if fix_gravity:
        lower = np.array([*lo_common, 0.3, z_lo], dtype=np.float64)
        upper = np.array([*hi_common, 0.95, z_hi], dtype=np.float64)
    else:
        lower = np.array([*lo_common, -14.0, 0.3, z_lo], dtype=np.float64)
        upper = np.array([*hi_common, -4.0, 0.95, z_hi], dtype=np.float64)

    def residuals(x: np.ndarray) -> np.ndarray:
        if fix_gravity:
            p = x[0:3]
            v = x[3:6]
            g = -9.81
            e = x[6]
            z_g = x[7]
        else:
            p = x[0:3]
            v = x[3:6]
            g = x[6]
            e = x[7]
            z_g = x[8]
        pred = simulate_trajectory(
            times,
            p0=p,
            v0=v,
            gravity_z=g,
            restitution=e,
            ground_z=z_g,
        )
        return (pred - obs).reshape(-1)

    fit = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        loss=robust_loss,
        f_scale=float(f_scale),
        max_nfev=int(max_nfev),
    )

    x = fit.x
    if fix_gravity:
        params = PhysicsParams(
            p0=x[0:3].copy(),
            v0=x[3:6].copy(),
            gravity_z=-9.81,
            restitution=float(x[6]),
            ground_z=float(x[7]),
        )
    else:
        params = PhysicsParams(
            p0=x[0:3].copy(),
            v0=x[3:6].copy(),
            gravity_z=float(x[6]),
            restitution=float(x[7]),
            ground_z=float(x[8]),
        )

    pred = simulate_trajectory(
        times,
        p0=params.p0,
        v0=params.v0,
        gravity_z=params.gravity_z,
        restitution=params.restitution,
        ground_z=params.ground_z,
    )
    mse = float(np.mean(np.sum((pred - obs) ** 2, axis=1)))
    return params, pred, mse


def _plot_fit(
    path: Path,
    times: np.ndarray,
    obs: np.ndarray,
    pred: np.ndarray,
    ground_z: float,
    bounces: list[BounceEvent],
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    labels = ("x", "y", "z")
    for i, name in enumerate(labels):
        axes[i].plot(times, obs[:, i], "o-", lw=1.0, ms=2, alpha=0.5, label="observed")
        axes[i].plot(times, pred[:, i], "-", lw=2.0, label="fitted")
        axes[i].set_ylabel(f"{name} (m)")
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc="best", fontsize=8)

    axes[2].axhline(float(ground_z), color="k", linestyle="--", linewidth=1, alpha=0.7, label="ground")
    for ev in bounces:
        axes[2].axvline(ev.t_sec, color="tab:orange", alpha=0.35, linewidth=1)

    err = np.linalg.norm(pred - obs, axis=1)
    axes[3].plot(times, err, "-", lw=1.5, color="tab:red")
    axes[3].set_ylabel("||error|| (m)")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(alpha=0.3)
    fig.suptitle(f"Step 4b — physics fit ({len(bounces)} bounces detected)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_step4b(
    *,
    trajectory_csv: Path,
    out_dir: Path,
    fps: float | None = None,
    fix_gravity: bool = True,
    gt_poses_csv: Path | None = None,
    align_translation_to_gt: bool = False,
    robust_loss: str = "soft_l1",
    robust_f_scale: float = 0.05,
) -> Step4bResult:
    """Fit physics parameters and write ``physics_params.json`` + plot."""

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, times, obs = load_trajectory_csv(trajectory_csv)
    obs_for_fit = obs.copy()
    alignment_translation = np.zeros(3, dtype=np.float64)
    alignment_mse_before: float | None = None
    alignment_mse_after: float | None = None

    if align_translation_to_gt:
        if gt_poses_csv is None:
            raise ValueError("align_translation_to_gt requires gt_poses_csv")
        gt_traj = load_object_poses_csv(gt_poses_csv.resolve())
        gt = np.stack([gt_traj.by_frame(int(f)).position for f in frames.tolist()], axis=0)
        alignment_translation = np.mean(gt - obs_for_fit, axis=0)
        alignment_mse_before = float(np.mean(np.sum((obs_for_fit - gt) ** 2, axis=1)))
        obs_for_fit = obs_for_fit + alignment_translation[None, :]
        alignment_mse_after = float(np.mean(np.sum((obs_for_fit - gt) ** 2, axis=1)))

    if fps is None:
        dt = float(np.median(np.diff(times)))
        fps = 1.0 / max(dt, 1e-9)

    bounces = detect_bounces(times, obs_for_fit[:, 2])
    params, pred_fit_space, mse = fit_physics_params(
        times,
        obs_for_fit,
        fix_gravity=fix_gravity,
        robust_loss=robust_loss,
        f_scale=robust_f_scale,
    )

    if align_translation_to_gt:
        params = PhysicsParams(
            p0=params.p0 - alignment_translation,
            v0=params.v0.copy(),
            gravity_z=float(params.gravity_z),
            restitution=float(params.restitution),
            ground_z=float(params.ground_z - alignment_translation[2]),
        )
    pred = pred_fit_space - alignment_translation[None, :]

    params_json = out_dir / "physics_params.json"
    fit_plot_png = out_dir / "physics_fit_plot.png"
    _plot_fit(fit_plot_png, times, obs, pred, params.ground_z, bounces)

    payload = {
        "fit_mse_m2": mse,
        "fps": float(fps),
        "num_frames": int(frames.shape[0]),
        "frame_start": int(frames[0]),
        "frame_end": int(frames[-1]),
        "fix_gravity": bool(fix_gravity),
        "robust_loss": robust_loss,
        "robust_f_scale": float(robust_f_scale),
        "bounce_count": len(bounces),
        "bounces": [
            {
                "index": int(ev.index),
                "time_s": float(ev.t_sec),
                "v_before_m_s": float(ev.v_before),
                "v_after_m_s": float(ev.v_after),
            }
            for ev in bounces
        ],
        "params": {
            "p0": params.p0.tolist(),
            "v0": params.v0.tolist(),
            "gravity_z_m_s2": float(params.gravity_z),
            "restitution": float(params.restitution),
            "ground_z_m": float(params.ground_z),
        },
        "fit_alignment": {
            "enabled": bool(align_translation_to_gt),
            "translation_xyz_m": alignment_translation.tolist(),
            "mse_before_m2": alignment_mse_before,
            "mse_after_m2": alignment_mse_after,
        },
    }
    params_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return Step4bResult(
        out_dir=out_dir,
        params_json=params_json,
        fit_plot_png=fit_plot_png,
        fit_mse_m2=mse,
        bounce_count=len(bounces),
    )
