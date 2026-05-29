#!/usr/bin/env python
"""Baseline comparison for held-out trajectory prediction (Milestone 3).

Compares our physics-fit extrapolation against three reference baselines that a
pure-reconstruction or naive predictor would produce, on the held-out test
frames, scored against PyBullet ground truth:

  - 4dgs_only   : freeze the last observed centroid. This is what a pure 4DGS
                  reconstruction yields beyond the training window, because the
                  Wu deformation field is only defined on the observed time range
                  and clamps at t=1 (no future motion).
  - const_vel   : linear extrapolation from the train/test boundary state
                  (estimated velocity, no gravity).
  - ballistic   : gravity-only roll-forward from the boundary state, with NO
                  bounce model (no restitution / no floor contact). Isolates the
                  value of modeling the bounce.
  - ours        : our fitted discrete-bounce physics extrapolation (phase3/step4c).

All methods start from the SAME boundary state derived only from the observed
(train-window) 4DGS trajectory, so no ground-truth leaks into any predictor.

Outputs (under <out-dir>/baselines/):
  - metrics.json        per-method pos RMSE (m) and velocity / acceleration R^2
  - comparison.csv      flat table for dropping into the slide
  - compare_plot.png    z-vs-time overlay: GT + every method on the held-out span
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.metrics import _r2_pearson, _safe_gradient  # noqa: E402
from phys4d.bounce.physics import load_trajectory_csv, simulate_trajectory  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402

# A floor far below the scene disables bounce contact in simulate_trajectory.
_NO_FLOOR_Z = -1.0e9


def _read_pred_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a frame,t_sec,x,y,z CSV into (frames, times, xyz)."""

    frames, times, xyz = [], [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frames.append(int(float(row["frame"])))
            times.append(float(row["t_sec"]))
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return (
        np.asarray(frames, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
        np.asarray(xyz, dtype=np.float64),
    )


def _boundary_state(
    times_obs: np.ndarray,
    pos_obs: np.ndarray,
    *,
    window: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate (t_end, p_end, v_end) at the last observed frame.

    Position is the last observed sample; velocity is the least-squares slope over
    the final ``window`` observed samples (per axis), which is robust to the small
    centroid jitter in the 4DGS trajectory.
    """

    k = int(min(window, times_obs.shape[0]))
    t_tail = times_obs[-k:]
    p_tail = pos_obs[-k:]
    t_end = float(times_obs[-1])
    p_end = pos_obs[-1].copy()
    v_end = np.empty(3, dtype=np.float64)
    for axis in range(3):
        slope, _intercept = np.polyfit(t_tail - t_tail[0], p_tail[:, axis], 1)
        v_end[axis] = float(slope)
    return t_end, p_end, v_end


def _roll(
    t_end: float,
    p_end: np.ndarray,
    v_end: np.ndarray,
    times_test: np.ndarray,
    *,
    gravity_z: float,
    bounce: bool,
    restitution: float = 0.0,
    ground_z: float = _NO_FLOOR_Z,
) -> np.ndarray:
    """Integrate from the boundary state to the test times.

    We prepend t_end so the integrator advances exactly one step into the first
    test frame, then drop that seed row.
    """

    times = np.concatenate([[t_end], times_test])
    traj = simulate_trajectory(
        times,
        p0=p_end,
        v0=v_end,
        gravity_z=gravity_z,
        restitution=restitution if bounce else 0.0,
        ground_z=ground_z if bounce else _NO_FLOOR_Z,
    )
    return traj[1:]


def _metrics(pred: np.ndarray, gt: np.ndarray, fps: float) -> dict[str, float]:
    """Position RMSE plus velocity / acceleration Pearson R^2 (matches Step 6)."""

    rmse = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))
    dt = 1.0 / max(float(fps), 1e-9)
    vel_r2 = _r2_pearson(_safe_gradient(pred, dt), _safe_gradient(gt, dt))
    acc_r2 = _r2_pearson(
        _safe_gradient(_safe_gradient(pred, dt), dt),
        _safe_gradient(_safe_gradient(gt, dt), dt),
    )
    return {"pos_rmse_m": rmse, "vel_r2": vel_r2, "acc_r2": acc_r2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="bounce_pipeline/<ball> dir containing phase1/ and phase3/.",
    )
    parser.add_argument("--gt-poses", type=Path, required=True, help="object_poses.csv")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=7,
        help="Observed frames used to estimate boundary velocity.",
    )
    parser.add_argument(
        "--observed-csv",
        type=Path,
        default=None,
        help="Override observed train CSV (default: <out-dir>/phase1/trajectory_smoothed.csv).",
    )
    parser.add_argument(
        "--ours-csv",
        type=Path,
        default=None,
        help="Override our prediction CSV (default: <out-dir>/phase3/trajectory_predicted.csv).",
    )
    parser.add_argument("--label", type=str, default=None, help="Scene label for the plot title.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    observed_csv = args.observed_csv or out_dir / "phase1" / "trajectory_smoothed.csv"
    ours_csv = args.ours_csv or out_dir / "phase3" / "trajectory_predicted.csv"

    _obs_frames, t_obs, p_obs = load_trajectory_csv(observed_csv)
    test_frames, t_test, ours = _read_pred_csv(ours_csv)

    gt_traj = load_object_poses_csv(args.gt_poses.resolve())
    gt = np.stack([gt_traj.by_frame(int(f)).position for f in test_frames], axis=0)

    t_end, p_end, v_end = _boundary_state(t_obs, p_obs, window=args.boundary_window)

    preds: dict[str, np.ndarray] = {
        "4dgs_only": np.repeat(p_end[None, :], test_frames.shape[0], axis=0),
        "const_vel": _roll(t_end, p_end, v_end, t_test, gravity_z=0.0, bounce=False),
        "ballistic": _roll(t_end, p_end, v_end, t_test, gravity_z=-9.81, bounce=False),
        "ours": ours,
    }

    results = {name: _metrics(pred, gt, args.fps) for name, pred in preds.items()}

    bdir = out_dir / "baselines"
    bdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "scene": args.label or out_dir.name,
        "fps": float(args.fps),
        "test_frames": [int(test_frames[0]), int(test_frames[-1])],
        "num_test_frames": int(test_frames.shape[0]),
        "boundary_window": int(args.boundary_window),
        "boundary_state": {
            "t_end_s": t_end,
            "p_end_m": p_end.tolist(),
            "v_end_m_s": v_end.tolist(),
        },
        "methods": results,
        "sources": {
            "observed_csv": str(observed_csv),
            "ours_csv": str(ours_csv),
            "gt_poses_csv": str(args.gt_poses.resolve()),
        },
    }
    (bdir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    order = ["4dgs_only", "const_vel", "ballistic", "ours"]
    with (bdir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "pos_rmse_m", "vel_r2", "acc_r2"])
        for name in order:
            r = results[name]
            writer.writerow([name, f"{r['pos_rmse_m']:.6f}", f"{r['vel_r2']:.6f}", f"{r['acc_r2']:.6f}"])

    _plot(bdir / "compare_plot.png", test_frames, gt, preds, order, label=payload["scene"])

    print(f"Baseline comparison -> {bdir}")
    print(f"  scene: {payload['scene']}  test frames {payload['test_frames']}")
    print(f"  {'method':<12}{'RMSE (m)':>12}{'vel R2':>10}{'acc R2':>10}")
    for name in order:
        r = results[name]
        print(f"  {name:<12}{r['pos_rmse_m']:>12.4f}{r['vel_r2']:>10.4f}{r['acc_r2']:>10.4f}")
    return 0


def _plot(path: Path, frames, gt, preds, order, *, label: str) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "4dgs_only": dict(color="tab:gray", ls=":", marker=""),
        "const_vel": dict(color="tab:green", ls="-.", marker=""),
        "ballistic": dict(color="tab:orange", ls="--", marker=""),
        "ours": dict(color="tab:blue", ls="-", marker="o"),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frames, gt[:, 2], color="k", lw=2.5, alpha=0.85, label="GT PyBullet")
    for name in order:
        st = styles[name]
        ax.plot(
            frames,
            preds[name][:, 2],
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            ms=3,
            lw=2,
            label=name,
        )
    ax.set_xlabel("simulation frame (held-out)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Held-out trajectory prediction — {label}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
