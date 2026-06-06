#!/usr/bin/env python
"""Re-fit bounce physics in the GT metric frame on the dataset's own train/test split.

Rationale (see scene config.json): the dataset defines
``train_frames=[0,119]`` and ``test_frames=[120,179]`` and ships exact GT
(gravity -9.80665, restitution 0.93, ball radius 0.2, table-top 0.40 ->
contact-center z = 0.60). The Step 4a trajectory lives in the *un-scaled 3DGS
frame*, so fitting free gravity/ground there is ill-posed.

This script:
  1. Loads the Step 4a trajectory (3DGS frame) and GT object_poses.csv.
  2. Estimates a similarity transform (scale,R,t) on the TRAIN frames only
     (Umeyama) mapping 3DGS -> GT metric frame, and applies it to all frames.
  3. Fits (p0, v0, e) on the train window with gravity and ground PLANE fixed
     to GT (optionally free via flags).
  4. Forward-simulates over the held-out TEST window and reports position RMSE
     vs GT, plus bounce counts over train / full.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.dataset_split import split_from_gt_poses  # noqa: E402
from phys4d.bounce.physics import detect_bounces, simulate_trajectory  # noqa: E402


def _load_traj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fr = np.array([int(float(r["frame"])) for r in rows], dtype=np.int64)
    t = np.array([float(r["t_sec"]) for r in rows], dtype=np.float64)
    p = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float64)
    return fr, t, p


def _load_gt(path: Path) -> dict[int, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out: dict[int, np.ndarray] = {}
    for r in rows:
        out[int(float(r["frame"]))] = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])], dtype=np.float64)
    return out


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform mapping src -> dst: dst ~= s * R @ src + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (s * (R @ pts.T)).T + t[None, :]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True, help="Step 4a trajectory_smoothed.csv (3DGS frame)")
    ap.add_argument("--gt-poses", type=Path, required=True, help="scene object_poses.csv (metric GT)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--train-start",
        type=int,
        default=None,
        help="Train window start frame (default: canonical 65%% split from GT poses)",
    )
    ap.add_argument("--train-end", type=int, default=None)
    ap.add_argument("--test-start", type=int, default=None)
    ap.add_argument("--test-end", type=int, default=None)
    ap.add_argument("--gravity", type=float, default=-9.80665)
    ap.add_argument("--ground", type=float, default=None, help="contact-center z (default: GT z-min)")
    ap.add_argument("--fit-gravity", action="store_true")
    ap.add_argument("--fit-ground", action="store_true")
    ap.add_argument(
        "--fit-objective",
        choices=["pos", "pos_zvel_ext", "pos_zvel_gt"],
        default="pos",
        help=(
            "Residual used by each least-squares candidate: xyz position only, "
            "or xyz position plus z-velocity agreement against extracted/GT train trajectory."
        ),
    )
    ap.add_argument(
        "--selection-metric",
        choices=[
            "train_z_rmse",
            "train_xyz_rmse",
            "train_ext_zvel_r2",
            "train_gt_zvel_r2",
            "train_ext_vel_r2",
            "train_gt_vel_r2",
        ],
        default="train_xyz_rmse",
        help="Metric used to choose among multi-start physics fits.",
    )
    ap.add_argument(
        "--velocity-residual-weight",
        type=float,
        default=0.03,
        help="Scale applied to velocity residual terms for --fit-objective pos_zvel_*.",
    )
    ap.add_argument(
        "--restitution-starts",
        type=str,
        default="0.35,0.5,0.65,0.75,0.85,0.9,0.95",
        help="Comma-separated initial restitution guesses for multi-start fitting.",
    )
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    canonical = split_from_gt_poses(args.gt_poses)
    train_start = int(args.train_start if args.train_start is not None else canonical["train"][0])
    train_end = int(args.train_end if args.train_end is not None else canonical["train"][1])
    test_start = int(args.test_start if args.test_start is not None else canonical["test"][0])
    test_end = int(args.test_end if args.test_end is not None else canonical["test"][1])

    fr, t, traj3d = _load_traj(args.traj)
    gt = _load_gt(args.gt_poses)
    frame_to_i = {int(f): i for i, f in enumerate(fr.tolist())}

    # frames we have in BOTH traj and GT
    common = [f for f in fr.tolist() if f in gt]
    train_f = [f for f in common if train_start <= f <= train_end]
    test_f = [f for f in common if test_start <= f <= test_end]
    if len(train_f) < 8:
        raise SystemExit(f"Too few train frames ({len(train_f)}) in [{train_start},{train_end}]")

    src_train = np.stack([traj3d[frame_to_i[f]] for f in train_f], axis=0)
    dst_train = np.stack([gt[f] for f in train_f], axis=0)
    s, R, t_vec = umeyama(src_train, dst_train)

    # map full trajectory into metric frame
    traj_m = apply_sim(s, R, t_vec, traj3d)

    times = t.copy()
    z_metric = traj_m[:, 2]
    ground = float(args.ground) if args.ground is not None else float(min(gt[f][2] for f in common))

    # --- fit (p0,v0,e[,g][,ground]) on TRAIN window in metric frame ---
    tr_idx = np.array([frame_to_i[f] for f in train_f])
    t_tr = times[tr_idx]
    t_tr = t_tr - t_tr[0]  # start at 0 for the simulator
    obs_tr = traj_m[tr_idx]
    gt_tr = np.stack([gt[f] for f in train_f], axis=0)

    # initial guesses
    p0_0 = obs_tr[0].copy()
    k = min(6, obs_tr.shape[0])
    v0_0 = np.median((obs_tr[1:k] - obs_tr[: k - 1]) / np.maximum(np.diff(t_tr[:k])[:, None], 1e-9), axis=0)
    g0 = args.gravity
    grd0 = ground

    free_g = bool(args.fit_gravity)
    free_grd = bool(args.fit_ground)
    lo = [-50, -50, -50, -50, -50, -50, 0.3]
    hi = [50, 50, 50, 50, 50, 50, 0.999]
    if free_g:
        lo.append(-14.0); hi.append(-4.0)
    if free_grd:
        lo.append(ground - 0.3); hi.append(ground + 0.3)

    def unpack(x):
        p = np.array(x[0:3]); v = np.array(x[3:6]); e = float(x[6])
        i = 7
        g = float(x[i]) if free_g else g0
        if free_g: i += 1
        grd = float(x[i]) if free_grd else grd0
        return p, v, g, e, grd

    def resid(x):
        p, v, g, e, grd = unpack(x)
        pred = simulate_trajectory(t_tr, p0=p, v0=v, gravity_z=g, restitution=e, ground_z=grd)
        pos_resid = (pred - obs_tr).reshape(-1)
        if args.fit_objective == "pos":
            return pos_resid
        dt = float(np.median(np.diff(t_tr))) if t_tr.shape[0] > 1 else 1.0 / 120.0
        pred_vz = np.gradient(pred[:, 2], dt)
        if args.fit_objective == "pos_zvel_ext":
            target_vz = np.gradient(obs_tr[:, 2], dt)
        elif args.fit_objective == "pos_zvel_gt":
            target_vz = np.gradient(gt_tr[:, 2], dt)
        else:
            raise AssertionError(args.fit_objective)
        return np.concatenate([pos_resid, float(args.velocity_residual_weight) * (pred_vz - target_vz)])

    def r2_pearson(pred: np.ndarray, target: np.ndarray) -> float:
        p = pred.reshape(-1).astype(np.float64)
        q = target.reshape(-1).astype(np.float64)
        p = p - float(np.mean(p))
        q = q - float(np.mean(q))
        denom = float(np.linalg.norm(p) * np.linalg.norm(q))
        if denom <= 1e-12:
            return 0.0
        corr = float(np.dot(p, q) / denom)
        return float(np.clip(corr * corr, 0.0, 1.0))

    starts = [float(v.strip()) for v in args.restitution_starts.split(",") if v.strip()]
    if not starts:
        starts = [0.9]
    candidates: list[dict[str, object]] = []
    for e0 in starts:
        x0 = [*p0_0, *v0_0, float(np.clip(e0, 0.3, 0.999))]
        if free_g:
            x0.append(g0)
        if free_grd:
            x0.append(grd0)
        fit_i = least_squares(
            resid,
            x0=np.array(x0),
            bounds=(np.array(lo), np.array(hi)),
            method="trf",
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=4000,
        )
        p_i, v_i, g_i, e_i, grd_i = unpack(fit_i.x)
        pred_i = simulate_trajectory(t_tr, p0=p_i, v0=v_i, gravity_z=g_i, restitution=e_i, ground_z=grd_i)
        err_i = pred_i - obs_tr
        dt_i = float(np.median(np.diff(t_tr))) if t_tr.shape[0] > 1 else 1.0 / 120.0
        pred_vel_i = np.gradient(pred_i, dt_i, axis=0)
        obs_vel_i = np.gradient(obs_tr, dt_i, axis=0)
        gt_vel_i = np.gradient(gt_tr, dt_i, axis=0)
        train_xyz_rmse_i = float(np.sqrt(np.mean(np.sum(err_i ** 2, axis=1))))
        train_z_rmse_i = float(np.sqrt(np.mean(err_i[:, 2] ** 2)))
        candidates.append(
            {
                "initial_restitution": float(e0),
                "params": fit_i.x.tolist(),
                "cost": float(fit_i.cost),
                "success": bool(fit_i.success),
                "message": str(fit_i.message),
                "train_xyz_rmse_m": train_xyz_rmse_i,
                "train_z_rmse_m": train_z_rmse_i,
                "train_ext_vel_r2": r2_pearson(pred_vel_i, obs_vel_i),
                "train_gt_vel_r2": r2_pearson(pred_vel_i, gt_vel_i),
                "train_ext_zvel_r2": r2_pearson(pred_vel_i[:, 2], obs_vel_i[:, 2]),
                "train_gt_zvel_r2": r2_pearson(pred_vel_i[:, 2], gt_vel_i[:, 2]),
                "fitted_restitution": float(e_i),
            }
        )

    metric_to_key = {
        "train_z_rmse": "train_z_rmse_m",
        "train_xyz_rmse": "train_xyz_rmse_m",
        "train_ext_zvel_r2": "train_ext_zvel_r2",
        "train_gt_zvel_r2": "train_gt_zvel_r2",
        "train_ext_vel_r2": "train_ext_vel_r2",
        "train_gt_vel_r2": "train_gt_vel_r2",
    }
    key = metric_to_key[args.selection_metric]
    if key.endswith("_r2"):
        best = max(candidates, key=lambda c: float(c[key]))
    else:
        best = min(candidates, key=lambda c: float(c[key]))
    p_fit, v_fit, g_fit, e_fit, grd_fit = unpack(np.asarray(best["params"], dtype=np.float64))
    pred_train = simulate_trajectory(t_tr, p0=p_fit, v0=v_fit, gravity_z=g_fit, restitution=e_fit, ground_z=grd_fit)
    train_err = pred_train - obs_tr
    train_mse = float(np.mean(np.sum(train_err ** 2, axis=1)))
    train_xyz_rmse = float(np.sqrt(train_mse))
    train_z_rmse = float(np.sqrt(np.mean(train_err[:, 2] ** 2)))

    # --- forward simulate over FULL available window from train start ---
    full_idx = np.array([frame_to_i[f] for f in common])
    t_full = times[full_idx] - times[tr_idx[0]]
    pred_full = simulate_trajectory(t_full, p0=p_fit, v0=v_fit, gravity_z=g_fit,
                                    restitution=e_fit, ground_z=grd_fit)
    gt_full = np.stack([gt[f] for f in common], axis=0)

    # held-out test RMSE
    test_mask = np.array([(test_start <= f <= test_end) for f in common])
    test_err = np.linalg.norm(pred_full[test_mask] - gt_full[test_mask], axis=1)
    test_rmse = float(np.sqrt(np.mean(test_err ** 2))) if test_mask.any() else float("nan")

    # bounce counts (contacts) — GT vs predicted vs extracted, over full window
    def contacts(z, zc):
        c = 0
        for i in range(1, len(z) - 1):
            if z[i] < z[i - 1] and z[i] <= z[i + 1] and z[i] <= zc + 0.03:
                c += 1
        return c
    zc = ground
    gt_bounces = contacts(gt_full[:, 2], zc)
    pred_bounces = contacts(pred_full[:, 2], zc)
    extracted_bounces = contacts(traj_m[full_idx][:, 2], zc)
    train_detect = len(detect_bounces(t_tr, obs_tr[:, 2]))

    result = {
        "similarity": {"scale": s, "scale_inv": 1.0 / s, "R": R.tolist(), "t": t_vec.tolist()},
        "split": {
            "canonical": canonical,
            "train": [train_start, train_end],
            "test": [test_start, test_end],
        },
        "fixed": {"gravity": not free_g, "ground": not free_grd},
        "params": {
            "p0_m": p_fit.tolist(), "v0_m_s": v_fit.tolist(),
            "gravity_z_m_s2": g_fit, "restitution": e_fit, "ground_z_m": grd_fit,
        },
        "gt": {"gravity": args.gravity, "restitution": 0.93, "ground_contact_z_m": ground},
        "train_fit_mse_m2": train_mse,
        "train_fit_rmse_m": train_xyz_rmse,
        "train_z_rmse_m": train_z_rmse,
        "fit_selection": {
            "metric": args.selection_metric,
            "fit_objective": args.fit_objective,
            "velocity_residual_weight": float(args.velocity_residual_weight),
            "selected_initial_restitution": best["initial_restitution"],
            "candidates": candidates,
        },
        "heldout_test_rmse_m": test_rmse,
        "bounces": {
            "gt_full": gt_bounces, "predicted_full": pred_bounces,
            "extracted_full": extracted_bounces, "train_detected": train_detect,
        },
        "n_frames": {"train": len(train_f), "test": len(test_f), "full": len(common)},
    }
    (out / "refit_metric.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # --- plot z(t): GT vs extracted(metric) vs fitted/predicted, mark split ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tt = t_full
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    names = ["x", "y", "z"]
    for j, nm in enumerate(names):
        axes[j].plot(tt, gt_full[:, j], "g-", lw=2, label="GT")
        axes[j].plot(tt, traj_m[full_idx][:, j], "k.", ms=3, alpha=0.45, label="extracted (4DGS→metric)")
        axes[j].plot(tt, pred_full[:, j], "r-", lw=1.8, label="physics fit/predict")
        axes[j].axvline(t_full[test_mask.argmax()] if test_mask.any() else tt[-1],
                        color="b", ls="--", alpha=0.5)
        axes[j].set_ylabel(f"{nm} (m)")
        axes[j].grid(alpha=0.3)
        axes[j].legend(loc="best", fontsize=8)
    axes[2].axhline(ground, color="gray", ls=":", lw=1, label="contact plane")
    axes[0].set_title(
        f"Metric-frame refit  |  e={e_fit:.3f} (GT 0.93)  g={g_fit:.2f}  "
        f"scale={s:.3f}  test RMSE={test_rmse*100:.2f} cm  "
        f"bounces GT={gt_bounces}/pred={pred_bounces}"
    )
    axes[2].set_xlabel("time since train start (s)")
    fig.tight_layout()
    fig.savefig(out / "refit_metric_plot.png", dpi=140)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out/'refit_metric.json'} and {out/'refit_metric_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
