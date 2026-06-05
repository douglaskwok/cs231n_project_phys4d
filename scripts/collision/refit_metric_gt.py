#!/usr/bin/env python
"""Re-fit multi-body collision physics in the GT metric frame (collision pipeline).

Joint analogue of ``scripts/bounce/refit_metric_gt.py``. For each object it:
  1. Loads the Step 4a per-object trajectory (3DGS frame) and that object's GT poses.
  2. Estimates a per-object similarity transform (Umeyama) on TRAIN frames mapping
     3DGS -> GT metric frame, and applies it to all frames.
Then it fits ALL bodies jointly on the train window with gravity + ground plane
fixed to GT (optionally free), using the coupled multi-body simulator. Masses and
radii come from the scene ``config.json`` (object-object collisions only constrain
the mass ratio, so masses are fixed by default).

Outputs ``refit_metric.json`` with a per-body ``bodies`` array plus the shared
``restitution_pair``, mirroring the bounce schema.
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
from phys4d.collision.physics import Body, count_pair_collisions, simulate_scene  # noqa: E402


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


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
        out[int(float(r["frame"]))] = np.array(
            [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])], dtype=np.float64
        )
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


def _radii_masses_from_config(config_path: Path | None, n_obj: int) -> tuple[list[float], list[float]]:
    radii = [0.2] * n_obj
    masses = [1.0] * n_obj
    if config_path is not None and config_path.is_file():
        blob = json.loads(config_path.read_text(encoding="utf-8"))
        scene = blob.get("scene") or {}
        objects = scene.get("objects") or []
        params = scene.get("scenario_params") or blob.get("scenario_params") or {}
        # Fallback radius = in-plane box half-extent (collision axis), shared by both
        # objects unless a per-object radius_m is present.
        half = params.get("object_half_extents_m")
        default_radius = float(half[0]) if half else None
        mass_keys = ["object_a_mass_kg", "object_b_mass_kg"]
        for k in range(n_obj):
            obj = objects[k] if k < len(objects) else {}
            if "radius_m" in obj:
                radii[k] = float(obj["radius_m"])
            elif default_radius is not None:
                radii[k] = default_radius
            if "mass_kg" in obj:
                masses[k] = float(obj["mass_kg"])
            elif k < len(mass_keys) and mass_keys[k] in params:
                masses[k] = float(params[mass_keys[k]])
    return radii, masses


def _walls_from_config(config_path: Path | None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return ``(wall_x, wall_y)`` surface bounds from the scene config (or ``None``)."""
    if config_path is None or not config_path.is_file():
        return None, None
    blob = json.loads(config_path.read_text(encoding="utf-8"))
    scene = blob.get("scene") or {}
    params = scene.get("scenario_params") or blob.get("scenario_params") or {}
    if not params.get("walled"):
        return None, None
    wx = params.get("wall_half_x_m")
    wy = params.get("wall_half_y_m")
    wall_x = (-float(wx), float(wx)) if wx else None
    wall_y = (-float(wy), float(wy)) if wy else None
    return wall_x, wall_y


def _parse_bound(value: str | None) -> tuple[float, float] | None:
    if not value or value.strip().lower() in ("", "none"):
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--wall-x/--wall-y expect 'min,max', got: {value}")
    return (float(parts[0]), float(parts[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", required=True, help="Comma-separated per-object trajectory_smoothed.csv")
    ap.add_argument("--gt-poses", required=True, help="Comma-separated per-object object_poses.csv")
    ap.add_argument("--config", type=Path, default=None, help="scene config.json (radii + masses)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-start", type=int, default=None)
    ap.add_argument("--train-end", type=int, default=None)
    ap.add_argument("--test-start", type=int, default=None)
    ap.add_argument("--test-end", type=int, default=None)
    ap.add_argument("--gravity", type=float, default=-9.80665)
    ap.add_argument("--ground", type=float, default=None, help="contact-center z (default: GT z-min over objects)")
    ap.add_argument("--restitution-pair", type=float, default=0.9, help="initial object-object COR")
    ap.add_argument("--restitution-wall", type=float, default=0.9, help="initial wall COR")
    ap.add_argument("--substeps", type=int, default=8, help="integrator sub-steps per frame")
    ap.add_argument("--fit-gravity", action="store_true")
    ap.add_argument("--fit-ground", action="store_true")
    ap.add_argument("--no-fit-wall-restitution", action="store_true", help="keep wall COR fixed")
    ap.add_argument("--no-walls", action="store_true", help="disable room walls even if config has them")
    ap.add_argument("--wall-x", default=None, help="override x wall surfaces 'min,max' (metric)")
    ap.add_argument("--wall-y", default=None, help="override y wall surfaces 'min,max' (metric)")
    ap.add_argument(
        "--fit-mass-ratio",
        action="store_true",
        help="fit per-object mass ratio m_k/m_0 (body 0 fixed as reference; absolute mass unobservable)",
    )
    ap.add_argument(
        "--fit-walls",
        action="store_true",
        help="fit x-wall surface positions (only identifiable where a body bounces off them)",
    )
    args = ap.parse_args()

    traj_paths = [Path(p) for p in _split(args.traj)]
    gt_paths = [Path(p) for p in _split(args.gt_poses)]
    n_obj = len(traj_paths)
    if len(gt_paths) != n_obj:
        raise SystemExit("--traj and --gt-poses must list the same number of objects")
    if n_obj < 2:
        raise SystemExit("collision refit needs at least 2 objects")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Canonical split from the first object's GT (all objects share the timeline).
    canonical = split_from_gt_poses(gt_paths[0])
    train_start = int(args.train_start if args.train_start is not None else canonical["train"][0])
    train_end = int(args.train_end if args.train_end is not None else canonical["train"][1])
    test_start = int(args.test_start if args.test_start is not None else canonical["test"][0])
    test_end = int(args.test_end if args.test_end is not None else canonical["test"][1])

    radii, masses = _radii_masses_from_config(args.config, n_obj)

    # --- room walls (axis-aligned enclosure) ---
    if args.no_walls:
        wall_x, wall_y = None, None
    else:
        wall_x, wall_y = _walls_from_config(args.config)
        if args.wall_x is not None:
            wall_x = _parse_bound(args.wall_x)
        if args.wall_y is not None:
            wall_y = _parse_bound(args.wall_y)
    fit_wall = (not args.no_fit_wall_restitution) and (wall_x is not None or wall_y is not None)

    # --- per-object load + similarity to metric frame ---
    sims: list[tuple[float, np.ndarray, np.ndarray]] = []
    traj_m: list[np.ndarray] = []
    frame_to_i: list[dict[int, int]] = []
    gts: list[dict[int, np.ndarray]] = []
    times_ref: np.ndarray | None = None
    common: list[int] | None = None

    for k in range(n_obj):
        fr, t, traj3d = _load_traj(traj_paths[k])
        gt = _load_gt(gt_paths[k])
        f2i = {int(f): i for i, f in enumerate(fr.tolist())}
        ck = [f for f in fr.tolist() if f in gt]
        train_f = [f for f in ck if train_start <= f <= train_end]
        if len(train_f) < 8:
            raise SystemExit(f"obj{k}: too few train frames ({len(train_f)})")
        src = np.stack([traj3d[f2i[f]] for f in train_f], axis=0)
        dst = np.stack([gt[f] for f in train_f], axis=0)
        s, R, tvec = umeyama(src, dst)
        sims.append((s, R, tvec))
        traj_m.append(apply_sim(s, R, tvec, traj3d))
        frame_to_i.append(f2i)
        gts.append(gt)
        if times_ref is None:
            times_ref = t
            common = ck
        else:
            common = [f for f in common if f in set(ck)]

    assert common is not None and times_ref is not None
    train_f = [f for f in common if train_start <= f <= train_end]
    test_f = [f for f in common if test_start <= f <= test_end]

    ground = (
        float(args.ground)
        if args.ground is not None
        else float(min(min(gts[k][f][2] for f in common) for k in range(n_obj)))
    )

    # --- train-window observations per body in metric frame ---
    tr_idx = [np.array([frame_to_i[k][f] for f in train_f]) for k in range(n_obj)]
    t_tr = times_ref[np.array([frame_to_i[0][f] for f in train_f])]
    t_tr = t_tr - t_tr[0]
    obs_tr = [traj_m[k][tr_idx[k]] for k in range(n_obj)]  # each (T,3)

    # --- initial guesses per body ---
    def init_body(k: int) -> tuple[np.ndarray, np.ndarray]:
        # Seed v0 with a least-squares line over a short pre-collision window rather
        # than a median of a few noisy finite differences. The extracted trajectory
        # is noisy (esp. in the near-constant lateral axes); a per-axis slope fit is
        # far more stable and avoids seeding spurious lateral velocity that a robust
        # loss would then "protect" as outliers.
        o = obs_tr[k]
        T = o.shape[0]
        m = max(2, min(T, 10))
        tt = t_tr[:m] - t_tr[0]
        A = np.vstack([tt, np.ones_like(tt)]).T
        coef, *_ = np.linalg.lstsq(A, o[:m], rcond=None)
        v0 = coef[0]
        p0 = coef[1]
        return p0, v0

    free_g = bool(args.fit_gravity)
    free_grd = bool(args.fit_ground)
    fit_mass_ratio = bool(args.fit_mass_ratio) and n_obj >= 2
    fit_walls = bool(args.fit_walls) and wall_x is not None

    x0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for k in range(n_obj):
        p0, v0 = init_body(k)
        x0 += [*p0, *v0, 0.9]  # p(3), v(3), e_floor
        lo += [-50, -50, -50, -50, -50, -50, 0.3]
        hi += [50, 50, 50, 50, 50, 50, 0.999]
    x0.append(float(args.restitution_pair)); lo.append(0.0); hi.append(0.999)  # e_pair
    g0 = float(args.gravity)
    grd0 = ground
    e_wall0 = float(args.restitution_wall)
    if free_g:
        x0.append(g0); lo.append(-14.0); hi.append(-4.0)
    if free_grd:
        x0.append(grd0); lo.append(ground - 0.3); hi.append(ground + 0.3)
    if fit_wall:
        x0.append(e_wall0); lo.append(0.0); hi.append(0.999)  # e_wall
    if fit_mass_ratio:
        # log(m_k / m_0) for k>=1; body 0 is the mass reference (absolute scale free).
        for k in range(1, n_obj):
            x0.append(float(np.log(masses[k] / masses[0])))
            lo.append(float(np.log(1.0 / 50.0)))
            hi.append(float(np.log(50.0)))
    if fit_walls:
        # x-wall surfaces (min, max); bounded near the config/prior to stay sane where
        # a wall is never approached (zero gradient => stays at init).
        x0.append(float(wall_x[0])); lo.append(float(wall_x[0]) - 0.5); hi.append(float(wall_x[0]) + 0.5)
        x0.append(float(wall_x[1])); lo.append(float(wall_x[1]) - 0.5); hi.append(float(wall_x[1]) + 0.5)

    per_body = 7

    def unpack(x: np.ndarray):
        i = n_obj * per_body
        e_pair = float(x[i]); i += 1
        g = float(x[i]) if free_g else g0
        if free_g:
            i += 1
        grd = float(x[i]) if free_grd else grd0
        if free_grd:
            i += 1
        e_wall = float(x[i]) if fit_wall else e_wall0
        if fit_wall:
            i += 1
        masses_eff = list(masses)
        if fit_mass_ratio:
            for k in range(1, n_obj):
                masses_eff[k] = float(masses[0] * np.exp(x[i])); i += 1
        wx = wall_x
        if fit_walls:
            wx = (float(x[i]), float(x[i + 1])); i += 2
        bodies: list[Body] = []
        for k in range(n_obj):
            base = k * per_body
            bodies.append(
                Body(
                    p0=np.array(x[base : base + 3]),
                    v0=np.array(x[base + 3 : base + 6]),
                    radius=radii[k],
                    mass=masses_eff[k],
                    restitution_floor=float(x[base + 6]),
                )
            )
        return bodies, e_pair, g, grd, e_wall, wx

    def resid(x: np.ndarray) -> np.ndarray:
        bodies, e_pair, g, grd, e_wall, wx = unpack(x)
        pred = simulate_scene(
            t_tr, bodies=bodies, gravity_z=g, ground_z=grd,
            restitution_pair=e_pair, substeps=int(args.substeps),
            wall_x=wx, wall_y=wall_y, restitution_wall=e_wall,
        )  # (T, nb, 3)
        res = np.concatenate(
            [(pred[:, k, :] - obs_tr[k]).reshape(-1) for k in range(n_obj)]
        )
        return res

    fit = least_squares(
        resid, x0=np.array(x0), bounds=(np.array(lo), np.array(hi)),
        method="trf", loss="soft_l1", f_scale=0.05, max_nfev=8000,
    )
    bodies, e_pair, g_fit, grd_fit, e_wall_fit, wall_x = unpack(fit.x)
    masses = [bodies[k].mass for k in range(n_obj)]
    pred_tr = simulate_scene(
        t_tr, bodies=bodies, gravity_z=g_fit, ground_z=grd_fit,
        restitution_pair=e_pair, substeps=int(args.substeps),
        wall_x=wall_x, wall_y=wall_y, restitution_wall=e_wall_fit,
    )
    train_mse = float(
        np.mean([np.mean(np.sum((pred_tr[:, k, :] - obs_tr[k]) ** 2, axis=1)) for k in range(n_obj)])
    )

    # --- forward simulate the full available window from train start ---
    full_idx0 = np.array([frame_to_i[0][f] for f in common])
    t_full = times_ref[full_idx0] - times_ref[tr_idx[0][0]]
    pred_full = simulate_scene(
        t_full, bodies=bodies, gravity_z=g_fit, ground_z=grd_fit,
        restitution_pair=e_pair, substeps=int(args.substeps),
        wall_x=wall_x, wall_y=wall_y, restitution_wall=e_wall_fit,
    )
    gt_full = [np.stack([gts[k][f] for f in common], axis=0) for k in range(n_obj)]

    test_mask = np.array([(test_start <= f <= test_end) for f in common])
    per_obj_test_rmse = []
    for k in range(n_obj):
        if test_mask.any():
            err = np.linalg.norm(pred_full[test_mask, k, :] - gt_full[k][test_mask], axis=1)
            per_obj_test_rmse.append(float(np.sqrt(np.mean(err ** 2))))
        else:
            per_obj_test_rmse.append(float("nan"))
    test_rmse = float(np.nanmean(per_obj_test_rmse))

    gt_stack = np.stack(gt_full, axis=1)  # (F, nb, 3)
    gt_collisions = count_pair_collisions(gt_stack, radii)
    pred_collisions = count_pair_collisions(pred_full, radii)

    result = {
        "n_objects": n_obj,
        "similarity": [
            {"scale": s, "scale_inv": 1.0 / s, "R": R.tolist(), "t": tv.tolist()}
            for (s, R, tv) in sims
        ],
        "split": {"canonical": canonical, "train": [train_start, train_end], "test": [test_start, test_end]},
        "fixed": {
            "gravity": not free_g,
            "ground": not free_grd,
            "masses": not fit_mass_ratio,
            "walls_x": not fit_walls,
        },
        "mass_ratio_to_obj0": [float(masses[k] / masses[0]) for k in range(n_obj)],
        "gravity_z_m_s2": g_fit,
        "ground_z_m": grd_fit,
        "restitution_pair": e_pair,
        "restitution_wall": e_wall_fit,
        "walls": {
            "x": list(wall_x) if wall_x is not None else None,
            "y": list(wall_y) if wall_y is not None else None,
        },
        "substeps": int(args.substeps),
        "bodies": [
            {
                "p0_m": bodies[k].p0.tolist(),
                "v0_m_s": bodies[k].v0.tolist(),
                "radius_m": radii[k],
                "mass_kg": masses[k],
                "restitution_floor": bodies[k].restitution_floor,
            }
            for k in range(n_obj)
        ],
        "train_fit_mse_m2": train_mse,
        "heldout_test_rmse_m": test_rmse,
        "heldout_test_rmse_per_object_m": per_obj_test_rmse,
        "collisions": {"gt_full": gt_collisions, "predicted_full": pred_collisions},
        "n_frames": {"train": len(train_f), "test": len(test_f), "full": len(common)},
    }
    (out / "refit_metric.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # --- plot z(t) per object ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, n_obj, figsize=(5.5 * n_obj, 9), sharex=True, squeeze=False)
    names = ["x", "y", "z"]
    for k in range(n_obj):
        for j, nm in enumerate(names):
            ax = axes[j][k]
            ax.plot(t_full, gt_full[k][:, j], "g-", lw=2, label="GT")
            ax.plot(t_full, traj_m[k][full_idx0][:, j], "k.", ms=3, alpha=0.45, label="extracted")
            ax.plot(t_full, pred_full[:, k, j], "r-", lw=1.8, label="fit/predict")
            if test_mask.any():
                ax.axvline(t_full[test_mask.argmax()], color="b", ls="--", alpha=0.5)
            if j == 2:
                ax.axhline(grd_fit, color="gray", ls=":", lw=1)
            if j == 0 and wall_x is not None:
                for w, off in ((wall_x[0], radii[k]), (wall_x[1], -radii[k])):
                    ax.axhline(w + off, color="purple", ls=":", lw=1, alpha=0.6)
            if j == 1 and wall_y is not None:
                for w, off in ((wall_y[0], radii[k]), (wall_y[1], -radii[k])):
                    ax.axhline(w + off, color="purple", ls=":", lw=1, alpha=0.6)
            ax.set_ylabel(f"obj{k} {nm} (m)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=7)
        axes[0][k].set_title(f"obj{k}  e_floor={bodies[k].restitution_floor:.3f}")
    wall_txt = "off" if wall_x is None and wall_y is None else f"e_wall={e_wall_fit:.3f}"
    extra = ""
    if fit_mass_ratio and n_obj >= 2:
        extra += f" m1/m0={masses[1] / masses[0]:.3f}"
    if fit_walls and wall_x is not None:
        extra += f" wallx=[{wall_x[0]:.3f},{wall_x[1]:.3f}]"
    fig.suptitle(
        f"Collision metric-frame refit | e_pair={e_pair:.3f} {wall_txt} g={g_fit:.2f}{extra} "
        f"test RMSE={test_rmse*100:.2f} cm  collisions GT={gt_collisions}/pred={pred_collisions}"
    )
    fig.tight_layout()
    fig.savefig(out / "refit_metric_plot.png", dpi=140)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out/'refit_metric.json'} and {out/'refit_metric_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
