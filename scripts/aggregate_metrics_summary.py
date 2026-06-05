#!/usr/bin/env python3
"""Aggregate METRICS.md metrics from all bounce/collision pipeline outputs."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.poses import load_object_poses_csv  # noqa: E402

_RENDER_NAME_RE = re.compile(r"^(\d+)_cam(\d+)\.png$", re.IGNORECASE)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_m(v: float | None, cm: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if cm:
        return f"{v * 100:.2f} cm"
    return f"{v:.6f} m"


def _fmt_pct(rel: float | None) -> str:
    if rel is None or math.isnan(rel):
        return "n/a"
    return f"{rel * 100:+.1f}%"


def _pass_tag(ok: bool | None) -> str:
    if ok is None:
        return "n/a"
    return "PASS" if ok else "MISS"


def _parse_cfg_args(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    out: dict = {}
    for key in ("iterations", "sh_degree", "coarse_iterations"):
        m = re.search(rf"\b{key}=(\d+)", text)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r"start_checkpoint='([^']*)'", text)
    if m:
        out["start_checkpoint"] = m.group(1)
    m = re.search(r"expname='([^']*)'", text)
    if m:
        out["expname"] = m.group(1)
    return out


def _find_cfg_args(run_dir: Path) -> dict:
    candidates = sorted(run_dir.glob("**/cfg_args"))
    for p in candidates:
        if "_step4a_stage" in str(p) or "_step5_stage" in str(p):
            cfg = _parse_cfg_args(p)
            if cfg:
                return cfg
    for p in candidates:
        cfg = _parse_cfg_args(p)
        if cfg:
            return cfg
    return {}


def _per_frame_errors(predicted_csv: Path, gt_poses_csv: Path) -> dict[str, float] | None:
    if not predicted_csv.is_file() or not gt_poses_csv.is_file():
        return None
    frames: list[int] = []
    pred_xyz: list[list[float]] = []
    with predicted_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(float(row["frame"])))
            pred_xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if not frames:
        return None
    traj = load_object_poses_csv(gt_poses_csv)
    errs = []
    for frame, xyz in zip(frames, pred_xyz):
        gt = traj.by_frame(frame).position
        errs.append(float(np.linalg.norm(np.array(xyz) - gt)))
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "mean_m": float(np.mean(arr)),
        "max_m": float(np.max(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr**2))),
    }


def _per_camera_render(rendered_dir: Path, gt_rgb_root: Path) -> list[dict] | None:
    if not rendered_dir.is_dir() or not gt_rgb_root.is_dir():
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    by_cam: dict[int, list[float]] = {}
    for png in sorted(rendered_dir.glob("*.png")):
        m = _RENDER_NAME_RE.match(png.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        cam_idx = int(m.group(2))
        gt_path = gt_rgb_root / f"cam{cam_idx:02d}" / f"frame{frame_idx:05d}.png"
        if not gt_path.is_file():
            continue
        pred = imageio.imread(png)[..., :3].astype(np.float64) / 255.0
        gt = imageio.imread(gt_path)[..., :3].astype(np.float64) / 255.0
        h, w = min(pred.shape[0], gt.shape[0]), min(pred.shape[1], gt.shape[1])
        pred, gt = pred[:h, :w], gt[:h, :w]
        mse = float(np.mean((pred - gt) ** 2))
        psnr = 99.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
        by_cam.setdefault(cam_idx, []).append(psnr)

    if not by_cam:
        return None
    return [
        {"camera": cam, "psnr_db_mean": float(np.mean(vals)), "n_pairs": len(vals)}
        for cam, vals in sorted(by_cam.items())
    ]


def _infer_scenario(run_name: str, pipeline: str) -> dict[str, str]:
    info: dict[str, str] = {"pipeline": pipeline, "run_name": run_name}
    if pipeline == "bounce":
        if "collision" in run_name:
            info["scenario"] = "collision (misplaced in bounce dir)"
        elif "debug_ball_huge" in run_name:
            info["scenario"] = "debug bounce — large ball (r≈0.35)"
            info["restitution_gt"] = "0.90"
            info["fps"] = "120"
            info["duration"] = "0.5s"
        elif "debug_ball_r0p20" in run_name:
            info["scenario"] = "debug bounce — small ball (r=0.20)"
            info["restitution_gt"] = "0.90"
            info["fps"] = "120"
            info["duration"] = "0.5s"
        elif "scene_0003" in run_name:
            info["scenario"] = "wu75k scene_0003 ball drop"
            info["restitution_gt"] = "0.93"
            info["initial_v"] = "am5p0 (angled)"
            info["fps"] = "120"
        elif "scene_0005" in run_name:
            info["scenario"] = "wu75k scene_0005 ball drop"
            info["restitution_gt"] = "0.93"
            info["initial_v"] = "a5p0"
            info["fps"] = "120"
        elif "ball12" in run_name or "run_50k" in run_name or "wu4dgs" in run_name:
            info["scenario"] = "main ball12 2s blue ball (12-cam grid)"
            info["restitution_gt"] = "0.93"
            info["fps"] = "60 (main) / 120 (eval in some copies)"
        else:
            info["scenario"] = "bounce (see run name)"
    else:
        info["scenario"] = "two-sphere collision (scene_0000)"
        info["scale"] = "1.75"
        info["initial_speed"] = "1.4 m/s"
        info["fps"] = "60"
        info["cameras"] = "side cams subset"
    return info


def _write_refit_physics(lines: list[str], refit: dict, indent: str = "  ") -> None:
    if not refit:
        lines.append(f"{indent}(no refit_metric.json)")
        return

    gt = refit.get("gt") or {}
    params = refit.get("params") or {}
    if "restitution" in params:
        e_fit = params["restitution"]
        e_gt = gt.get("restitution")
        rel = abs(e_fit - e_gt) / e_gt if e_gt else None
        lines.append(f"{indent}restitution e: fitted={e_fit:.4f}  GT={e_gt}  rel_err={_fmt_pct(rel)}")

    if "p0_m" in params:
        lines.append(f"{indent}p0 (m): {[round(x, 4) for x in params['p0_m']]}")
        lines.append(f"{indent}v0 (m/s): {[round(x, 4) for x in params['v0_m_s']]}")
        lines.append(f"{indent}ground_z (m): {params.get('ground_z_m', 'n/a')}")
        lines.append(f"{indent}gravity (m/s²): {params.get('gravity_z_m_s2', 'n/a')}")

    if "restitution_pair" in refit:
        lines.append(f"{indent}restitution pair: fitted={refit['restitution_pair']:.4f}")
        lines.append(f"{indent}restitution wall: fitted={refit.get('restitution_wall', 'n/a')}")
        lines.append(f"{indent}ground_z (m): {refit.get('ground_z_m', 'n/a')}")
        lines.append(f"{indent}gravity (m/s²): {refit.get('gravity_z_m_s2', 'n/a')}")
        lines.append(f"{indent}substeps: {refit.get('substeps', 'n/a')}")
        for i, body in enumerate(refit.get("bodies") or []):
            lines.append(
                f"{indent}obj{i}: r={body.get('radius_m')} m  mass={body.get('mass_kg', 'n/a'):.4f} kg  "
                f"e_floor={body.get('restitution_floor', 'n/a')}"
            )

    train_mse = refit.get("train_fit_mse_m2")
    heldout = refit.get("heldout_test_rmse_m")
    lines.append(f"{indent}train-window fit MSE: {train_mse if train_mse is not None else 'n/a'} m²")
    lines.append(f"{indent}held-out RMSE (refit): {_fmt_m(heldout, cm=True)}")

    bounces = refit.get("bounces")
    if bounces:
        lines.append(
            f"{indent}bounces: GT={bounces.get('gt_full')}  predicted={bounces.get('predicted_full')}  "
            f"extracted={bounces.get('extracted_full')}  train_detected={bounces.get('train_detected')}"
        )
    collisions = refit.get("collisions")
    if collisions:
        lines.append(
            f"{indent}collisions: GT={collisions.get('gt_full')}  predicted={collisions.get('predicted_full')}"
        )

    per_obj = refit.get("heldout_test_rmse_per_object_m")
    if per_obj:
        lines.append(
            f"{indent}held-out RMSE per object: "
            + ", ".join(f"obj{i}={_fmt_m(v, cm=True)}" for i, v in enumerate(per_obj))
        )


def _summarize_bounce_run(run_dir: Path, lines: list[str]) -> None:
    run_name = run_dir.name
    scenario = _infer_scenario(run_name, "bounce")
    cfg = _find_cfg_args(run_dir)
    metrics = _load_json(run_dir / "step6" / "metrics.json")
    refit = _load_json(run_dir / "step4b_metric" / "refit_metric.json")

    lines.append("=" * 88)
    lines.append(f"BOUNCE RUN: {run_name}")
    lines.append("-" * 88)
    lines.append(f"  Scenario: {scenario.get('scenario', 'n/a')}")
    for k, v in scenario.items():
        if k not in ("pipeline", "run_name", "scenario"):
            lines.append(f"  {k}: {v}")

    if cfg:
        iters = cfg.get("iterations")
        sh = cfg.get("sh_degree")
        start_ckpt = cfg.get("start_checkpoint")
        lines.append("  Wu 4DGS hyperparameters:")
        lines.append(f"    iterations: {iters}")
        lines.append(f"    sh_degree: {sh}")
        if start_ckpt:
            lines.append(f"  resumed_from: {Path(str(start_ckpt)).name}")

    name_hints = []
    for token in ("iter50k", "iter75k", "w9", "fg20", "mask1", "den4000", "den1500", "scaleiso"):
        if token in run_name:
            name_hints.append(token)
    if name_hints:
        lines.append(f"  run-name tags: {', '.join(name_hints)}")

    if metrics:
        split = metrics.get("split") or {}
        canon = split.get("canonical") or {}
        traj = metrics.get("trajectory") or {}
        render = metrics.get("render") or {}
        milestone = metrics.get("milestone_pass") or {}

        lines.append("  Train/test split:")
        lines.append(
            f"    train frames {canon.get('train', ['?','?'])} ({canon.get('n_train', '?')} frames)"
        )
        lines.append(
            f"    test  frames {canon.get('test', ['?','?'])} ({canon.get('n_test', '?')} frames)"
        )
        lines.append(f"    eval fps: {metrics.get('fps', 'n/a')}")

        lines.append("  Held-out trajectory metrics (METRICS.md §5.1):")
        lines.append(f"    position RMSE: {_fmt_m(traj.get('pos_rmse_m'), cm=True)}  (target < 5 cm) [{_pass_tag(milestone.get('pos_rmse_lt_0p05m'))}]")
        lines.append(f"    velocity R²:   {traj.get('vel_r2', 'n/a'):.4f}  (target > 0.9) [{_pass_tag(milestone.get('vel_r2_gt_0p9'))}]")
        lines.append(f"    acceleration R²: {traj.get('acc_r2', 'n/a'):.4f}")

        sources = metrics.get("sources") or {}
        pf = _per_frame_errors(
            Path(sources.get("predicted_csv", "")),
            Path(sources.get("gt_poses_csv", "")),
        )
        if pf:
            lines.append(f"    per-frame error: mean {_fmt_m(pf['mean_m'], cm=True)}, max {_fmt_m(pf['max_m'], cm=True)}")

        if render:
            lines.append("  Held-out rendering metrics:")
            lines.append(
                f"    PSNR: {render.get('psnr_db_mean', 'n/a'):.2f} dB  (target > 25) [{_pass_tag(milestone.get('psnr_gt_25db'))}]"
                if render.get("psnr_db_mean") is not None
                else "    PSNR: n/a"
            )
            lines.append(
                f"    SSIM: {render.get('ssim_mean', 'n/a'):.4f}  (target > 0.85) [{_pass_tag(milestone.get('ssim_gt_0p85'))}]"
                if render.get("ssim_mean") is not None
                else "    SSIM: n/a"
            )
            lines.append(f"    MAE:  {render.get('mae_mean', 'n/a')}")
            lines.append(
                f"    matched pairs: {render.get('num_matched')}  missing GT: {render.get('num_missing_gt')}"
            )

            per_cam = _per_camera_render(
                Path(sources.get("rendered_dir", "")),
                Path(sources.get("gt_rgb_root", "")),
            )
            if per_cam:
                cam_str = ", ".join(f"cam{c['camera']}={c['psnr_db_mean']:.1f}dB" for c in per_cam[:6])
                if len(per_cam) > 6:
                    cam_str += f", ... ({len(per_cam)} cams total)"
                lines.append(f"    per-camera PSNR (mean): {cam_str}")
    else:
        lines.append("  (no step6/metrics.json)")

    lines.append("  Physics fit (step4b_metric):")
    _write_refit_physics(lines, refit or {}, indent="    ")

    # Smoothing ablation if present
    refit_w9 = _load_json(run_dir / "step4b_metric_w9" / "refit_metric.json")
    if refit_w9:
        lines.append("  Smoothing ablation (step4b_metric_w9):")
        _write_refit_physics(lines, refit_w9, indent="    ")

    lines.append("")


def _summarize_collision_run(run_dir: Path, lines: list[str]) -> None:
    run_name = run_dir.name
    scenario = _infer_scenario(run_name, "collision")
    cfg0 = _parse_cfg_args(run_dir / "_step4a_stage" / "obj0" / "cfg_args")
    summary = _load_json(run_dir / "step6" / "metrics.json")
    refit = _load_json(run_dir / "step4b_metric" / "refit_metric.json")

    lines.append("=" * 88)
    lines.append(f"COLLISION RUN: {run_name}")
    lines.append("-" * 88)
    lines.append(f"  Scenario: {scenario.get('scenario', 'n/a')}")
    for k, v in scenario.items():
        if k not in ("pipeline", "run_name", "scenario"):
            lines.append(f"  {k}: {v}")

    if cfg0:
        lines.append("  Wu 4DGS hyperparameters (per object, obj0):")
        lines.append(f"    iterations: {cfg0.get('iterations', 'n/a')}")
        lines.append(f"    sh_degree: {cfg0.get('sh_degree', 'n/a')}")

    if summary:
        split = refit.get("split", {}).get("canonical", {}) if refit else {}
        lines.append("  Train/test split:")
        lines.append(
            f"    train frames {split.get('train', ['?','?'])} ({split.get('n_train', '?')} frames)"
        )
        lines.append(
            f"    test  frames {split.get('test', ['?','?'])} ({split.get('n_test', '?')} frames)"
        )

        lines.append("  Held-out trajectory metrics (per object + mean):")
        for obj in summary.get("objects") or []:
            k = obj["object"]
            lines.append(
                f"    obj{k}: pos RMSE {_fmt_m(obj['pos_rmse_m'], cm=True)}  "
                f"vel R²={obj['vel_r2']:.4f}  acc R²={obj['acc_r2']:.4f}"
            )
        lines.append(f"    mean pos RMSE: {_fmt_m(summary.get('mean_pos_rmse_m'), cm=True)}")

        render = summary.get("render")
        if render:
            lines.append("  Held-out rendering metrics (obj0 composites):")
            lines.append(f"    PSNR: {render.get('psnr_db_mean', 'n/a'):.2f} dB")
            lines.append(f"    SSIM: {render.get('ssim_mean', 'n/a'):.4f}")
            lines.append(f"    MAE:  {render.get('mae_mean', 'n/a')}")
            lines.append(f"    matched pairs: {render.get('num_matched')}")

    lines.append("  Physics fit (step4b_metric):")
    _write_refit_physics(lines, refit or {}, indent="    ")
    lines.append("")


def _write_comparison_table(lines: list[str], rows: list[dict]) -> None:
    lines.append("=" * 88)
    lines.append("CROSS-RUN COMPARISON TABLE (held-out position RMSE)")
    lines.append("-" * 88)
    lines.append(f"{'Run':<55} {'RMSE':>10} {'Vel R²':>8} {'PSNR':>8} {'SSIM':>8} {'e err':>8}")
    lines.append("-" * 88)
    for r in rows:
        psnr = f"{r['psnr']:.1f}" if r.get("psnr") is not None else "n/a"
        ssim = f"{r['ssim']:.3f}" if r.get("ssim") is not None else "n/a"
        e_err = f"{r['e_err_pct']:+.1f}%" if r.get("e_err_pct") is not None else "n/a"
        lines.append(
            f"{r['name'][:55]:<55} {r['rmse_cm']:>9.2f}cm {r['vel_r2']:>8.3f} {psnr:>8} {ssim:>8} {e_err:>8}"
        )
    lines.append("")


def main() -> int:
    out_path = _REPO / "outputs" / "METRICS_SUMMARY.txt"
    lines: list[str] = []
    rows: list[dict] = []

    lines.append("PHYS4D PIPELINE METRICS SUMMARY")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Source: METRICS.md definitions (position RMSE, vel/acc R², PSNR/SSIM/MAE, physics recovery)")
    lines.append(f"Repo: {_REPO}")
    lines.append("")
    lines.append("Milestone targets: pos RMSE < 5 cm | vel R² > 0.9 | PSNR > 25 dB | SSIM > 0.85")
    lines.append("Physics fit defaults: scipy least_squares soft_l1, gravity fixed -9.81 m/s², Sav-Gol w9 poly2")
    lines.append("")

    bounce_root = _REPO / "outputs" / "bounce_pipeline"
    collision_root = _REPO / "outputs" / "collision_pipeline"

    bounce_runs = sorted(p for p in bounce_root.iterdir() if p.is_dir() and not p.name.endswith(".log"))
    collision_runs = sorted(p for p in collision_root.iterdir() if p.is_dir())

    lines.append(f"BOUNCE PIPELINE RUNS ({len(bounce_runs)})")
    lines.append("")
    for run_dir in bounce_runs:
        _summarize_bounce_run(run_dir, lines)
        m = _load_json(run_dir / "step6" / "metrics.json")
        r = _load_json(run_dir / "step4b_metric" / "refit_metric.json")
        if m:
            traj = m.get("trajectory") or {}
            render = m.get("render") or {}
            e_fit = (r or {}).get("params", {}).get("restitution")
            e_gt = (r or {}).get("gt", {}).get("restitution")
            e_err = (e_fit - e_gt) / e_gt * 100 if e_fit is not None and e_gt else None
            rows.append(
                {
                    "name": f"bounce/{run_dir.name}",
                    "rmse_cm": (traj.get("pos_rmse_m") or 0) * 100,
                    "vel_r2": traj.get("vel_r2") or 0,
                    "psnr": render.get("psnr_db_mean") if render else None,
                    "ssim": render.get("ssim_mean") if render else None,
                    "e_err_pct": e_err,
                }
            )

    lines.append(f"COLLISION PIPELINE RUNS ({len(collision_runs)})")
    lines.append("")
    for run_dir in collision_runs:
        _summarize_collision_run(run_dir, lines)
        s = _load_json(run_dir / "step6" / "metrics.json")
        if s:
            render = s.get("render") or {}
            rows.append(
                {
                    "name": f"collision/{run_dir.name}",
                    "rmse_cm": (s.get("mean_pos_rmse_m") or 0) * 100,
                    "vel_r2": float(np.mean([o["vel_r2"] for o in s.get("objects", [])])),
                    "psnr": render.get("psnr_db_mean") if render else None,
                    "ssim": render.get("ssim_mean") if render else None,
                    "e_err_pct": None,
                }
            )

    rows.sort(key=lambda r: r["rmse_cm"])
    _write_comparison_table(lines, rows)

    lines.append("NOTES")
    lines.append("-" * 88)
    lines.append("- Velocity R² can be low when x/y centroid wobble dominates finite-difference derivatives.")
    lines.append("- Rendering PSNR/SSIM reflect composite appearance (ball scale/opacity), not trajectory fit.")
    lines.append("- wu75k scenes use 120 fps eval; main ball12/run_50k uses 60 fps in step6 for run_50k_from30k.")
    lines.append("- Baseline comparisons (4DGS-only, const-vel, ballistic) not present in outputs/; see METRICS.md §5.4.")
    lines.append("- run_50k_from30k step4b_metric vs step4b_metric_w9: identical held-out RMSE on this snapshot.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
