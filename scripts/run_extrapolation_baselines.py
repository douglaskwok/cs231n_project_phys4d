#!/usr/bin/env python
"""Compare temporal extrapolation baselines across batch scenes; write baselines table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phys4d import BounceConfig  # noqa: E402
from phys4d.extrapolation_baselines import (  # noqa: E402
    constant_hold_baseline,
    linear_delta_baseline,
    physics_bounce_baseline,
    position_mse_3d,
    predict_physics_z,
)
from phys4d.object_state import stack_state_vectors, trajectory_to_states, window_states  # noqa: E402
from phys4d.poses import load_object_poses_csv  # noqa: E402
from phys4d.visual_dynamics.dataset import discover_batch_scenes, split_scenes  # noqa: E402
from phys4d.visual_dynamics.dynamics_head import VisualDynamicsModel  # noqa: E402


def _scene_bounce_cfg(cfg: dict) -> tuple[BounceConfig, float, int]:
    sim = cfg["simulation"]
    scene = cfg["scene"]
    sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
    ground = next(o for o in scene["objects"] if o["name"] == "ground")
    z0 = float(sphere["initial_position_m"][2])
    bounce_cfg = BounceConfig(
        steps=int(sim["num_frames"]),
        dt=float(sim["dt_s"]),
        z0=z0,
        vz0=float(sphere["initial_velocity_m_s"][2]),
        gravity=float(scene["gravity_m_s2"][2]),
        radius=float(sphere["radius_m"]),
        ground_z=float(ground["height_m"]),
    )
    train_end = int(sim["train_frames"][1])
    return bounce_cfg, z0, train_end


def _eval_learned(
    scene,
    ckpt_path: Path,
    *,
    train_end_frame: int,
    device: torch.device,
) -> dict[str, float]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    history = int(payload["history"])
    model = VisualDynamicsModel(history=history).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    traj = load_object_poses_csv(scene.poses_path)
    states = trajectory_to_states(traj)
    z0 = states[0].position[2]
    z_errors: list[float] = []
    pos_errors: list[float] = []
    with torch.no_grad():
        for frame in range(train_end_frame, len(states) - 1):
            hist = window_states(states, frame, history)
            hist_t = torch.from_numpy(stack_state_vectors(hist, include_velocity=True)).float()
            hist_t = hist_t.unsqueeze(0).to(device)
            from phys4d.visual_dynamics.dataset import VisualDynamicsDataset

            ds = VisualDynamicsDataset([scene], history=history, train_end_frame=train_end_frame)
            crop = ds._load_crop(scene, frame).unsqueeze(0).to(device)
            pred = model.predict_next(hist_t, crop).cpu().numpy()[0]
            target = stack_state_vectors([states[frame + 1]], include_velocity=True)[0]
            pos_errors.append(float(((pred - target) ** 2).mean()))
            z_errors.append(float((pred[2] - target[2]) ** 2))
    return {
        "z_test_mse": float(np.mean(z_errors)) if z_errors else float("nan"),
        "pos_test_mse": float(np.mean(pos_errors)) if pos_errors else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sphere_bounce_batch",
    )
    parser.add_argument(
        "--single-config",
        type=Path,
        default=None,
        help="Also evaluate outputs/sphere_bounce_m2 if batch empty",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "visual_dynamics" / "visual_dynamics.pt",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "extrapolation_baselines.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "outputs" / "extrapolation_baselines.md",
    )
    args = parser.parse_args()

    scenes = discover_batch_scenes(args.batch_root)
    if not scenes and args.single_config:
        cfg_path = args.single_config.resolve()
        from phys4d.visual_dynamics.dataset import SceneRecord

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        parent = cfg_path.parent
        scene_cfg = cfg["scene"]
        sphere = next(o for o in scene_cfg["objects"] if o["name"] == "sphere")
        scenes = [
            SceneRecord(
                scene_id=parent.name,
                config_path=cfg_path,
                poses_path=parent / "object_poses.csv",
                rgb_root=parent / "rgb",
                restitution=float(scene_cfg["true_restitution"]),
                mass_kg=float(sphere["mass_kg"]),
                drop_z_m=float(sphere["initial_position_m"][2]),
            )
        ]

    if not scenes:
        print("No scenes found.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregate: dict[str, list[float]] = {
        "constant_hold": [],
        "linear_delta": [],
        "physics_bounce_fit": [],
        "learned_visual_dynamics_z": [],
        "learned_visual_dynamics_pos": [],
    }
    per_scene: list[dict] = []

    for scene in scenes:
        with scene.config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        bounce_cfg, z0, train_end = _scene_bounce_cfg(cfg)
        traj = load_object_poses_csv(scene.poses_path)
        rows = {
            "scene_id": scene.scene_id,
            "restitution": scene.restitution,
            "mass_kg": scene.mass_kg,
        }
        for fn in (constant_hold_baseline, linear_delta_baseline):
            r = fn(traj, train_end_frame=train_end, z0=z0)
            rows[r.name] = {"z_test_mse": r.z_test_mse, "test_mse": r.test_mse}
            aggregate[r.name].append(r.z_test_mse)
        phys = physics_bounce_baseline(
            traj, bounce_cfg=bounce_cfg, train_end_frame=train_end, z0=z0
        )
        pred_z = predict_physics_z(
            traj, bounce_cfg=bounce_cfg, train_end_frame=train_end, z0=z0
        )
        _, pos_test = position_mse_3d(traj, pred_z, train_end_frame=train_end, z0=z0)
        rows[phys.name] = {"z_test_mse": phys.z_test_mse, "pos_test_mse": pos_test}
        aggregate[phys.name].append(phys.z_test_mse)

        if args.checkpoint.is_file():
            learned = _eval_learned(scene, args.checkpoint, train_end_frame=train_end, device=device)
            rows["learned_visual_dynamics"] = learned
            aggregate["learned_visual_dynamics_z"].append(learned["z_test_mse"])
            aggregate["learned_visual_dynamics_pos"].append(learned["pos_test_mse"])
        per_scene.append(rows)

    summary = {
        name: float(np.mean(vals)) if vals else None
        for name, vals in aggregate.items()
    }
    report = {"per_scene": per_scene, "mean_z_test_mse": summary, "n_scenes": len(scenes)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Temporal extrapolation baselines (z MSE, test frames)",
        "",
        f"Scenes: {len(scenes)} | train frames 0–59 | test 60–89",
        "",
        "| Method | Mean z test MSE |",
        "|--------|-----------------|",
    ]
    for key in ("constant_hold", "linear_delta", "physics_bounce_fit", "learned_visual_dynamics_z"):
        val = summary.get(key)
        lines.append(f"| {key} | {val:.6f} |" if val is not None else f"| {key} | — |")
    lines.append("")
    lines.append(f"Full JSON: `{args.out_json.relative_to(REPO_ROOT)}`")
    lines.append("")
    lines.append("**4DGS note:** vanilla 4DGS reconstructs observed motion; it does not extrapolate ")
    lines.append("to unseen future frames without a separate predictor. Use `export_4dgs_batch.py` + Modal ")
    lines.append("for per-scene reconstruction PSNR; compare render loss after applying predicted poses.")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
