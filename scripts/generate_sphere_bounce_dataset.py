#!/usr/bin/env python
"""Generate multi-view RGB + masks + poses for the M2 sphere-bounce experiment (PyBullet).

Reads paths and parameters from configs/sphere_bounce_m2.json (or --config).
Writes under repo-relative output directories from the config.

Intended Python: the repo's phys_sim venv (PyBullet + imageio).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_pybullet():
    try:
        import pybullet as p  # type: ignore

        return p
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            "PyBullet is not installed in this interpreter. "
            "Activate the repo's phys_sim venv or: pip install pybullet",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _ensure_imageio():
    try:
        import imageio.v2 as imageio  # type: ignore

        return imageio
    except ImportError as exc:  # pragma: no cover
        print("imageio is required for PNG writing: pip install imageio", file=sys.stderr)
        raise SystemExit(1) from exc


def _camera_positions(cfg: dict) -> list[tuple[float, float, float]]:
    cams = cfg["cameras"]
    n = int(cams["num_cameras"])
    r = float(cams["radius_m"])
    hz = float(cams["height_m"])
    positions: list[tuple[float, float, float]] = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        positions.append((r * math.cos(ang), r * math.sin(ang), hz))
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sphere_bounce_m2.json",
        help="Path to experiment JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and exit without simulating",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="Override cameras.num_cameras (ring layout, evenly spaced angles)",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = _load_config(cfg_path)
    if args.num_cameras is not None:
        n = int(args.num_cameras)
        cfg.setdefault("cameras", {})["num_cameras"] = n
        if n >= 10 and "train_cameras" not in cfg["cameras"]:
            cfg["cameras"]["train_cameras"] = list(range(8))
            cfg["cameras"]["test_cameras"] = [8, 9]

    out_rgb = REPO_ROOT / cfg["outputs"]["rgb_frames"]
    out_masks = REPO_ROOT / cfg["outputs"]["masks"]
    out_cams = REPO_ROOT / cfg["outputs"]["camera_poses"]
    out_poses = REPO_ROOT / cfg["outputs"]["object_poses"]
    out_meta = REPO_ROOT / cfg["outputs"]["metadata"]

    if args.dry_run:
        print("config:", cfg_path)
        print("rgb:", out_rgb)
        print("masks:", out_masks)
        print("cameras:", out_cams)
        print("poses:", out_poses)
        print("metadata:", out_meta)
        return 0

    p = _ensure_pybullet()
    imageio = _ensure_imageio()

    import numpy as np

    import pybullet_data

    scene = cfg["scene"]
    sim = cfg["simulation"]
    cams_cfg = cfg["cameras"]

    dt = float(sim["dt_s"])
    num_frames = int(sim["num_frames"])
    w, h = int(cams_cfg["image_size"][0]), int(cams_cfg["image_size"][1])
    look_at = [float(x) for x in cams_cfg["look_at_m"]]
    fov_deg = float(cams_cfg.get("fov_deg", 60.0))
    near = float(cams_cfg.get("near", 0.01))
    far = float(cams_cfg.get("far", 10.0))

    sphere_spec = next(o for o in scene["objects"] if o["name"] == "sphere")
    radius = float(sphere_spec["radius_m"])
    mass = float(sphere_spec["mass_kg"])
    p0 = [float(x) for x in sphere_spec["initial_position_m"]]
    v0 = [float(x) for x in sphere_spec["initial_velocity_m_s"]]
    restitution = float(scene["true_restitution"])
    grav = [float(x) for x in scene["gravity_m_s2"]]

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(grav[0], grav[1], grav[2], physicsClientId=client)
    p.setTimeStep(dt, physicsClientId=client)

    plane_id = p.loadURDF("plane.urdf", [0.0, 0.0, 0.0], useFixedBase=True, physicsClientId=client)
    p.changeDynamics(plane_id, -1, restitution=restitution, physicsClientId=client)

    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[0.85, 0.2, 0.15, 1.0],
        physicsClientId=client,
    )
    sphere_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=p0,
        physicsClientId=client,
    )
    p.resetBaseVelocity(sphere_id, linearVelocity=v0, angularVelocity=[0.0, 0.0, 0.0], physicsClientId=client)
    p.changeDynamics(sphere_id, -1, restitution=restitution, physicsClientId=client)

    cam_positions = _camera_positions(cfg)
    proj = p.computeProjectionMatrixFOV(
        fov=fov_deg,
        aspect=float(w) / float(h),
        nearVal=near,
        farVal=far,
        physicsClientId=client,
    )

    out_rgb.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_cams.parent.mkdir(parents=True, exist_ok=True)
    out_poses.parent.mkdir(parents=True, exist_ok=True)

    camera_records: list[dict] = []
    for ci, eye in enumerate(cam_positions):
        view = p.computeViewMatrix(
            cameraEyePosition=list(eye),
            cameraTargetPosition=look_at,
            cameraUpVector=[0.0, 0.0, 1.0],
            physicsClientId=client,
        )
        camera_records.append(
            {
                "index": ci,
                "eye_m": list(eye),
                "look_at_m": look_at,
                "up": [0.0, 0.0, 1.0],
                "fov_deg": fov_deg,
                "near": near,
                "far": far,
                "image_size": [w, h],
                "view_matrix_row_major": list(view),
                "projection_matrix_row_major": list(proj),
            }
        )

    pose_rows: list[list[float]] = []

    for t in range(num_frames):
        p.stepSimulation(physicsClientId=client)
        pos, orn = p.getBasePositionAndOrientation(sphere_id, physicsClientId=client)
        lin_vel, ang_vel = p.getBaseVelocity(sphere_id, physicsClientId=client)
        pose_rows.append(
            [
                float(t) * dt,
                float(t),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(orn[0]),
                float(orn[1]),
                float(orn[2]),
                float(orn[3]),
                float(lin_vel[0]),
                float(lin_vel[1]),
                float(lin_vel[2]),
                float(ang_vel[0]),
                float(ang_vel[1]),
                float(ang_vel[2]),
            ]
        )

        for ci, eye in enumerate(cam_positions):
            view = p.computeViewMatrix(
                cameraEyePosition=list(eye),
                cameraTargetPosition=look_at,
                cameraUpVector=[0.0, 0.0, 1.0],
                physicsClientId=client,
            )
            img = p.getCameraImage(
                width=w,
                height=h,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=client,
            )
            rgba = np.reshape(img[2], (h, w, 4))[:, :, :3]
            seg = np.reshape(img[4], (h, w))
            mask = (seg == sphere_id).astype(np.uint8) * 255

            cam_dir_rgb = out_rgb / f"cam{ci:02d}"
            cam_dir_mask = out_masks / f"cam{ci:02d}"
            cam_dir_rgb.mkdir(parents=True, exist_ok=True)
            cam_dir_mask.mkdir(parents=True, exist_ok=True)

            frame_name = f"frame{t:05d}.png"
            imageio.imwrite(cam_dir_rgb / frame_name, rgba)
            imageio.imwrite(cam_dir_mask / frame_name, mask)

    p.disconnect(physicsClientId=client)

    with out_cams.open("w", encoding="utf-8") as f:
        json.dump({"cameras": camera_records}, f, indent=2)

    with out_poses.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "frame",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
                "vx_m_s",
                "vy_m_s",
                "vz_m_s",
                "wx_rad_s",
                "wy_rad_s",
                "wz_rad_s",
            ]
        )
        writer.writerows(pose_rows)

    try:
        cfg_rel = str(cfg_path.relative_to(REPO_ROOT))
    except ValueError:
        cfg_rel = str(cfg_path)

    gravity_z = float(grav[2])
    param_names = ["gravity_z", "restitution", "mass_kg", "drop_z_m"]
    param_vector = [gravity_z, restitution, mass, float(p0[2])]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg_rel,
        "experiment": cfg.get("experiment"),
        "num_frames": num_frames,
        "num_cameras": len(cam_positions),
        "dt_s": dt,
        "sphere_body_unique_id_note": "segmentation mask values match PyBullet body unique id for the sphere",
        "physics_params": {
            "param_names": param_names,
            "param_vector": param_vector,
            "predict_v1": ["restitution", "mass_kg", "drop_z_m"],
            "note": "v1 CNN predicts restitution, mass, drop_z; gravity is fixed in sim",
        },
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    physics_params_path = out_meta.parent / "physics_params.json"
    with physics_params_path.open("w", encoding="utf-8") as f:
        json.dump(metadata["physics_params"], f, indent=2)

    print(f"Wrote RGB under: {out_rgb}")
    print(f"Wrote masks under: {out_masks}")
    print(f"Wrote cameras: {out_cams}")
    print(f"Wrote poses: {out_poses}")
    print(f"Wrote metadata: {out_meta}")
    print(f"Wrote physics params: {physics_params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
