"""PyBullet sphere-ground rollout from physics parameters (no rendering)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

POSE_HEADER = [
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


def rollout_sphere_bounce(
    *,
    restitution: float,
    mass_kg: float,
    drop_z_m: float,
    num_frames: int = 90,
    dt_s: float = 1.0 / 60.0,
    radius_m: float = 0.1,
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81),
    initial_xy_m: tuple[float, float] = (0.0, 0.0),
    initial_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    """Simulate vertical bounce; return pose rows matching ``object_poses.csv`` format."""
    import pybullet as p
    import pybullet_data

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(*gravity_m_s2, physicsClientId=client)
    p.setTimeStep(dt_s, physicsClientId=client)

    plane_id = p.loadURDF("plane.urdf", [0.0, 0.0, 0.0], useFixedBase=True, physicsClientId=client)
    p.changeDynamics(plane_id, -1, restitution=restitution, physicsClientId=client)

    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius_m, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius_m,
        rgbaColor=[0.85, 0.2, 0.15, 1.0],
        physicsClientId=client,
    )
    p0 = [float(initial_xy_m[0]), float(initial_xy_m[1]), float(drop_z_m)]
    sphere_id = p.createMultiBody(
        baseMass=mass_kg,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=p0,
        physicsClientId=client,
    )
    p.resetBaseVelocity(
        sphere_id,
        linearVelocity=list(initial_velocity_m_s),
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=client,
    )
    p.changeDynamics(sphere_id, -1, restitution=restitution, physicsClientId=client)

    rows: list[list[float]] = []
    for t in range(num_frames):
        p.stepSimulation(physicsClientId=client)
        pos, orn = p.getBasePositionAndOrientation(sphere_id, physicsClientId=client)
        lin_vel, ang_vel = p.getBaseVelocity(sphere_id, physicsClientId=client)
        rows.append(
            [
                float(t) * dt_s,
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
    p.disconnect(physicsClientId=client)
    return rows


def write_pose_csv(rows: list[list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(POSE_HEADER)
        writer.writerows(rows)


def rollout_from_config(cfg: dict, params: dict[str, float]) -> list[list[float]]:
    """Rollout using M2 config scene block + predicted ``restitution``, ``mass_kg``, ``drop_z_m``."""
    scene = cfg["scene"]
    sim = cfg["simulation"]
    sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
    p0 = sphere["initial_position_m"]
    v0 = sphere["initial_velocity_m_s"]
    grav = scene["gravity_m_s2"]
    return rollout_sphere_bounce(
        restitution=float(params["restitution"]),
        mass_kg=float(params["mass_kg"]),
        drop_z_m=float(params["drop_z_m"]),
        num_frames=int(sim["num_frames"]),
        dt_s=float(sim["dt_s"]),
        radius_m=float(sphere["radius_m"]),
        gravity_m_s2=(float(grav[0]), float(grav[1]), float(grav[2])),
        initial_xy_m=(float(p0[0]), float(p0[1])),
        initial_velocity_m_s=(float(v0[0]), float(v0[1]), float(v0[2])),
    )
