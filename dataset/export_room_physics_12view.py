#!/usr/bin/env python
"""Generate 12-view room datasets for non-bounce physics scenarios.

This is the companion exporter to ``export_ping_pong_12view.py``. It keeps the
same camera rig, room setting, and dataset tree shape, but generates three
additional PyBullet scenarios:

* collision: two sliding boxes collide on the table
* stacking: blocks are released sequentially into a small stack
* deformable: a soft torus drops onto the table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.export_ping_pong_12view import (  # noqa: E402
    CAMERA_NAMES,
    DATASET_OUTPUTS_ROOT,
    GRAVITY,
    HEIGHT,
    ROOM_FLOOR_Z,
    TABLE_LENGTH,
    TABLE_THICKNESS,
    TABLE_TOP_Z,
    TABLE_WIDTH,
    VIDEO_FPS,
    WIDTH,
    _camera_records,
    _create_room_geometry,
    _ensure_pybullet,
    _frame_split_ranges,
    _open_mp4_writer,
    _parse_video_camera_names,
    _repo_path,
    _steps_per_frame,
)

DEFAULT_OUTPUT_DIR = DATASET_OUTPUTS_ROOT / "room_physics_12view"
BODY_UNIQUE_ID_MASK = (1 << 24) - 1

SCENARIO_DEFAULTS = {
    "collision": {"duration_sec": 2.6, "sim_hz": 480.0},
    "stacking": {"duration_sec": 5.0, "sim_hz": 480.0},
    "deformable": {"duration_sec": 2.6, "sim_hz": 480.0},
}

COLLISION_TABLE_TOP_Z = 0.35
COLLISION_OBJECT_HALF_EXTENTS_M = [0.120, 0.120, 0.050]
STACKING_NUM_BLOCKS = 3
STACKING_BLOCK_SIZE_M = [0.120, 0.120, 0.070]
STACKING_BLOCK_MASS_KG = 0.180
STACKING_RELEASE_INTERVAL_S = 0.80


@dataclass
class BodyInfo:
    name: str
    body_id: int
    mass_kg: float
    kind: str
    active: bool = True


def _default_camera_rows() -> list[dict]:
    """Copy of the room 12-view rig, with a slightly roomier FOV for blocks."""

    rows = []
    ring_radius = 1.75
    for i, name in enumerate(CAMERA_NAMES[:8]):
        angle = -math.pi / 2.0 + i * (2.0 * math.pi / 8.0)
        rows.append(
            {
                "name": name,
                "eye_x": str(ring_radius * math.cos(angle)),
                "eye_y": str(ring_radius * math.sin(angle)),
                "eye_z": "1.15",
                "up_x": "0",
                "up_y": "0",
                "up_z": "1",
                "fov_deg": "48",
            }
        )
    rows.extend(
        [
            {
                "name": "top",
                "eye_x": "0.0",
                "eye_y": "0.0",
                "eye_z": "2.65",
                "up_x": "0",
                "up_y": "1",
                "up_z": "0",
                "fov_deg": "42",
            },
            {
                "name": "top_oblique",
                "eye_x": "0.75",
                "eye_y": "-0.75",
                "eye_z": "2.35",
                "up_x": "0",
                "up_y": "0",
                "up_z": "1",
                "fov_deg": "46",
            },
            {
                "name": "low_front_left",
                "eye_x": "-1.15",
                "eye_y": "-1.65",
                "eye_z": "0.62",
                "up_x": "0",
                "up_y": "0",
                "up_z": "1",
                "fov_deg": "58",
            },
            {
                "name": "low_back_right",
                "eye_x": "1.15",
                "eye_y": "1.65",
                "eye_z": "0.62",
                "up_x": "0",
                "up_y": "0",
                "up_z": "1",
                "fov_deg": "58",
            },
        ]
    )
    return rows


def _init_pybullet(*, deformable: bool, sim_hz: float, solver_iterations: int) -> tuple[Any, int]:
    p = _ensure_pybullet()
    import pybullet_data

    client = p.connect(p.DIRECT)
    if deformable:
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=client)
    else:
        p.resetSimulation(physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0.0, 0.0, GRAVITY, physicsClientId=client)
    p.setTimeStep(1.0 / sim_hz, physicsClientId=client)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / sim_hz,
        numSolverIterations=solver_iterations,
        numSubSteps=2 if not deformable else 1,
        deterministicOverlappingPairs=1,
        restitutionVelocityThreshold=0.0,
        physicsClientId=client,
    )
    return p, client


def _create_table(
    p,
    client: int,
    *,
    restitution: float,
    lateral_friction: float,
    table_top_z: float = TABLE_TOP_Z,
    rgba_color: list[float] | None = None,
) -> int:
    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_LENGTH / 2.0, TABLE_WIDTH / 2.0, TABLE_THICKNESS / 2.0],
        physicsClientId=client,
    )
    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_LENGTH / 2.0, TABLE_WIDTH / 2.0, TABLE_THICKNESS / 2.0],
        rgbaColor=rgba_color or [0.14, 0.40, 0.28, 1.0],
        physicsClientId=client,
    )
    table_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[0.0, 0.0, table_top_z - TABLE_THICKNESS / 2.0],
        physicsClientId=client,
    )
    p.changeDynamics(
        table_id,
        -1,
        restitution=restitution,
        lateralFriction=lateral_friction,
        spinningFriction=0.01,
        rollingFriction=0.01,
        contactProcessingThreshold=0.0,
        physicsClientId=client,
    )
    return table_id


def _seg_body_mask(seg: np.ndarray, body_id: int) -> np.ndarray:
    ids = np.bitwise_and(seg.astype(np.int64), BODY_UNIQUE_ID_MASK)
    return ids == int(body_id)


def _blue_soft_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (b > 75) & (b > r + 25) & (b > g + 5) & (g > 35)


def _write_mask(path: Path, mask: np.ndarray, imageio_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio_module.imwrite(path, mask.astype(np.uint8) * 255)


def _write_rgb(path: Path, rgb: np.ndarray, imageio_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio_module.imwrite(path, rgb.astype(np.uint8))


def _write_pose_rows(rows: list[dict], path: Path) -> None:
    fields = [
        "time_s",
        "frame",
        "object_name",
        "object_index",
        "active",
        "kind",
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
        "mesh_z_min_m",
        "mesh_z_max_m",
        "mesh_z_span_m",
        "mesh_xy_radius_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _add_collision_object(
    p,
    client: int,
    *,
    name: str,
    mass_kg: float,
    half_extents: list[float],
    position: list[float],
    velocity: list[float],
    rgba_color: list[float],
    restitution: float = 0.80,
) -> BodyInfo:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgba_color,
        physicsClientId=client,
    )
    body = p.createMultiBody(
        mass_kg,
        col,
        vis,
        position,
        physicsClientId=client,
    )
    p.changeDynamics(
        body,
        -1,
        lateralFriction=0.02,
        restitution=restitution,
        rollingFriction=0.001,
        spinningFriction=0.001,
        contactProcessingThreshold=0.0,
        physicsClientId=client,
    )
    p.resetBaseVelocity(body, linearVelocity=velocity, physicsClientId=client)
    return BodyInfo(name=name, body_id=body, mass_kg=mass_kg, kind="rigid_box")


def _setup_collision_scene(
    p,
    client: int,
    *,
    mass_a_kg: float = 0.220,
    velocity_scale: float = 1.0,
    restitution: float = 0.80,
) -> tuple[int, list[BodyInfo], list[str], dict]:
    table_top_z = COLLISION_TABLE_TOP_Z
    table_id = _create_table(
        p,
        client,
        restitution=0.10,
        lateral_friction=0.02,
        table_top_z=table_top_z,
    )
    half = COLLISION_OBJECT_HALF_EXTENTS_M
    vel_a = 0.95 * velocity_scale
    vel_b = 0.75 * velocity_scale
    objects = [
        _add_collision_object(
            p,
            client,
            name="object_a",
            mass_kg=mass_a_kg,
            half_extents=half,
            position=[-0.38, 0.0, table_top_z + half[2]],
            velocity=[vel_a, 0.0, 0.0],
            rgba_color=[0.90, 0.20, 0.16, 1.0],
            restitution=restitution,
        ),
        _add_collision_object(
            p,
            client,
            name="object_b",
            mass_kg=0.160,
            half_extents=half,
            position=[0.38, 0.0, table_top_z + half[2]],
            velocity=[-vel_b, 0.0, 0.0],
            rgba_color=[0.12, 0.50, 0.92, 1.0],
            restitution=restitution,
        ),
    ]
    meta = {
        "scenario": "collision",
        "object_half_extents_m": half,
        "object_full_size_m": [2.0 * v for v in half],
        "object_a_mass_kg": mass_a_kg,
        "object_b_mass_kg": 0.160,
        "object_restitution": restitution,
        "object_lateral_friction": 0.02,
        "velocity_scale": velocity_scale,
        "object_a_initial_velocity_m_s": vel_a,
        "object_b_initial_velocity_m_s": vel_b,
        "table_top_z_m": table_top_z,
    }
    return table_id, objects, [obj.name for obj in objects], meta


def _stack_block_pose(index: int) -> tuple[list[float], list[float]]:
    p = _ensure_pybullet()
    offsets = [
        [0.000, 0.000],
        [0.014, -0.008],
        [-0.012, 0.010],
    ]
    yaws = [0.0, 0.10, -0.08]
    x, y = offsets[index]
    # Drop blocks from slightly above the existing stack. The release interval
    # produces contact events without artificial constraints.
    z = TABLE_TOP_Z + 0.42 + index * 0.05
    return [x, y, z], p.getQuaternionFromEuler([0.0, 0.0, yaws[index]])


def _create_stack_block(p, client: int, index: int) -> BodyInfo:
    half = [v / 2.0 for v in STACKING_BLOCK_SIZE_M]
    colors = [
        [0.88, 0.15, 0.12, 1.0],
        [0.10, 0.55, 0.85, 1.0],
        [0.12, 0.70, 0.25, 1.0],
    ]
    pos, orn = _stack_block_pose(index)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=colors[index],
        physicsClientId=client,
    )
    body = p.createMultiBody(
        baseMass=STACKING_BLOCK_MASS_KG,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn,
        physicsClientId=client,
    )
    p.changeDynamics(
        body,
        -1,
        lateralFriction=0.85,
        restitution=0.05,
        spinningFriction=0.01,
        rollingFriction=0.01,
        contactProcessingThreshold=0.0,
        physicsClientId=client,
    )
    return BodyInfo(
        name=f"block_{index:02d}",
        body_id=body,
        mass_kg=STACKING_BLOCK_MASS_KG,
        kind="rigid_box",
    )


def _setup_stacking_scene(p, client: int) -> tuple[int, list[BodyInfo], list[str], dict]:
    table_id = _create_table(p, client, restitution=0.08, lateral_friction=0.82)
    expected = [f"block_{idx:02d}" for idx in range(STACKING_NUM_BLOCKS)]
    meta = {
        "scenario": "stacking",
        "num_blocks": STACKING_NUM_BLOCKS,
        "block_size_m": STACKING_BLOCK_SIZE_M,
        "block_mass_kg": STACKING_BLOCK_MASS_KG,
        "release_interval_s": STACKING_RELEASE_INTERVAL_S,
        "block_lateral_friction": 0.85,
        "block_restitution": 0.05,
    }
    return table_id, [], expected, meta


def _setup_deformable_scene(p, client: int) -> tuple[int, list[BodyInfo], list[str], dict]:
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.03, physicsClientId=client)
    table_id = _create_table(p, client, restitution=0.05, lateral_friction=0.80)
    soft_id = p.loadSoftBody(
        "torus/torus_textured.obj",
        basePosition=[0.0, 0.0, TABLE_TOP_Z + 0.50],
        scale=0.22,
        mass=0.18,
        useNeoHookean=0,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=10,
        springDampingStiffness=0.25,
        springDampingAllDirections=1,
        useSelfCollision=0,
        frictionCoeff=0.90,
        collisionMargin=0.006,
        physicsClientId=client,
    )
    p.changeVisualShape(
        soft_id,
        -1,
        rgbaColor=[0.10, 0.52, 0.88, 1.0],
        physicsClientId=client,
    )
    obj = BodyInfo(name="soft_torus", body_id=soft_id, mass_kg=0.18, kind="soft_body")
    meta = {
        "scenario": "deformable",
        "soft_body_asset": "torus/torus_textured.obj",
        "soft_mass_kg": 0.18,
        "soft_scale": 0.22,
        "spring_elastic_stiffness": 10,
        "spring_damping_stiffness": 0.25,
        "soft_lateral_friction": 0.90,
        "collision_margin_m": 0.006,
    }
    return table_id, [obj], [obj.name], meta


def _soft_mesh_stats(p, client: int, body_id: int) -> dict[str, float]:
    _, vertices = p.getMeshData(
        body_id,
        flags=p.MESH_DATA_SIMULATION_MESH,
        physicsClientId=client,
    )
    verts = np.asarray(vertices, dtype=np.float64)
    center = verts.mean(axis=0)
    xy_radius = float(np.linalg.norm(verts[:, :2] - center[:2], axis=1).max())
    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    return {
        "x_m": float(center[0]),
        "y_m": float(center[1]),
        "z_m": float(center[2]),
        "mesh_z_min_m": z_min,
        "mesh_z_max_m": z_max,
        "mesh_z_span_m": z_max - z_min,
        "mesh_xy_radius_m": xy_radius,
    }


def _append_pose_rows(
    *,
    p,
    client: int,
    frame_idx: int,
    time_s: float,
    bodies: list[BodyInfo],
    pose_rows: list[dict],
) -> None:
    for object_index, info in enumerate(bodies):
        base = {
            "time_s": time_s,
            "frame": frame_idx,
            "object_name": info.name,
            "object_index": object_index,
            "active": int(info.active),
            "kind": info.kind,
            "mesh_z_min_m": "",
            "mesh_z_max_m": "",
            "mesh_z_span_m": "",
            "mesh_xy_radius_m": "",
        }
        if info.kind == "soft_body":
            stats = _soft_mesh_stats(p, client, info.body_id)
            pose_rows.append(
                {
                    **base,
                    **stats,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                    "vx_m_s": 0.0,
                    "vy_m_s": 0.0,
                    "vz_m_s": 0.0,
                    "wx_rad_s": 0.0,
                    "wy_rad_s": 0.0,
                    "wz_rad_s": 0.0,
                }
            )
        else:
            pos, orn = p.getBasePositionAndOrientation(info.body_id, physicsClientId=client)
            lin_vel, ang_vel = p.getBaseVelocity(info.body_id, physicsClientId=client)
            pose_rows.append(
                {
                    **base,
                    "x_m": float(pos[0]),
                    "y_m": float(pos[1]),
                    "z_m": float(pos[2]),
                    "qx": float(orn[0]),
                    "qy": float(orn[1]),
                    "qz": float(orn[2]),
                    "qw": float(orn[3]),
                    "vx_m_s": float(lin_vel[0]),
                    "vy_m_s": float(lin_vel[1]),
                    "vz_m_s": float(lin_vel[2]),
                    "wx_rad_s": float(ang_vel[0]),
                    "wy_rad_s": float(ang_vel[1]),
                    "wz_rad_s": float(ang_vel[2]),
                }
            )


def _open_video_writers(
    *,
    imageio_module,
    scene_dir: Path,
    camera_records: list[dict],
    video_fps: float,
    video_camera_names: set[str] | None,
    overwrite: bool,
) -> dict[int, tuple[object | None, object | None, object | None, object | None]]:
    writers = {}
    for cam in camera_records:
        cam_idx = int(cam["index"])
        name = str(cam["name"])
        if video_camera_names is not None and name not in video_camera_names:
            continue

        paths = [
            scene_dir / "videos" / "rgb" / f"cam{cam_idx:02d}_{name}.mp4",
            scene_dir / "videos" / "masks" / f"cam{cam_idx:02d}_{name}.mp4",
            scene_dir / "videos" / "masks_table" / f"cam{cam_idx:02d}_{name}.mp4",
            scene_dir / "videos" / "masks_object_table" / f"cam{cam_idx:02d}_{name}.mp4",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        opened = []
        for path in paths:
            opened.append(
                _open_mp4_writer(imageio_module, path, fps=video_fps)
                if overwrite or not path.exists()
                else None
            )
        writers[cam_idx] = tuple(opened)  # type: ignore[assignment]
    return writers


def _open_per_object_mask_video_writers(
    *,
    imageio_module,
    scene_dir: Path,
    camera_records: list[dict],
    video_fps: float,
    video_camera_names: set[str] | None,
    object_names: list[str],
    overwrite: bool,
) -> dict[int, dict[str, object | None]]:
    writers: dict[int, dict[str, object | None]] = {}
    for cam in camera_records:
        cam_idx = int(cam["index"])
        cam_name = str(cam["name"])
        if video_camera_names is not None and cam_name not in video_camera_names:
            continue

        per_object: dict[str, object | None] = {}
        for object_name in object_names:
            path = scene_dir / "videos" / f"masks_{object_name}" / f"cam{cam_idx:02d}_{cam_name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            per_object[object_name] = (
                _open_mp4_writer(imageio_module, path, fps=video_fps)
                if overwrite or not path.exists()
                else None
            )
        writers[cam_idx] = per_object
    return writers


def _scenario_setup(
    p,
    client: int,
    scenario: str,
    *,
    collision_mass_a_kg: float = 0.220,
    collision_velocity_scale: float = 1.0,
    collision_restitution: float = 0.80,
) -> tuple[int, list[BodyInfo], list[str], dict]:
    if scenario == "collision":
        return _setup_collision_scene(
            p, client,
            mass_a_kg=collision_mass_a_kg,
            velocity_scale=collision_velocity_scale,
            restitution=collision_restitution,
        )
    if scenario == "stacking":
        return _setup_stacking_scene(p, client)
    if scenario == "deformable":
        return _setup_deformable_scene(p, client)
    raise ValueError(f"Unknown scenario: {scenario}")


def simulate_scenario(
    *,
    scenario: str,
    scene_dir: Path,
    video_fps: float,
    sim_hz: float,
    duration_sec: float,
    max_frames: int | None,
    render_camera_names: set[str] | None,
    video_camera_names: set[str] | None,
    overwrite: bool,
    write_videos: bool,
    collision_mass_a_kg: float = 0.220,
    collision_velocity_scale: float = 1.0,
    collision_restitution: float = 0.80,
) -> dict:
    import imageio.v2 as imageio

    deformable = scenario == "deformable"
    p, client = _init_pybullet(
        deformable=deformable,
        sim_hz=sim_hz,
        solver_iterations=120 if deformable else 180,
    )
    steps_per_frame = _steps_per_frame(video_fps, sim_hz)
    environment_info = _create_room_geometry(p, client)
    table_id, bodies, expected_names, scenario_meta = _scenario_setup(
        p, client, scenario,
        collision_mass_a_kg=collision_mass_a_kg,
        collision_velocity_scale=collision_velocity_scale,
        collision_restitution=collision_restitution,
    )

    camera_rows = _default_camera_rows()
    if render_camera_names is not None:
        camera_rows = [row for row in camera_rows if row["name"] in render_camera_names]
        if not camera_rows:
            raise ValueError(f"No cameras selected from: {sorted(render_camera_names)}")
    camera_records = _camera_records(camera_rows)
    camera_names = [str(cam["name"]) for cam in camera_records]
    camera_layout = (
        "12view_notebook_named_rig"
        if len(camera_records) == len(CAMERA_NAMES)
        else "selected_from_12view_notebook_named_rig"
    )
    train_cameras = list(range(min(10, len(camera_records))))
    test_cameras = [idx for idx in range(len(camera_records)) if idx not in train_cameras]

    scene_dir.mkdir(parents=True, exist_ok=True)
    with (scene_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump({"cameras": camera_records}, f, indent=2)

    roots = {
        "rgb": scene_dir / "rgb",
        "masks": scene_dir / "masks",
        "masks_table": scene_dir / "masks_table",
        "masks_object_table": scene_dir / "masks_object_table",
    }
    object_roots = {
        name: scene_dir / f"masks_{name}"
        for name in expected_names
    }
    for root in [*roots.values(), *object_roots.values()]:
        root.mkdir(parents=True, exist_ok=True)

    num_frames = int(duration_sec * video_fps) + 1
    if max_frames is not None:
        num_frames = min(num_frames, int(max_frames))
    # For these multi-event scenarios, the complete timeline is usually the
    # training target. Test cameras still provide held-out viewpoints.
    train_frames = [0, max(0, num_frames - 1)]
    test_frames = [0, max(0, num_frames - 1)]
    short_train_frames, short_test_frames = _frame_split_ranges(num_frames, video_fps)

    video_writers = (
        _open_video_writers(
            imageio_module=imageio,
            scene_dir=scene_dir,
            camera_records=camera_records,
            video_fps=video_fps,
            video_camera_names=video_camera_names,
            overwrite=overwrite,
        )
        if write_videos
        else {}
    )
    per_object_video_writers = (
        _open_per_object_mask_video_writers(
            imageio_module=imageio,
            scene_dir=scene_dir,
            camera_records=camera_records,
            video_fps=video_fps,
            video_camera_names=video_camera_names,
            object_names=expected_names,
            overwrite=overwrite,
        )
        if write_videos
        else {}
    )

    pose_rows: list[dict] = []
    empty_masks = {"objects": 0, "table": 0, "object_table": 0}
    internal_step = 0
    next_stack_release_step = 0
    release_interval_steps = int(round(STACKING_RELEASE_INTERVAL_S * sim_hz))

    def maybe_release_stack() -> None:
        nonlocal next_stack_release_step
        if scenario != "stacking":
            return
        if len(bodies) >= len(expected_names):
            return
        if internal_step >= next_stack_release_step:
            bodies.append(_create_stack_block(p, client, len(bodies)))
            next_stack_release_step += release_interval_steps

    try:
        maybe_release_stack()
        for frame_idx in range(num_frames):
            if frame_idx == 0 or frame_idx % 30 == 0 or frame_idx == num_frames - 1:
                print(
                    f"  {scenario}: frame {frame_idx + 1}/{num_frames}",
                    flush=True,
                )

            time_s = frame_idx / video_fps
            _append_pose_rows(
                p=p,
                client=client,
                frame_idx=frame_idx,
                time_s=time_s,
                bodies=bodies,
                pose_rows=pose_rows,
            )

            bodies_by_name = {body.name: body for body in bodies}
            for cam in camera_records:
                cam_idx = int(cam["index"])
                frame_name = f"frame{frame_idx:05d}.png"
                img = p.getCameraImage(
                    width=WIDTH,
                    height=HEIGHT,
                    viewMatrix=cam["view_matrix_row_major"],
                    projectionMatrix=cam["projection_matrix_row_major"],
                    renderer=p.ER_TINY_RENDERER,
                    lightDirection=[-0.5, -0.4, -1.0],
                    physicsClientId=client,
                )
                rgb = np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)
                seg = np.reshape(img[4], (HEIGHT, WIDTH))

                object_masks: dict[str, np.ndarray] = {}
                for name in expected_names:
                    info = bodies_by_name.get(name)
                    if info is None:
                        object_masks[name] = np.zeros((HEIGHT, WIDTH), dtype=bool)
                        continue
                    mask = _seg_body_mask(seg, info.body_id)
                    if info.kind == "soft_body" and not mask.any():
                        mask = _blue_soft_mask(rgb)
                    object_masks[name] = mask

                objects_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
                for mask in object_masks.values():
                    objects_mask |= mask
                table_mask = _seg_body_mask(seg, table_id)
                object_table_mask = objects_mask | table_mask

                if not objects_mask.any():
                    empty_masks["objects"] += 1
                if not table_mask.any():
                    empty_masks["table"] += 1
                if not object_table_mask.any():
                    empty_masks["object_table"] += 1

                if overwrite or not (roots["rgb"] / f"cam{cam_idx:02d}" / frame_name).exists():
                    _write_rgb(roots["rgb"] / f"cam{cam_idx:02d}" / frame_name, rgb, imageio)
                _write_mask(roots["masks"] / f"cam{cam_idx:02d}" / frame_name, objects_mask, imageio)
                _write_mask(roots["masks_table"] / f"cam{cam_idx:02d}" / frame_name, table_mask, imageio)
                _write_mask(
                    roots["masks_object_table"] / f"cam{cam_idx:02d}" / frame_name,
                    object_table_mask,
                    imageio,
                )
                for name, mask in object_masks.items():
                    _write_mask(object_roots[name] / f"cam{cam_idx:02d}" / frame_name, mask, imageio)

                if cam_idx in video_writers:
                    rgb_writer, mask_writer, table_writer, object_table_writer = video_writers[cam_idx]
                    if rgb_writer is not None:
                        rgb_writer.append_data(rgb)
                    if mask_writer is not None:
                        mask_writer.append_data(np.repeat(objects_mask[..., None], 3, axis=2).astype(np.uint8) * 255)
                    if table_writer is not None:
                        table_writer.append_data(np.repeat(table_mask[..., None], 3, axis=2).astype(np.uint8) * 255)
                    if object_table_writer is not None:
                        object_table_writer.append_data(
                            np.repeat(object_table_mask[..., None], 3, axis=2).astype(np.uint8) * 255
                        )
                if cam_idx in per_object_video_writers:
                    for name, writer in per_object_video_writers[cam_idx].items():
                        if writer is not None:
                            writer.append_data(
                                np.repeat(object_masks[name][..., None], 3, axis=2).astype(np.uint8) * 255
                            )

            for _ in range(steps_per_frame):
                internal_step += 1
                maybe_release_stack()
                p.stepSimulation(physicsClientId=client)
    finally:
        for writers in video_writers.values():
            for writer in writers:
                if writer is not None:
                    writer.close()
        for writers in per_object_video_writers.values():
            for writer in writers.values():
                if writer is not None:
                    writer.close()
        p.disconnect(physicsClientId=client)

    _write_pose_rows(pose_rows, scene_dir / "object_poses.csv")

    table_top_z = float(scenario_meta.get("table_top_z_m", TABLE_TOP_Z))
    config = {
        "experiment": f"{scenario}_12view_room",
        "description": f"Generated 12-view room {scenario} scene.",
        "scene": {
            "scenario": scenario,
            "environment": environment_info,
            "objects": [
                {
                    "name": name,
                    "mask": _repo_path(object_roots[name]),
                }
                for name in expected_names
            ],
            "table": {
                "name": "table",
                "shape": "box",
                "top_center_m": [0.0, 0.0, table_top_z],
                "size_m": [TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS],
            },
            "scenario_params": scenario_meta,
        },
        "simulation": {
            "physics_engine": "pybullet",
            "dt_s": 1.0 / video_fps,
            "internal_dt_s": 1.0 / sim_hz,
            "steps_per_frame": steps_per_frame,
            "num_frames": num_frames,
            "duration_s": (num_frames - 1) / video_fps if num_frames else 0.0,
            "gravity_m_s2": [0.0, 0.0, GRAVITY],
            "train_frames": train_frames,
            "test_frames": test_frames,
            "short_split_train_frames": short_train_frames,
            "short_split_test_frames": short_test_frames,
        },
        "cameras": {
            "num_cameras": len(camera_records),
            "camera_names": camera_names,
            "layout": camera_layout,
            "train_cameras": train_cameras,
            "test_cameras": test_cameras,
            "image_size": [WIDTH, HEIGHT],
        },
        "outputs": {
            "rgb_frames": _repo_path(roots["rgb"]),
            "masks": _repo_path(roots["masks"]),
            "table_masks": _repo_path(roots["masks_table"]),
            "object_table_masks": _repo_path(roots["masks_object_table"]),
            "object_masks": {name: _repo_path(path) for name, path in object_roots.items()},
            "rgb_videos": _repo_path(scene_dir / "videos" / "rgb"),
            "mask_videos": _repo_path(scene_dir / "videos" / "masks"),
            "table_mask_videos": _repo_path(scene_dir / "videos" / "masks_table"),
            "object_table_mask_videos": _repo_path(scene_dir / "videos" / "masks_object_table"),
            "object_mask_videos": {
                name: _repo_path(scene_dir / "videos" / f"masks_{name}")
                for name in expected_names
            },
            "camera_poses": _repo_path(scene_dir / "cameras.json"),
            "object_poses": _repo_path(scene_dir / "object_poses.csv"),
            "metadata": _repo_path(scene_dir / "metadata.json"),
        },
    }
    with (scene_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "dataset/export_room_physics_12view.py",
        "output_dir": _repo_path(scene_dir),
        "scenario": scenario,
        "scenario_params": scenario_meta,
        "num_frames": num_frames,
        "num_cameras": len(camera_records),
        "camera_names": camera_names,
        "video_fps": video_fps,
        "sim_hz": sim_hz,
        "steps_per_frame": steps_per_frame,
        "duration_s": (num_frames - 1) / video_fps if num_frames else 0.0,
        "environment": environment_info,
        "empty_masks": empty_masks,
        "mask_outputs": {
            "objects": _repo_path(roots["masks"]),
            "table": _repo_path(roots["masks_table"]),
            "objects_table": _repo_path(roots["masks_object_table"]),
            "per_object": {name: _repo_path(path) for name, path in object_roots.items()},
        },
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def generate_dataset(
    *,
    scenarios: list[str],
    output_dir: Path,
    video_fps: float,
    sim_hz: float | None,
    duration_sec: float | None,
    max_frames: int | None,
    render_camera_names: set[str] | None,
    video_camera_names: set[str] | None,
    overwrite: bool,
    write_videos: bool,
    dry_run: bool,
) -> dict:
    planned = []
    for idx, scenario in enumerate(scenarios):
        scenario_sim_hz = float(sim_hz or SCENARIO_DEFAULTS[scenario]["sim_hz"])
        scenario_duration_sec = float(duration_sec or SCENARIO_DEFAULTS[scenario]["duration_sec"])
        scene_id = f"scene_{idx:04d}_{scenario}_room"
        planned.append(
            {
                "scene_id": scene_id,
                "scenario": scenario,
                "path": _repo_path(output_dir / scene_id),
                "video_fps": video_fps,
                "sim_hz": scenario_sim_hz,
                "duration_sec": scenario_duration_sec,
                "steps_per_frame": _steps_per_frame(video_fps, scenario_sim_hz),
            }
        )

    if dry_run:
        return {
            "experiment": "room_physics_12view",
            "output_dir": str(output_dir),
            "num_scenes": len(planned),
            "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
            "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
            "scenes": planned,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_scenes = []
    for item in planned:
        print(f"generating {item['scene_id']} ({item['scenario']})", flush=True)
        scene_dir = output_dir / item["scene_id"]
        simulate_scenario(
            scenario=str(item["scenario"]),
            scene_dir=scene_dir,
            video_fps=video_fps,
            sim_hz=float(item["sim_hz"]),
            duration_sec=float(item["duration_sec"]),
            max_frames=max_frames,
            render_camera_names=render_camera_names,
            video_camera_names=video_camera_names,
            overwrite=overwrite,
            write_videos=write_videos,
        )
        manifest_scenes.append(
            {
                **item,
                "config": _repo_path(scene_dir / "config.json"),
                "metadata": _repo_path(scene_dir / "metadata.json"),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "room_physics_12view",
        "batch_root": _repo_path(output_dir),
        "num_scenes": len(manifest_scenes),
        "scenarios": scenarios,
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "scenes": manifest_scenes,
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _parse_scenarios(raw: str) -> list[str]:
    value = raw.strip().lower()
    if value == "all":
        return ["collision", "stacking", "deformable"]
    scenarios = [part.strip().lower() for part in value.split(",") if part.strip()]
    valid = {"collision", "stacking", "deformable"}
    unknown = sorted(set(scenarios).difference(valid))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Expected collision, stacking, deformable, or all.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="all",
        help="collision, stacking, deformable, comma-separated list, or all.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=VIDEO_FPS,
        help=f"Output/render FPS. Default: {VIDEO_FPS}.",
    )
    parser.add_argument(
        "--sim-hz",
        type=float,
        default=None,
        help="Internal PyBullet rate. Defaults are scenario-specific.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Override scenario default clip duration.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--render-camera-names",
        default="all",
        help="Comma-separated camera names to render, or all. Keep all for 4DGS.",
    )
    parser.add_argument(
        "--video-camera-names",
        default="all",
        help="Comma-separated camera names for MP4 previews, or all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = _parse_scenarios(args.scenario)
    render_camera_names = _parse_video_camera_names(args.render_camera_names)
    video_camera_names = _parse_video_camera_names(args.video_camera_names)
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.sim_hz is not None and args.sim_hz <= 0:
        raise ValueError("--sim-hz must be positive.")
    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")

    manifest = generate_dataset(
        scenarios=scenarios,
        output_dir=args.output_dir.resolve(),
        video_fps=args.video_fps,
        sim_hz=args.sim_hz,
        duration_sec=args.duration_sec,
        max_frames=args.max_frames,
        render_camera_names=render_camera_names,
        video_camera_names=video_camera_names,
        overwrite=not args.no_overwrite,
        write_videos=not args.no_videos,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))
    if not args.dry_run:
        print(f"Wrote room physics dataset: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
