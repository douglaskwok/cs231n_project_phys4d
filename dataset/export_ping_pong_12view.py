#!/usr/bin/env python
"""Export or generate 12-view ping-pong datasets.

Default mode exports the existing notebook MP4s into a structured dataset tree.
Variation modes generate new PyBullet scenes over restitution x ball trajectory angle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

CAMERA_NAMES = [
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
    "top",
    "top_oblique",
    "low_front_left",
    "low_back_right",
]

WIDTH = 960
HEIGHT = 544
VIDEO_FPS = 60.0
SIM_HZ = 480.0
DURATION_SEC = 4.0
TRAIN_DURATION_SEC = 1.0
TEST_DURATION_SEC = 0.5
TABLE_TOP_Z = 0.75
TABLE_LENGTH = 1.20
TABLE_WIDTH = 0.80
TABLE_THICKNESS = 0.06
BALL_RADIUS = 0.100
BALL_MASS = 0.0027
BALL_START_XY = [-0.12, -0.04]
BALL_START_HEIGHT_ABOVE_SURFACE = 0.80
BALL_INITIAL_ANGULAR_VELOCITY = [0.0, 0.0, 0.0]
GRAVITY = -9.80665
BALL_LATERAL_FRICTION = 0.20
BALL_ROLLING_FRICTION = 0.0005
BALL_SPINNING_FRICTION = 0.0005
TABLE_LATERAL_FRICTION = 0.35
CONTACT_PROCESSING_THRESHOLD = 0.0
RESTITUTION_VELOCITY_THRESHOLD = 0.0
TARGET = [0.0, 0.0, TABLE_TOP_Z + 0.20]
ROOM_HALF_X = 2.60
ROOM_HALF_Y = 2.60
ROOM_HEIGHT = 2.20
ROOM_WALL_THICKNESS = 0.06
ROOM_FLOOR_THICKNESS = 0.06
ROOM_FLOOR_Z = 0.0
DATASET_OUTPUTS_ROOT = REPO_ROOT / "dataset" / "outputs"
DEFAULT_OUTPUT_DIR = DATASET_OUTPUTS_ROOT / "ping_pong_12view"

FULL_RESTITUTIONS = [0.35, 0.50, 0.65, 0.80, 0.93]
FULL_BALL_ANGLES_DEG = [0.0, 5.0, 10.0, 15.0, 20.0]
POC_RESTITUTIONS = [0.50, 0.90]
POC_BALL_ANGLES_DEG = [0.0, 15.0]


def _ensure_pybullet():
    try:
        import pybullet as p  # type: ignore

        return p
    except ImportError as exc:
        print(
            "PyBullet is required for camera matrix export. Use phys_sim/bin/python "
            "or install pybullet.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _open_mp4_writer(imageio_module, path: Path, *, fps: float = VIDEO_FPS):
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MP4 writing requires imageio-ffmpeg. Use phys_sim/bin/python for this "
            "repo, install imageio-ffmpeg, or pass --no-videos to write PNGs only."
        ) from exc

    return imageio_module.get_writer(
        path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )


def _read_camera_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _default_camera_rows() -> list[dict]:
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
                "eye_x": "-0.9",
                "eye_y": "-1.35",
                "eye_z": str(TABLE_TOP_Z + 0.12),
                "up_x": "0",
                "up_y": "0",
                "up_z": "1",
                "fov_deg": "40",
            },
            {
                "name": "low_back_right",
                "eye_x": "0.9",
                "eye_y": "1.35",
                "eye_z": str(TABLE_TOP_Z + 0.12),
                "up_x": "0",
                "up_y": "0",
                "up_z": "1",
                "fov_deg": "40",
            },
        ]
    )
    return rows


def _blue_ball_mask(rgb: np.ndarray, *, dilate: int) -> np.ndarray:
    """Segment the blue notebook ball from compressed RGB video frames."""

    arr = rgb.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (b > 105) & (b > r + 35) & (b > g + 20) & (g > 35)
    mask = mask.astype(np.uint8) * 255
    if dilate <= 0:
        return mask

    try:
        import cv2  # type: ignore

        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        # Tiny fallback dilation to avoid making OpenCV a hard local dependency.
        out = mask.copy()
        for _ in range(dilate):
            padded = np.pad(out, 1, mode="constant")
            neigh = [
                padded[0:-2, 0:-2],
                padded[0:-2, 1:-1],
                padded[0:-2, 2:],
                padded[1:-1, 0:-2],
                padded[1:-1, 1:-1],
                padded[1:-1, 2:],
                padded[2:, 0:-2],
                padded[2:, 1:-1],
                padded[2:, 2:],
            ]
            out = np.maximum.reduce(neigh)
        return out


def _write_object_poses(src: Path, dst: Path) -> int:
    """Convert notebook trajectory CSV to the object_poses.csv schema used elsewhere."""

    rows = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(
                {
                    "time_s": float(row["time"]),
                    "frame": idx,
                    "x_m": float(row["x"]),
                    "y_m": float(row["y"]),
                    "z_m": float(row["z"]),
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                    "vx_m_s": float(row["vx"]),
                    "vy_m_s": float(row["vy"]),
                    "vz_m_s": float(row["vz"]),
                    "wx_rad_s": float(row["wx"]),
                    "wy_rad_s": float(row["wy"]),
                    "wz_rad_s": float(row["wz"]),
                    "num_table_contacts": int(float(row["num_table_contacts"])),
                }
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
        "num_table_contacts",
    ]
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _camera_records(camera_rows: list[dict]) -> list[dict]:
    p = _ensure_pybullet()
    records = []
    for idx, row in enumerate(camera_rows):
        name = row["name"]
        eye = [float(row["eye_x"]), float(row["eye_y"]), float(row["eye_z"])]
        up = [float(row["up_x"]), float(row["up_y"]), float(row["up_z"])]
        fov = float(row["fov_deg"])
        view = p.computeViewMatrix(eye, TARGET, up)
        proj = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=WIDTH / HEIGHT,
            nearVal=0.02,
            farVal=5.0,
        )
        records.append(
            {
                "index": idx,
                "name": name,
                "eye_m": eye,
                "look_at_m": TARGET,
                "up": up,
                "fov_deg": fov,
                "near": 0.02,
                "far": 5.0,
                "image_size": [WIDTH, HEIGHT],
                "view_matrix_row_major": list(view),
                "projection_matrix_row_major": list(proj),
            }
        )
    return records


def _flat_surface_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return tangent_x, tangent_y, normal


def _create_static_box(
    p,
    client: int,
    *,
    half_extents: list[float],
    position: list[float],
    rgba_color: list[float],
    collision: bool = True,
    lateral_friction: float = 0.80,
    restitution: float = 0.05,
) -> int:
    col = -1
    if collision:
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=client,
        )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgba_color,
        physicsClientId=client,
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        physicsClientId=client,
    )
    if collision:
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=lateral_friction,
            restitution=restitution,
            physicsClientId=client,
        )
    return body_id


def _create_room_geometry(p, client: int) -> dict:
    """Create a simple room around the table so full-video 4DGS has anchors."""

    wall_z = ROOM_FLOOR_Z + ROOM_HEIGHT / 2.0
    floor_z = ROOM_FLOOR_Z - ROOM_FLOOR_THICKNESS / 2.0
    floor_color = [0.70, 0.69, 0.64, 1.0]
    wall_front_color = [0.78, 0.82, 0.86, 1.0]
    wall_back_color = [0.83, 0.79, 0.72, 1.0]
    wall_left_color = [0.73, 0.80, 0.76, 1.0]
    wall_right_color = [0.84, 0.78, 0.82, 1.0]

    bodies = {
        "floor": _create_static_box(
            p,
            client,
            half_extents=[ROOM_HALF_X, ROOM_HALF_Y, ROOM_FLOOR_THICKNESS / 2.0],
            position=[0.0, 0.0, floor_z],
            rgba_color=floor_color,
            lateral_friction=0.85,
            restitution=0.10,
        ),
        "front_wall": _create_static_box(
            p,
            client,
            half_extents=[ROOM_HALF_X, ROOM_WALL_THICKNESS / 2.0, ROOM_HEIGHT / 2.0],
            position=[0.0, -ROOM_HALF_Y - ROOM_WALL_THICKNESS / 2.0, wall_z],
            rgba_color=wall_front_color,
        ),
        "back_wall": _create_static_box(
            p,
            client,
            half_extents=[ROOM_HALF_X, ROOM_WALL_THICKNESS / 2.0, ROOM_HEIGHT / 2.0],
            position=[0.0, ROOM_HALF_Y + ROOM_WALL_THICKNESS / 2.0, wall_z],
            rgba_color=wall_back_color,
        ),
        "left_wall": _create_static_box(
            p,
            client,
            half_extents=[ROOM_WALL_THICKNESS / 2.0, ROOM_HALF_Y, ROOM_HEIGHT / 2.0],
            position=[-ROOM_HALF_X - ROOM_WALL_THICKNESS / 2.0, 0.0, wall_z],
            rgba_color=wall_left_color,
        ),
        "right_wall": _create_static_box(
            p,
            client,
            half_extents=[ROOM_WALL_THICKNESS / 2.0, ROOM_HALF_Y, ROOM_HEIGHT / 2.0],
            position=[ROOM_HALF_X + ROOM_WALL_THICKNESS / 2.0, 0.0, wall_z],
            rgba_color=wall_right_color,
        ),
    }

    # Low-relief visual markers make the synthetic room usable as a 4DGS anchor,
    # without adding collision clutter near the table or ball.
    marker_specs = [
        ("back_blue", [0.28, 0.42, 0.95, 1.0], [-0.95, ROOM_HALF_Y - 0.002, 0.92], [0.28, 0.006, 0.20]),
        ("back_yellow", [0.95, 0.72, 0.18, 1.0], [0.95, ROOM_HALF_Y - 0.002, 1.35], [0.22, 0.006, 0.24]),
        ("front_red", [0.90, 0.30, 0.25, 1.0], [0.85, -ROOM_HALF_Y + 0.002, 0.98], [0.24, 0.006, 0.18]),
        ("left_teal", [0.12, 0.62, 0.62, 1.0], [-ROOM_HALF_X + 0.002, -0.85, 1.18], [0.006, 0.24, 0.24]),
        ("right_green", [0.32, 0.68, 0.28, 1.0], [ROOM_HALF_X - 0.002, 0.75, 1.08], [0.006, 0.28, 0.20]),
        ("floor_marker", [0.35, 0.35, 0.38, 1.0], [0.95, 0.70, ROOM_FLOOR_Z + 0.004], [0.28, 0.18, 0.004]),
    ]
    markers = {}
    for name, color, position, half_extents in marker_specs:
        markers[name] = _create_static_box(
            p,
            client,
            half_extents=half_extents,
            position=position,
            rgba_color=color,
            collision=False,
        )

    return {
        "name": "room_with_floor_and_walls",
        "floor_z_m": ROOM_FLOOR_Z,
        "half_extents_m": [ROOM_HALF_X, ROOM_HALF_Y, ROOM_HEIGHT],
        "wall_thickness_m": ROOM_WALL_THICKNESS,
        "floor_thickness_m": ROOM_FLOOR_THICKNESS,
        "has_ceiling": False,
        "anchor_markers": list(markers.keys()),
        "body_ids": {**bodies, **markers},
    }


def _initial_velocity_for_ball_angle(
    ball_angle_deg: float,
    *,
    ball_radius_m: float = BALL_RADIUS,
) -> list[float]:
    drop_to_contact_m = max(0.0, BALL_START_HEIGHT_ABOVE_SURFACE - ball_radius_m)
    impact_vertical_speed = math.sqrt(2.0 * abs(GRAVITY) * drop_to_contact_m)
    horizontal_speed = impact_vertical_speed * math.tan(math.radians(ball_angle_deg))
    return [horizontal_speed, 0.0, 0.0]


def _scene_name(index: int, restitution: float, ball_angle_deg: float) -> str:
    e = f"{restitution:.2f}".replace(".", "p")
    a = f"{ball_angle_deg:.1f}".replace("-", "m").replace(".", "p")
    return f"scene_{index:04d}_e{e}_a{a}"


def _variation_grid(
    variation_set: str,
    *,
    single_restitution: float = 0.90,
    single_ball_angle_deg: float = 0.0,
) -> list[dict]:
    if variation_set == "single":
        return [
            {
                "restitution": float(single_restitution),
                "ball_angle_deg": float(single_ball_angle_deg),
            }
        ]
    if variation_set == "poc":
        restitutions = POC_RESTITUTIONS
        angles = POC_BALL_ANGLES_DEG
    elif variation_set == "full":
        restitutions = FULL_RESTITUTIONS
        angles = FULL_BALL_ANGLES_DEG
    else:
        raise ValueError(f"Unknown variation set: {variation_set}")

    scenes = []
    for restitution in restitutions:
        for angle in angles:
            scenes.append(
                {
                    "restitution": float(restitution),
                    "ball_angle_deg": float(angle),
                }
            )
    return scenes


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def export_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    max_frames: int | None,
    mask_dilate: int,
    overwrite: bool,
    write_videos: bool,
    video_camera_names: set[str] | None,
) -> dict:
    import imageio.v2 as imageio

    cameras_csv = input_dir / "ping_pong_bounce_12view_cameras.csv"
    trajectory_csv = input_dir / "ping_pong_bounce_12view_trajectory.csv"
    if not cameras_csv.is_file():
        raise FileNotFoundError(f"Missing camera CSV: {cameras_csv}")
    if not trajectory_csv.is_file():
        raise FileNotFoundError(f"Missing trajectory CSV: {trajectory_csv}")

    camera_rows = _read_camera_csv(cameras_csv)
    names = [row["name"] for row in camera_rows]
    if names != CAMERA_NAMES:
        raise ValueError(f"Unexpected camera order: {names}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_root = output_dir / "rgb"
    masks_root = output_dir / "masks"
    videos_rgb_root = output_dir / "videos" / "rgb"
    videos_masks_root = output_dir / "videos" / "masks"
    if write_videos:
        videos_rgb_root.mkdir(parents=True, exist_ok=True)
        videos_masks_root.mkdir(parents=True, exist_ok=True)

    frame_counts = {}
    empty_masks = 0
    for cam_idx, name in enumerate(CAMERA_NAMES):
        video = input_dir / f"ping_pong_bounce_12view_{name}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"Missing 12-view video: {video}")
        rgb_dir = rgb_root / f"cam{cam_idx:02d}"
        mask_dir = masks_root / f"cam{cam_idx:02d}"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        reader = imageio.get_reader(video)
        rgb_writer = None
        mask_writer = None
        if write_videos and (video_camera_names is None or name in video_camera_names):
            rgb_video_path = videos_rgb_root / f"cam{cam_idx:02d}_{name}.mp4"
            mask_video_path = videos_masks_root / f"cam{cam_idx:02d}_{name}.mp4"
            if overwrite or not rgb_video_path.exists():
                rgb_writer = _open_mp4_writer(imageio, rgb_video_path)
            if overwrite or not mask_video_path.exists():
                mask_writer = _open_mp4_writer(imageio, mask_video_path)
        count = 0
        try:
            for frame_idx, frame in enumerate(reader):
                if max_frames is not None and frame_idx >= max_frames:
                    break
                rgb = np.asarray(frame)[..., :3].astype(np.uint8)
                mask = _blue_ball_mask(rgb, dilate=mask_dilate)
                if not mask.any():
                    empty_masks += 1

                frame_name = f"frame{frame_idx:05d}.png"
                rgb_path = rgb_dir / frame_name
                mask_path = mask_dir / frame_name
                if overwrite or not rgb_path.exists():
                    imageio.imwrite(rgb_path, rgb)
                if overwrite or not mask_path.exists():
                    imageio.imwrite(mask_path, mask)
                if rgb_writer is not None:
                    rgb_writer.append_data(rgb)
                if mask_writer is not None:
                    mask_writer.append_data(np.repeat(mask[..., None], 3, axis=2))
                count += 1
        finally:
            reader.close()
            if rgb_writer is not None:
                rgb_writer.close()
            if mask_writer is not None:
                mask_writer.close()
        frame_counts[name] = count

    pose_count = _write_object_poses(trajectory_csv, output_dir / "object_poses.csv")
    cameras = _camera_records(camera_rows)
    with (output_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump({"cameras": cameras}, f, indent=2)

    num_frames = min(frame_counts.values()) if frame_counts else 0
    config = {
        "experiment": "ping_pong_12view",
        "description": "Structured dataset exported from pybullet_tabletop_12view.ipynb videos.",
        "simulation": {
            "dt_s": 1.0 / VIDEO_FPS,
            "num_frames": num_frames,
            "train_frames": [0, max(0, min(59, num_frames - 1))],
            "test_frames": [min(60, max(0, num_frames - 1)), max(0, min(89, num_frames - 1))],
        },
        "cameras": {
            "num_cameras": len(CAMERA_NAMES),
            "layout": "12view_notebook_named_rig",
            "train_cameras": list(range(10)),
            "test_cameras": [10, 11],
            "image_size": [WIDTH, HEIGHT],
            "fov_deg": 48.0,
            "note": "Per-camera fov is stored in cameras.json; config fov_deg is a fallback for exporters.",
        },
        "outputs": {
            "rgb_frames": _repo_path(output_dir / "rgb"),
            "masks": _repo_path(output_dir / "masks"),
            "rgb_videos": _repo_path(output_dir / "videos" / "rgb"),
            "mask_videos": _repo_path(output_dir / "videos" / "masks"),
            "camera_poses": _repo_path(output_dir / "cameras.json"),
            "object_poses": _repo_path(output_dir / "object_poses.csv"),
            "metadata": _repo_path(output_dir / "metadata.json"),
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = {
        "source": "pybullet_tabletop_12view.ipynb",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "camera_names": CAMERA_NAMES,
        "num_cameras": len(CAMERA_NAMES),
        "frame_counts": frame_counts,
        "num_frames_min": num_frames,
        "pose_rows": pose_count,
        "mask_source": "RGB blue-ball color threshold",
        "mask_dilate": mask_dilate,
        "empty_masks": empty_masks,
        "video_fps": VIDEO_FPS,
        "videos_written": write_videos,
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "rgb_videos": _repo_path(videos_rgb_root) if write_videos else None,
        "mask_videos": _repo_path(videos_masks_root) if write_videos else None,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _write_pose_rows(pose_rows: list[dict], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
        "num_table_contacts",
    ]
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pose_rows)


def _steps_per_frame(video_fps: float, sim_hz: float) -> int:
    raw = sim_hz / video_fps
    steps = int(round(raw))
    if steps < 1:
        raise ValueError(
            f"--sim-hz ({sim_hz}) must be at least --video-fps ({video_fps})."
        )
    if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"--sim-hz ({sim_hz}) must be an integer multiple of --video-fps "
            f"({video_fps}) so frames land on simulation steps. Try 480/60, "
            "480/120, 480/240, or raise --sim-hz."
        )
    return steps


def _frame_split_ranges(num_frames: int, video_fps: float) -> tuple[list[int], list[int]]:
    train_end = max(0, min(int(round(TRAIN_DURATION_SEC * video_fps)) - 1, num_frames - 1))
    test_start = min(train_end + 1, max(0, num_frames - 1))
    test_end = max(
        test_start,
        min(
            int(round((TRAIN_DURATION_SEC + TEST_DURATION_SEC) * video_fps)) - 1,
            max(0, num_frames - 1),
        ),
    )
    return [0, train_end], [test_start, test_end]


def _simulate_variation_scene(
    *,
    scene_dir: Path,
    restitution: float,
    ball_angle_deg: float,
    ball_radius_m: float,
    environment: str,
    video_fps: float,
    sim_hz: float,
    duration_sec: float,
    render_camera_names: set[str] | None,
    max_frames: int | None,
    overwrite: bool,
    write_videos: bool,
    video_camera_names: set[str] | None,
) -> dict:
    p = _ensure_pybullet()
    import imageio.v2 as imageio
    import pybullet_data

    steps_per_frame = _steps_per_frame(video_fps, sim_hz)
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0.0, 0.0, GRAVITY, physicsClientId=client)
    p.setTimeStep(1.0 / sim_hz, physicsClientId=client)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / sim_hz,
        numSolverIterations=150,
        numSubSteps=2,
        deterministicOverlappingPairs=1,
        restitutionVelocityThreshold=RESTITUTION_VELOCITY_THRESHOLD,
        physicsClientId=client,
    )

    if environment == "room":
        environment_info = _create_room_geometry(p, client)
    elif environment == "tabletop":
        environment_info = {
            "name": "tabletop_only_black_background",
            "floor_z_m": None,
            "has_ceiling": False,
            "anchor_markers": [],
        }
    else:
        raise ValueError(f"Unknown environment: {environment}")

    tangent_x, tangent_y, normal = _flat_surface_basis()
    table_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    surface_center = np.array([0.0, 0.0, TABLE_TOP_Z], dtype=np.float64)
    table_center = surface_center - normal * (TABLE_THICKNESS / 2.0)

    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_LENGTH / 2.0, TABLE_WIDTH / 2.0, TABLE_THICKNESS / 2.0],
        physicsClientId=client,
    )
    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_LENGTH / 2.0, TABLE_WIDTH / 2.0, TABLE_THICKNESS / 2.0],
        rgbaColor=[0.14, 0.40, 0.28, 1.0],
        physicsClientId=client,
    )
    table_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=table_center.tolist(),
        baseOrientation=table_orientation,
        physicsClientId=client,
    )
    p.changeDynamics(
        table_id,
        -1,
        restitution=restitution,
        lateralFriction=TABLE_LATERAL_FRICTION,
        contactProcessingThreshold=CONTACT_PROCESSING_THRESHOLD,
        physicsClientId=client,
    )

    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius_m, physicsClientId=client)
    ball_vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=ball_radius_m,
        rgbaColor=[0.10, 0.52, 0.88, 1.0],
        physicsClientId=client,
    )
    start_surface_point = surface_center + tangent_x * BALL_START_XY[0] + tangent_y * BALL_START_XY[1]
    ball_start = start_surface_point + normal * BALL_START_HEIGHT_ABOVE_SURFACE
    ball_id = p.createMultiBody(
        baseMass=BALL_MASS,
        baseCollisionShapeIndex=ball_col,
        baseVisualShapeIndex=ball_vis,
        basePosition=ball_start.tolist(),
        physicsClientId=client,
    )
    p.changeDynamics(
        ball_id,
        -1,
        restitution=restitution,
        lateralFriction=BALL_LATERAL_FRICTION,
        rollingFriction=BALL_ROLLING_FRICTION,
        spinningFriction=BALL_SPINNING_FRICTION,
        contactProcessingThreshold=CONTACT_PROCESSING_THRESHOLD,
        physicsClientId=client,
    )
    p.resetBaseVelocity(
        ball_id,
        linearVelocity=_initial_velocity_for_ball_angle(ball_angle_deg, ball_radius_m=ball_radius_m),
        angularVelocity=BALL_INITIAL_ANGULAR_VELOCITY,
        physicsClientId=client,
    )

    camera_rows = _default_camera_rows()
    if render_camera_names is not None:
        camera_rows = [row for row in camera_rows if row["name"] in render_camera_names]
        if not camera_rows:
            raise ValueError(f"No cameras selected from: {sorted(render_camera_names)}")
    camera_records = _camera_records(camera_rows)
    camera_names = [str(cam["name"]) for cam in camera_records]
    if len(camera_records) == len(CAMERA_NAMES):
        camera_layout = "12view_notebook_named_rig"
        train_cameras = list(range(10))
        test_cameras = [10, 11]
    else:
        camera_layout = "selected_from_12view_notebook_named_rig"
        train_cameras = list(range(len(camera_records)))
        test_cameras = []
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb_root = scene_dir / "rgb"
    masks_root = scene_dir / "masks"
    table_masks_root = scene_dir / "masks_table"
    ball_table_masks_root = scene_dir / "masks_ball_table"
    videos_rgb_root = scene_dir / "videos" / "rgb"
    videos_masks_root = scene_dir / "videos" / "masks"
    videos_table_masks_root = scene_dir / "videos" / "masks_table"
    videos_ball_table_masks_root = scene_dir / "videos" / "masks_ball_table"
    rgb_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    table_masks_root.mkdir(parents=True, exist_ok=True)
    ball_table_masks_root.mkdir(parents=True, exist_ok=True)
    if write_videos:
        videos_rgb_root.mkdir(parents=True, exist_ok=True)
        videos_masks_root.mkdir(parents=True, exist_ok=True)
        videos_table_masks_root.mkdir(parents=True, exist_ok=True)
        videos_ball_table_masks_root.mkdir(parents=True, exist_ok=True)
    with (scene_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump({"cameras": camera_records}, f, indent=2)

    num_frames = int(duration_sec * video_fps) + 1
    if max_frames is not None:
        num_frames = min(num_frames, int(max_frames))
    train_frames, test_frames = _frame_split_ranges(num_frames, video_fps)

    pose_rows: list[dict] = []
    empty_masks = 0
    empty_table_masks = 0
    empty_ball_table_masks = 0
    video_writers: dict[int, tuple[object | None, object | None, object | None, object | None]] = {}
    try:
        if write_videos:
            for cam in camera_records:
                cam_idx = int(cam["index"])
                name = str(cam["name"])
                if video_camera_names is not None and name not in video_camera_names:
                    continue
                rgb_video_path = videos_rgb_root / f"cam{cam_idx:02d}_{name}.mp4"
                mask_video_path = videos_masks_root / f"cam{cam_idx:02d}_{name}.mp4"
                table_mask_video_path = videos_table_masks_root / f"cam{cam_idx:02d}_{name}.mp4"
                ball_table_mask_video_path = videos_ball_table_masks_root / f"cam{cam_idx:02d}_{name}.mp4"
                rgb_writer = None
                mask_writer = None
                table_mask_writer = None
                ball_table_mask_writer = None
                if overwrite or not rgb_video_path.exists():
                    rgb_writer = _open_mp4_writer(imageio, rgb_video_path, fps=video_fps)
                if overwrite or not mask_video_path.exists():
                    mask_writer = _open_mp4_writer(imageio, mask_video_path, fps=video_fps)
                if overwrite or not table_mask_video_path.exists():
                    table_mask_writer = _open_mp4_writer(imageio, table_mask_video_path, fps=video_fps)
                if overwrite or not ball_table_mask_video_path.exists():
                    ball_table_mask_writer = _open_mp4_writer(
                        imageio,
                        ball_table_mask_video_path,
                        fps=video_fps,
                    )
                video_writers[cam_idx] = (
                    rgb_writer,
                    mask_writer,
                    table_mask_writer,
                    ball_table_mask_writer,
                )

        for frame_idx in range(num_frames):
            if frame_idx == 0 or frame_idx % 30 == 0 or frame_idx == num_frames - 1:
                print(
                    f"  frame {frame_idx + 1}/{num_frames} "
                    f"(e={restitution:.2f}, ball_angle={ball_angle_deg:.1f})",
                    flush=True,
                )
            pos, orn = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            lin_vel, ang_vel = p.getBaseVelocity(ball_id, physicsClientId=client)
            contacts = p.getContactPoints(bodyA=ball_id, bodyB=table_id, physicsClientId=client)
            pose_rows.append(
                {
                    "time_s": frame_idx / video_fps,
                    "frame": frame_idx,
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
                    "num_table_contacts": len(contacts),
                }
            )

            for cam in camera_records:
                cam_idx = int(cam["index"])
                rgb_dir = rgb_root / f"cam{cam_idx:02d}"
                mask_dir = masks_root / f"cam{cam_idx:02d}"
                table_mask_dir = table_masks_root / f"cam{cam_idx:02d}"
                ball_table_mask_dir = ball_table_masks_root / f"cam{cam_idx:02d}"
                rgb_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)
                table_mask_dir.mkdir(parents=True, exist_ok=True)
                ball_table_mask_dir.mkdir(parents=True, exist_ok=True)
                img = p.getCameraImage(
                    width=WIDTH,
                    height=HEIGHT,
                    viewMatrix=cam["view_matrix_row_major"],
                    projectionMatrix=cam["projection_matrix_row_major"],
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=client,
                )
                rgb = np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)
                seg = np.reshape(img[4], (HEIGHT, WIDTH))
                mask = (seg == ball_id).astype(np.uint8) * 255
                table_mask = (seg == table_id).astype(np.uint8) * 255
                ball_table_mask = ((seg == ball_id) | (seg == table_id)).astype(np.uint8) * 255
                if not mask.any():
                    empty_masks += 1
                if not table_mask.any():
                    empty_table_masks += 1
                if not ball_table_mask.any():
                    empty_ball_table_masks += 1

                frame_name = f"frame{frame_idx:05d}.png"
                rgb_path = rgb_dir / frame_name
                mask_path = mask_dir / frame_name
                table_mask_path = table_mask_dir / frame_name
                ball_table_mask_path = ball_table_mask_dir / frame_name
                if overwrite or not rgb_path.exists():
                    imageio.imwrite(rgb_path, rgb)
                if overwrite or not mask_path.exists():
                    imageio.imwrite(mask_path, mask)
                if overwrite or not table_mask_path.exists():
                    imageio.imwrite(table_mask_path, table_mask)
                if overwrite or not ball_table_mask_path.exists():
                    imageio.imwrite(ball_table_mask_path, ball_table_mask)
                if cam_idx in video_writers:
                    rgb_writer, mask_writer, table_mask_writer, ball_table_mask_writer = video_writers[cam_idx]
                    if rgb_writer is not None:
                        rgb_writer.append_data(rgb)
                    if mask_writer is not None:
                        mask_writer.append_data(np.repeat(mask[..., None], 3, axis=2))
                    if table_mask_writer is not None:
                        table_mask_writer.append_data(np.repeat(table_mask[..., None], 3, axis=2))
                    if ball_table_mask_writer is not None:
                        ball_table_mask_writer.append_data(
                            np.repeat(ball_table_mask[..., None], 3, axis=2)
                        )

            for _ in range(steps_per_frame):
                p.stepSimulation(physicsClientId=client)
    finally:
        for writers in video_writers.values():
            for writer in writers:
                if writer is not None:
                    writer.close()
        p.disconnect(physicsClientId=client)
    _write_pose_rows(pose_rows, scene_dir / "object_poses.csv")

    surface_normal = normal.tolist()
    config = {
        "experiment": "ping_pong_12view_restitution_ball_angle",
        "description": "Generated 12-view ping-pong scene with fixed surface and varied restitution/ball trajectory angle.",
        "scene": {
            "true_restitution": restitution,
            "ball_angle_deg": ball_angle_deg,
            "ball_angle_note": "Approximate first-impact trajectory angle from vertical in vacuum; implemented by horizontal launch speed along +x.",
            "environment": environment_info,
            "surface_normal": surface_normal,
            "objects": [
                {
                    "name": "ping_pong_ball",
                    "shape": "sphere",
                    "radius_m": ball_radius_m,
                    "mass_kg": BALL_MASS,
                    "initial_position_m": ball_start.tolist(),
                    "initial_velocity_m_s": _initial_velocity_for_ball_angle(
                        ball_angle_deg,
                        ball_radius_m=ball_radius_m,
                    ),
                },
                {
                    "name": "table",
                    "shape": "box",
                    "top_center_m": surface_center.tolist(),
                    "size_m": [TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS],
                },
            ],
        },
        "simulation": {
            "physics_engine": "pybullet",
            "dt_s": 1.0 / video_fps,
            "internal_dt_s": 1.0 / sim_hz,
            "steps_per_frame": steps_per_frame,
            "num_frames": num_frames,
            "duration_s": (num_frames - 1) / video_fps if num_frames else 0.0,
            "gravity_m_s2": [0.0, 0.0, GRAVITY],
            "contact_processing_threshold": CONTACT_PROCESSING_THRESHOLD,
            "restitution_velocity_threshold": RESTITUTION_VELOCITY_THRESHOLD,
            "train_frames": train_frames,
            "test_frames": test_frames,
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
            "rgb_frames": _repo_path(scene_dir / "rgb"),
            "masks": _repo_path(scene_dir / "masks"),
            "table_masks": _repo_path(scene_dir / "masks_table"),
            "ball_table_masks": _repo_path(scene_dir / "masks_ball_table"),
            "rgb_videos": _repo_path(scene_dir / "videos" / "rgb"),
            "mask_videos": _repo_path(scene_dir / "videos" / "masks"),
            "table_mask_videos": _repo_path(scene_dir / "videos" / "masks_table"),
            "ball_table_mask_videos": _repo_path(scene_dir / "videos" / "masks_ball_table"),
            "camera_poses": _repo_path(scene_dir / "cameras.json"),
            "object_poses": _repo_path(scene_dir / "object_poses.csv"),
            "metadata": _repo_path(scene_dir / "metadata.json"),
        },
    }
    with (scene_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    physics_params = {
        "param_names": ["restitution", "ball_angle_deg"],
        "param_vector": [restitution, ball_angle_deg],
        "fixed_params": {
            "mass_kg": BALL_MASS,
            "radius_m": ball_radius_m,
            "environment": environment,
            "initial_direction": "+x",
            "ball_lateral_friction": BALL_LATERAL_FRICTION,
            "table_lateral_friction": TABLE_LATERAL_FRICTION,
            "contact_processing_threshold": CONTACT_PROCESSING_THRESHOLD,
            "restitution_velocity_threshold": RESTITUTION_VELOCITY_THRESHOLD,
            "drag_enabled": False,
        },
    }
    with (scene_dir / "physics_params.json").open("w", encoding="utf-8") as f:
        json.dump(physics_params, f, indent=2)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "dataset/export_ping_pong_12view.py",
        "output_dir": _repo_path(scene_dir),
        "num_frames": num_frames,
        "num_cameras": len(camera_records),
        "camera_names": camera_names,
        "video_fps": video_fps,
        "sim_hz": sim_hz,
        "steps_per_frame": steps_per_frame,
        "duration_s": (num_frames - 1) / video_fps if num_frames else 0.0,
        "environment": environment_info,
        "empty_masks": empty_masks,
        "empty_table_masks": empty_table_masks,
        "empty_ball_table_masks": empty_ball_table_masks,
        "mask_outputs": {
            "ball": _repo_path(masks_root),
            "table": _repo_path(table_masks_root),
            "ball_table": _repo_path(ball_table_masks_root),
        },
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "rgb_videos": _repo_path(videos_rgb_root) if write_videos else None,
        "mask_videos": _repo_path(videos_masks_root) if write_videos else None,
        "table_mask_videos": _repo_path(videos_table_masks_root) if write_videos else None,
        "ball_table_mask_videos": _repo_path(videos_ball_table_masks_root) if write_videos else None,
        "physics_params": physics_params,
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def generate_variation_dataset(
    *,
    output_dir: Path,
    variation_set: str,
    restitution: float,
    ball_angle_deg: float,
    ball_radius_m: float,
    environment: str,
    video_fps: float,
    sim_hz: float,
    duration_sec: float,
    render_camera_names: set[str] | None,
    max_frames: int | None,
    overwrite: bool,
    write_videos: bool,
    video_camera_names: set[str] | None,
    dry_run: bool,
) -> dict:
    scenes = _variation_grid(
        variation_set,
        single_restitution=restitution,
        single_ball_angle_deg=ball_angle_deg,
    )
    planned = []
    for idx, params in enumerate(scenes):
        scene_id = _scene_name(idx, params["restitution"], params["ball_angle_deg"])
        planned.append({"scene_id": scene_id, **params})

    if dry_run:
        return {
            "variation_set": variation_set,
            "output_dir": str(output_dir),
            "num_scenes": len(planned),
            "ball_radius_m": ball_radius_m,
            "environment": environment,
            "video_fps": video_fps,
            "sim_hz": sim_hz,
            "duration_sec": duration_sec,
            "steps_per_frame": _steps_per_frame(video_fps, sim_hz),
            "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
            "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
            "scenes": planned,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_scenes = []
    for idx, params in enumerate(scenes):
        scene_id = _scene_name(idx, params["restitution"], params["ball_angle_deg"])
        scene_dir = output_dir / scene_id
        print(
            f"[{idx + 1}/{len(scenes)}] generating {scene_id} "
            f"(e={params['restitution']:.2f}, ball_angle={params['ball_angle_deg']:.1f})",
            flush=True,
        )
        _simulate_variation_scene(
            scene_dir=scene_dir,
            restitution=params["restitution"],
            ball_angle_deg=params["ball_angle_deg"],
            ball_radius_m=ball_radius_m,
            environment=environment,
            video_fps=video_fps,
            sim_hz=sim_hz,
            duration_sec=duration_sec,
            render_camera_names=render_camera_names,
            max_frames=max_frames,
            overwrite=overwrite,
            write_videos=write_videos,
            video_camera_names=video_camera_names,
        )
        manifest_scenes.append(
            {
                "scene_id": scene_id,
                "path": _repo_path(scene_dir),
                "restitution": params["restitution"],
                "ball_angle_deg": params["ball_angle_deg"],
                "config": _repo_path(scene_dir / "config.json"),
                "metadata": _repo_path(scene_dir / "metadata.json"),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "ping_pong_12view_restitution_ball_angle_grid",
        "variation_set": variation_set,
        "batch_root": _repo_path(output_dir),
        "num_scenes": len(manifest_scenes),
        "param_names": ["restitution", "ball_angle_deg"],
        "environment": environment,
        "video_fps": video_fps,
        "sim_hz": sim_hz,
        "duration_sec": duration_sec,
        "steps_per_frame": _steps_per_frame(video_fps, sim_hz),
        "fixed_params": {
            "mass_kg": BALL_MASS,
            "radius_m": ball_radius_m,
            "environment": environment,
            "initial_direction": "+x",
            "drop_height_above_surface_m": BALL_START_HEIGHT_ABOVE_SURFACE,
            "drag_enabled": False,
        },
        "videos_written": write_videos,
        "render_camera_names": "all" if render_camera_names is None else sorted(render_camera_names),
        "video_camera_names": "all" if video_camera_names is None else sorted(video_camera_names),
        "scenes": manifest_scenes,
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _parse_video_camera_names(raw: str) -> set[str] | None:
    value = raw.strip()
    if value.lower() == "all":
        return None
    names = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(names.difference(CAMERA_NAMES))
    if unknown:
        raise ValueError(f"Unknown video camera names: {unknown}. Expected one of {CAMERA_NAMES} or 'all'.")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "pybullet_outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--mask-dilate", type=int, default=1)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip MP4 preview videos and write only PNG frames/masks.",
    )
    parser.add_argument(
        "--video-camera-names",
        default="all",
        help=(
            "Comma-separated camera names for RGB/mask MP4 previews, or 'all'. "
            "PNG frames/masks are still written for every camera. Default: all."
        ),
    )
    parser.add_argument(
        "--render-camera-names",
        default="all",
        help=(
            "Comma-separated camera names to render for generated variation scenes, "
            "or 'all'. Keep 'all' for 4DGS. Default: all."
        ),
    )
    parser.add_argument(
        "--variation-set",
        choices=["none", "single", "poc", "full"],
        default="none",
        help=(
            "none exports existing notebook videos; single generates one 12-view "
            "scene; poc generates 2x2 restitution x ball-angle scenes; full "
            "generates the full configured grid."
        ),
    )
    parser.add_argument(
        "--restitution",
        type=float,
        default=0.90,
        help="Restitution for --variation-set single.",
    )
    parser.add_argument(
        "--ball-angle-deg",
        type=float,
        default=0.0,
        help="Ball trajectory angle for --variation-set single.",
    )
    parser.add_argument(
        "--ball-radius-m",
        type=float,
        default=BALL_RADIUS,
        help=(
            "Ball radius in meters for generated variation scenes. "
            f"Default: {BALL_RADIUS}."
        ),
    )
    parser.add_argument(
        "--environment",
        choices=["room", "tabletop"],
        default="room",
        help=(
            "Generated-scene background/environment. 'room' adds a floor, "
            "four walls, and visual anchor panels. 'tabletop' keeps the old "
            "black-background tabletop scene. Default: room."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=VIDEO_FPS,
        help=(
            "Output/render FPS for generated variation scenes. "
            f"Default: {VIDEO_FPS}. Try 120 for denser temporal supervision."
        ),
    )
    parser.add_argument(
        "--sim-hz",
        type=float,
        default=SIM_HZ,
        help=(
            "Internal PyBullet simulation frequency. Must be an integer multiple "
            f"of --video-fps. Default: {SIM_HZ}."
        ),
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=DURATION_SEC,
        help=(
            "Generated scene duration before --max-frames is applied. "
            f"Default: {DURATION_SEC}."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    video_camera_names = _parse_video_camera_names(args.video_camera_names)
    render_camera_names = _parse_video_camera_names(args.render_camera_names)
    if args.ball_radius_m <= 0:
        raise ValueError("--ball-radius-m must be positive.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.sim_hz <= 0:
        raise ValueError("--sim-hz must be positive.")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive.")
    _steps_per_frame(args.video_fps, args.sim_hz)

    output_dir = args.output_dir
    if args.variation_set != "none" and output_dir == DEFAULT_OUTPUT_DIR:
        suffix = args.variation_set
        output_dir = DATASET_OUTPUTS_ROOT / f"ping_pong_12view_restitution_ball_angle_{suffix}"

    if args.variation_set != "none":
        manifest = generate_variation_dataset(
            output_dir=output_dir.resolve(),
            variation_set=args.variation_set,
            restitution=args.restitution,
            ball_angle_deg=args.ball_angle_deg,
            ball_radius_m=args.ball_radius_m,
            environment=args.environment,
            video_fps=args.video_fps,
            sim_hz=args.sim_hz,
            duration_sec=args.duration_sec,
            render_camera_names=render_camera_names,
            max_frames=args.max_frames,
            overwrite=not args.no_overwrite,
            write_videos=not args.no_videos,
            video_camera_names=video_camera_names,
            dry_run=args.dry_run,
        )
        print(json.dumps(manifest, indent=2))
        if not args.dry_run:
            print(f"Wrote variation dataset: {output_dir.resolve()}")
        return 0

    if args.dry_run:
        print("Legacy export dry run")
        print(f"input_dir: {args.input_dir.resolve()}")
        print(f"output_dir: {output_dir.resolve()}")
        return 0

    meta = export_dataset(
        input_dir=args.input_dir.resolve(),
        output_dir=output_dir.resolve(),
        max_frames=args.max_frames,
        mask_dilate=args.mask_dilate,
        overwrite=not args.no_overwrite,
        write_videos=not args.no_videos,
        video_camera_names=video_camera_names,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote 12-view dataset: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
