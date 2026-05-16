"""Run Phase-2 perception export for one scene directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phys4d.object_state import stack_state_vectors
from phys4d.perception.states import load_states_csv_or_poses
from phys4d.perception.visual_features import extract_t0_visual_feature


def extract_scene_perception(
    scene_dir: Path,
    out_dir: Path | None = None,
    *,
    dt_s: float = 1.0 / 60.0,
    visual_frame: int = 0,
    image_size: int = 128,
    max_views: int = 10,
    use_sim_velocity: bool = False,
    device: str = "cpu",
) -> dict:
    """Write states (T,10), visual_feat (D,), meta.json under scene/perception/."""
    scene_dir = scene_dir.resolve()
    poses_path = scene_dir / "object_poses.csv"
    if not poses_path.is_file():
        raise FileNotFoundError(f"Missing {poses_path}")

    out = out_dir or (scene_dir / "perception")
    out.mkdir(parents=True, exist_ok=True)

    states = load_states_csv_or_poses(
        poses_path, dt_s=dt_s, use_sim_velocity=use_sim_velocity
    )
    state_mat = stack_state_vectors(states, include_velocity=True)
    np.save(out / "states.npy", state_mat.astype(np.float32))

    visual = extract_t0_visual_feature(
        scene_dir,
        frame=visual_frame,
        image_size=image_size,
        max_views=max_views,
        device=device,
    )
    np.save(out / "visual_feat_t0.npy", visual)

    meta = {
        "scene_dir": str(scene_dir),
        "num_frames": int(state_mat.shape[0]),
        "state_dim": int(state_mat.shape[1]),
        "visual_dim": int(visual.shape[0]),
        "visual_frame": visual_frame,
        "use_sim_velocity": use_sim_velocity,
        "object_labels": ["sphere"],
    }
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
