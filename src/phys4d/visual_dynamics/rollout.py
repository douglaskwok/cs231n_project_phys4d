"""Autoregressive pose rollout for extrapolation eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phys4d.object_state import ObjectState, stack_state_vectors, trajectory_to_states, window_states
from phys4d.poses import ObjectPose, ObjectPoseTrajectory, load_object_poses_csv
from .dynamics_head import VisualDynamicsModel
from .dataset import VisualDynamicsDataset


def load_visual_dynamics_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[VisualDynamicsModel, int, int]:
    device = device or torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    history = int(payload["history"])
    train_end = int(payload.get("train_end_frame", 59))
    model = VisualDynamicsModel(history=history).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, history, train_end


def rollout_learned_states(
    model: VisualDynamicsModel,
    scene,
    *,
    train_end_frame: int,
    history: int,
    device: torch.device,
) -> list[ObjectState]:
    """Predict states for frames train_end+1 .. T-1 autoregressively."""

    traj = load_object_poses_csv(scene.poses_path)
    states = trajectory_to_states(traj)
    ds = VisualDynamicsDataset([scene], history=history, train_end_frame=train_end_frame)
    predicted = states[: train_end_frame + 1]

    with torch.no_grad():
        for frame in range(train_end_frame, len(states) - 1):
            hist = window_states(predicted, frame, history)
            hist_t = torch.from_numpy(stack_state_vectors(hist, include_velocity=True)).float()
            hist_t = hist_t.unsqueeze(0).to(device)
            crop = ds._load_crop(scene, frame).unsqueeze(0).to(device)
            next_vec = model.predict_next(hist_t, crop).cpu().numpy()[0]
            pos = next_vec[:3]
            quat = next_vec[3:7]
            vel = next_vec[7:10] if next_vec.shape[0] >= 10 else np.zeros(3)
            gt = states[frame + 1]
            predicted.append(
                ObjectState(
                    frame=gt.frame,
                    time_s=gt.time_s,
                    position=pos.astype(np.float64),
                    quat_xyzw=quat.astype(np.float64),
                    linear_velocity=vel.astype(np.float64),
                )
            )
    return predicted


def states_to_trajectory(states: list[ObjectState]) -> ObjectPoseTrajectory:
    poses = [
        ObjectPose(
            frame=s.frame,
            time_s=s.time_s,
            position=s.position.copy(),
            quat_xyzw=s.quat_xyzw.copy(),
            linear_velocity=s.linear_velocity.copy(),
        )
        for s in states
    ]
    return ObjectPoseTrajectory(poses=poses)
