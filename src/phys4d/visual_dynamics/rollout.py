"""Autoregressive rollout for inference and training curriculum."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phys4d.object_state import ObjectState, stack_state_vectors, window_states
from phys4d.perception.states import load_states_csv_or_poses
from phys4d.visual_dynamics.dataset import _load_visual_feat
from phys4d.visual_dynamics.dynamics import ObjectTokenDynamics
from phys4d.visual_dynamics.scenes import SceneRecord


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[ObjectTokenDynamics, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    history = int(ckpt.get("history", 8))
    visual_dim = int(ckpt.get("visual_dim", 64))
    use_visual = bool(ckpt.get("use_visual", True))
    model = ObjectTokenDynamics(
        history=history,
        visual_dim=visual_dim,
        use_visual=use_visual,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _states_from_matrix(mat: np.ndarray, dt_s: float) -> list[ObjectState]:
    return [
        ObjectState(
            frame=i,
            time_s=i * dt_s,
            position=mat[i, :3].copy(),
            quat_xyzw=mat[i, 3:7].copy(),
            linear_velocity=mat[i, 7:10].copy(),
        )
        for i in range(mat.shape[0])
    ]


@torch.no_grad()
def rollout_scene(
    model: ObjectTokenDynamics,
    scene: SceneRecord,
    *,
    start_frame: int,
    horizon: int,
    history: int,
    dt_s: float,
    device: torch.device,
    history_noise_std: float = 0.0,
) -> np.ndarray:
    """Returns predicted states (horizon+1, 10) from start_frame inclusive."""
    states = load_states_csv_or_poses(scene.scene_dir / "object_poses.csv", dt_s=dt_s)
    mat = stack_state_vectors(states, include_velocity=True)
    visual = _load_visual_feat(scene.scene_dir)
    vis_t = torch.from_numpy(visual).float().view(1, 1, -1).to(device)

    current_states = _states_from_matrix(mat[: start_frame + 1], dt_s)
    pred_rows = [mat[start_frame]]

    for step in range(horizon):
        frame = start_frame + step
        hist = window_states(current_states, frame, history)
        hist_vec = stack_state_vectors(hist, include_velocity=True)
        hist_t = torch.from_numpy(hist_vec).float().view(1, 1, history, -1).to(device)
        if history_noise_std > 0:
            hist_t = hist_t + history_noise_std * torch.randn_like(hist_t)
        nxt = model.predict_next(hist_t, vis_t).cpu().numpy()[0, 0]
        pred_rows.append(nxt)
        nf = frame + 1
        current_states.append(
            ObjectState(
                frame=nf,
                time_s=nf * dt_s,
                position=nxt[:3].copy(),
                quat_xyzw=nxt[3:7].copy(),
                linear_velocity=nxt[7:10].copy(),
            )
        )
    return np.stack(pred_rows, axis=0).astype(np.float32)
