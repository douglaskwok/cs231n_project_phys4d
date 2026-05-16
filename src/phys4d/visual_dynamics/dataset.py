"""Training samples: K-step history + fixed t=0 visual feature → next state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from phys4d.object_state import stack_state_vectors, window_states
from phys4d.perception.states import load_states_csv_or_poses
from phys4d.visual_dynamics.scenes import SceneRecord


def _load_visual_feat(scene_dir: Path) -> np.ndarray:
    path = scene_dir / "perception" / "visual_feat_t0.npy"
    if path.is_file():
        return np.load(path)
    from phys4d.perception.visual_features import extract_t0_visual_feature

    return extract_t0_visual_feature(scene_dir)


class VisualDynamicsDataset(Dataset):
    def __init__(
        self,
        scenes: list[SceneRecord],
        *,
        history: int = 8,
        train_end_frame: int = 59,
        dt_s: float = 1.0 / 60.0,
        num_objects: int = 1,
    ) -> None:
        self.history = history
        self.train_end_frame = train_end_frame
        self.dt_s = dt_s
        self.num_objects = num_objects
        self.samples: list[tuple[SceneRecord, int]] = []
        self._visual: dict[str, np.ndarray] = {}
        self._states: dict[str, np.ndarray] = {}

        for scene in scenes:
            states = load_states_csv_or_poses(
                scene.scene_dir / "object_poses.csv", dt_s=dt_s
            )
            self._states[scene.scene_id] = stack_state_vectors(states, include_velocity=True)
            self._visual[scene.scene_id] = _load_visual_feat(scene.scene_dir)
            for frame in range(history - 1, train_end_frame):
                self.samples.append((scene, frame))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scene, frame = self.samples[index]
        mat = self._states[scene.scene_id]
        from phys4d.object_state import ObjectState

        states = [
            ObjectState(
                frame=i,
                time_s=i * self.dt_s,
                position=mat[i, :3],
                quat_xyzw=mat[i, 3:7],
                linear_velocity=mat[i, 7:10],
            )
            for i in range(mat.shape[0])
        ]
        hist = window_states(states, frame, self.history)
        hist_vec = stack_state_vectors(hist, include_velocity=True)
        target = mat[frame + 1]

        # (N, K, D) and (N, V)
        hist_t = torch.from_numpy(hist_vec).float().unsqueeze(0)
        vis = torch.from_numpy(self._visual[scene.scene_id]).float().unsqueeze(0)
        tgt = torch.from_numpy(target).float().unsqueeze(0)

        return {
            "state_history": hist_t,
            "visual_feat": vis,
            "target": tgt,
            "scene_id": scene.scene_id,
            "frame": frame,
        }


def collate_visual_dynamics(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "state_history": torch.stack([b["state_history"] for b in batch], dim=0),
        "visual_feat": torch.stack([b["visual_feat"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "scene_id": [b["scene_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.long),
    }
