"""Pose-history dataset for MLP baseline (oracle states from sim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from phys4d.param_ident.dataset import SceneSample, load_manifest
from phys4d.poses import load_object_poses_csv

__all__ = ["PoseHistoryDataset", "load_manifest"]


class PoseHistoryDataset(Dataset):
    def __init__(
        self,
        scenes: list[SceneSample],
        *,
        history: int = 16,
        frame_stride: int = 2,
        train_end_frame: int = 59,
    ) -> None:
        self.history = history
        self.frame_stride = frame_stride
        self.train_end_frame = train_end_frame
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene = self.scenes[index]
        traj = load_object_poses_csv(scene.scene_dir / "object_poses.csv")
        frames = [i * self.frame_stride for i in range(self.history)]
        frames = [f for f in frames if f <= self.train_end_frame]
        rows = []
        for f in frames:
            p = traj.by_frame(f)
            rows.append(
                np.concatenate([p.position, p.linear_velocity], dtype=np.float32)
            )
        while len(rows) < self.history:
            rows.append(rows[-1] if rows else np.zeros(6, dtype=np.float32))
        x = torch.from_numpy(np.stack(rows[: self.history], axis=0)).float()
        y = torch.from_numpy(scene.param_vector.copy()).float()
        return x, y
