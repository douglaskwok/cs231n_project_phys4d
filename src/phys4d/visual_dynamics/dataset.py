"""Cross-scene dataset for visual dynamics training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from phys4d.object_state import stack_state_vectors, window_states, trajectory_to_states
from phys4d.poses import load_object_poses_csv


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    config_path: Path
    poses_path: Path
    rgb_root: Path
    restitution: float
    mass_kg: float
    drop_z_m: float


def discover_batch_scenes(batch_root: Path) -> list[SceneRecord]:
    batch_root = batch_root.resolve()
    scenes: list[SceneRecord] = []
    for cfg_path in sorted(batch_root.glob("*/config.json")):
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        scene_id = cfg_path.parent.name
        poses = cfg_path.parent / "object_poses.csv"
        rgb = cfg_path.parent / "rgb"
        if not poses.is_file():
            continue
        scene = cfg["scene"]
        sphere = next(o for o in scene["objects"] if o["name"] == "sphere")
        scenes.append(
            SceneRecord(
                scene_id=scene_id,
                config_path=cfg_path,
                poses_path=poses,
                rgb_root=rgb,
                restitution=float(scene["true_restitution"]),
                mass_kg=float(sphere["mass_kg"]),
                drop_z_m=float(sphere["initial_position_m"][2]),
            )
        )
    return scenes


def split_scenes(
    scenes: list[SceneRecord],
    *,
    val_restitution: list[float] | None = None,
    val_mass: list[float] | None = None,
    val_fraction: float = 0.15,
) -> tuple[list[SceneRecord], list[SceneRecord]]:
    val_e = set(val_restitution or [0.75])
    val_m = set(val_mass or [1.0])
    train, val = [], []
    for s in scenes:
        if s.restitution in val_e and s.mass_kg in val_m:
            val.append(s)
        else:
            train.append(s)
    if not val and len(scenes) >= 4:
        n_val = max(1, int(len(scenes) * val_fraction))
        val = scenes[-n_val:]
        train = scenes[:-n_val]
    return train, val


class VisualDynamicsDataset(Dataset):
    """One sample = predict state at frame t+1 from history ending at t."""

    def __init__(
        self,
        scenes: list[SceneRecord],
        *,
        history: int = 8,
        train_end_frame: int = 59,
        image_size: int = 64,
        camera_index: int = 0,
    ) -> None:
        self.history = history
        self.train_end_frame = train_end_frame
        self.image_size = image_size
        self.camera_index = camera_index
        self.samples: list[tuple[SceneRecord, int]] = []
        for scene in scenes:
            traj = load_object_poses_csv(scene.poses_path)
            states = trajectory_to_states(traj)
            for frame in range(history - 1, train_end_frame):
                self.samples.append((scene, frame))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_crop(self, scene: SceneRecord, frame: int) -> torch.Tensor:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("imageio required for visual dynamics dataset") from exc

        cam = f"cam{self.camera_index:02d}"
        path = scene.rgb_root / cam / f"frame{frame:05d}.png"
        mask_path = scene.rgb_root.parent / "masks" / cam / f"frame{frame:05d}.png"
        if not path.is_file():
            return torch.zeros(3, self.image_size, self.image_size)
        rgb = imageio.imread(path).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        if mask_path.is_file():
            mask = imageio.imread(mask_path).astype(np.float32) / 255.0
            if mask.ndim == 3:
                mask = mask[..., 0]
            rgb = rgb * mask[..., None]
        else:
            mask = np.ones((h, w), dtype=np.float32)
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            cy, cx = h // 2, w // 2
            r = min(h, w) // 4
        else:
            cy = int(0.5 * (ys.min() + ys.max()))
            cx = int(0.5 * (xs.min() + xs.max()))
            r = max(8, int(0.6 * max(ys.max() - ys.min(), xs.max() - xs.min())))
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        crop = rgb[y0:y1, x0:x1]
        t = torch.from_numpy(crop).permute(2, 0, 1)
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return t

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scene, frame = self.samples[index]
        traj = load_object_poses_csv(scene.poses_path)
        states = trajectory_to_states(traj)
        hist = window_states(states, frame, self.history)
        hist_vec = stack_state_vectors(hist, include_velocity=True)
        target = states[frame + 1]
        target_vec = stack_state_vectors([target], include_velocity=True)[0]
        crop = self._load_crop(scene, frame)
        return {
            "state_history": torch.from_numpy(hist_vec).float(),
            "target": torch.from_numpy(target_vec).float(),
            "image": crop,
            "scene_id": scene.scene_id,
            "frame": frame,
        }
