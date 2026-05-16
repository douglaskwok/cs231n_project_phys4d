"""Dataset: multi-view video clips → GT physics parameter vector."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PREDICT_NAMES = ["restitution", "mass_kg", "drop_z_m"]


@dataclass(frozen=True)
class SceneSample:
    scene_id: str
    scene_dir: Path
    param_vector: np.ndarray


def resolve_scene_dir(data_root: Path, manifest: dict, scene_id: str) -> Path:
    batch_root = Path(manifest["batch_root"])
    return (data_root / batch_root / scene_id).resolve()


def load_manifest(
    manifest_path: Path,
    split: str,
    *,
    data_root: Path | None = None,
) -> list[SceneSample]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    root = (data_root or (manifest_path.parent.parent)).resolve()
    rows = data["splits"][split]
    samples: list[SceneSample] = []
    for row in rows:
        scene_id = row["scene_id"]
        if "scene_dir" in row:
            scene_dir = (root / row["scene_dir"]).resolve()
        else:
            scene_dir = resolve_scene_dir(root, data, scene_id)
        samples.append(
            SceneSample(
                scene_id=scene_id,
                scene_dir=scene_dir,
                param_vector=np.array(row["param_vector"], dtype=np.float32),
            )
        )
    return samples


class ParamIdDataset(Dataset):
    def __init__(
        self,
        scenes: list[SceneSample],
        *,
        num_frames: int = 16,
        frame_stride: int = 2,
        image_size: int = 128,
        min_views: int = 4,
        max_views: int = 10,
        augment: bool = False,
    ) -> None:
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.min_views = min_views
        self.max_views = max_views
        self.augment = augment
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_rgb_mask(self, rgb_path: Path, mask_path: Path) -> np.ndarray:
        import imageio.v2 as imageio

        rgb = imageio.imread(rgb_path).astype(np.float32) / 255.0
        if mask_path.is_file():
            mask = imageio.imread(mask_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 127).astype(np.float32)[..., None]
            rgb = rgb * mask
        h, w = rgb.shape[:2]
        size = self.image_size
        y0 = max(0, (h - size) // 2)
        x0 = max(0, (w - size) // 2)
        crop = rgb[y0 : y0 + size, x0 : x0 + size]
        if crop.shape[0] != size or crop.shape[1] != size:
            import torch.nn.functional as F

            t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            crop = t.squeeze(0).permute(1, 2, 0).numpy()
        if self.augment:
            scale = random.uniform(0.85, 1.15)
            crop = np.clip(crop * scale, 0.0, 1.0)
        return crop.transpose(2, 0, 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene = self.scenes[index]
        root = scene.scene_dir.resolve()
        rgb_root = root / "rgb"
        mask_root = root / "masks"
        cams = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())
        n_views = random.randint(self.min_views, min(self.max_views, len(cams))) if self.augment else len(cams)
        chosen = sorted(random.sample(cams, n_views)) if self.augment else cams[:n_views]

        max_frame = 59
        start = random.randint(0, max(0, max_frame - self.num_frames * self.frame_stride)) if self.augment else 0
        frames = [start + i * self.frame_stride for i in range(self.num_frames)]

        clips: list[np.ndarray] = []
        for fi in frames:
            view_tensors = []
            for cam in chosen:
                rgb_path = rgb_root / cam / f"frame{fi:05d}.png"
                mask_path = mask_root / cam / f"frame{fi:05d}.png"
                if not rgb_path.is_file():
                    view_tensors.append(np.zeros((3, self.image_size, self.image_size), dtype=np.float32))
                else:
                    view_tensors.append(self._load_rgb_mask(rgb_path, mask_path))
            clips.append(np.stack(view_tensors, axis=0))

        x = torch.from_numpy(np.stack(clips, axis=0)).float()
        y = torch.from_numpy(scene.param_vector.copy()).float()
        return x, y
