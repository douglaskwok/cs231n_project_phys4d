"""Load trained param predictor and run on a scene directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phys4d.param_ident.dataset import PREDICT_NAMES
from phys4d.param_ident.model import MultiViewParamPredictor

DEFAULT_CLIP_KWARGS = {
    "num_frames": 16,
    "frame_stride": 2,
    "image_size": 128,
    "max_views": 10,
    "train_end_frame": 59,
}


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[MultiViewParamPredictor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiViewParamPredictor(num_params=len(PREDICT_NAMES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_scene_clip(
    scene_dir: Path,
    *,
    num_frames: int = 16,
    frame_stride: int = 2,
    image_size: int = 128,
    max_views: int = 10,
    train_end_frame: int = 59,
    start_frame: int = 0,
) -> torch.Tensor:
    """Build (T, V, 3, H, W) tensor from ``rgb/`` and ``masks/``."""
    import imageio.v2 as imageio
    import torch.nn.functional as F

    rgb_root = scene_dir / "rgb"
    mask_root = scene_dir / "masks"
    cams = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())[:max_views]
    frames = [start_frame + i * frame_stride for i in range(num_frames)]
    frames = [f for f in frames if f <= train_end_frame]

    clips: list[np.ndarray] = []
    for fi in frames:
        view_tensors = []
        for cam in cams:
            rgb_path = rgb_root / cam / f"frame{fi:05d}.png"
            mask_path = mask_root / cam / f"frame{fi:05d}.png"
            if not rgb_path.is_file():
                view_tensors.append(np.zeros((3, image_size, image_size), dtype=np.float32))
                continue
            rgb = imageio.imread(rgb_path).astype(np.float32) / 255.0
            if mask_path.is_file():
                mask = imageio.imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                rgb = rgb * (mask > 127).astype(np.float32)[..., None]
            h, w = rgb.shape[:2]
            y0, x0 = max(0, (h - image_size) // 2), max(0, (w - image_size) // 2)
            crop = rgb[y0 : y0 + image_size, x0 : x0 + image_size]
            if crop.shape[0] != image_size or crop.shape[1] != image_size:
                t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
                t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
                crop = t.squeeze(0).permute(1, 2, 0).numpy()
            view_tensors.append(crop.transpose(2, 0, 1))
        clips.append(np.stack(view_tensors, axis=0))
    return torch.from_numpy(np.stack(clips, axis=0)).float()


@torch.no_grad()
def predict_params(
    model: MultiViewParamPredictor,
    clip: torch.Tensor,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    x = clip.unsqueeze(0).to(device)
    pred = model(x).squeeze(0).cpu().numpy()
    return {name: float(pred[i]) for i, name in enumerate(PREDICT_NAMES)}


def predict_scene(
    scene_dir: Path,
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
    clip_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model, ckpt = load_checkpoint(checkpoint, device=device)
    kw = {**DEFAULT_CLIP_KWARGS, **(clip_kwargs or {})}
    clip = load_scene_clip(scene_dir, **kw)
    predicted = predict_params(model, clip, device=device)
    gt: dict[str, float] | None = None
    physics_path = scene_dir / "physics_params.json"
    if physics_path.is_file():
        with physics_path.open("r", encoding="utf-8") as f:
            physics = json.load(f)
        names = physics.get("predict_v1", PREDICT_NAMES)
        idx = {n: i for i, n in enumerate(physics["param_names"])}
        gt = {n: float(physics["param_vector"][idx[n]]) for n in names if n in idx}
    return {
        "scene_dir": str(scene_dir),
        "predicted": predicted,
        "ground_truth": gt,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
    }
