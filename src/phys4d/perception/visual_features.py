"""t=0 visual features per object (fixed for rollout, per project.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _list_cam_dirs(rgb_root: Path, max_views: int) -> list[str]:
    cams = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())
    return cams[:max_views]


def load_masked_crop(
    rgb_path: Path,
    mask_path: Path,
    image_size: int,
) -> torch.Tensor:
    import imageio.v2 as imageio
    import torch.nn.functional as F

    if not rgb_path.is_file():
        return torch.zeros(3, image_size, image_size)
    rgb = imageio.imread(rgb_path).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    if mask_path.is_file():
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        rgb = rgb * (mask > 127).astype(np.float32)[..., None]
    else:
        mask = np.ones((h, w), dtype=np.float32)
    ys, xs = np.where(mask > 127 if mask_path.is_file() else mask > 0.5)
    if len(xs) == 0:
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
    else:
        cy = int(0.5 * (ys.min() + ys.max()))
        cx = int(0.5 * (xs.min() + xs.max()))
        r = max(8, int(0.55 * max(ys.max() - ys.min(), xs.max() - xs.min())))
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    crop = rgb[y0:y1, x0:x1]
    t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)


def extract_t0_visual_feature(
    scene_dir: Path,
    *,
    frame: int = 0,
    image_size: int = 128,
    max_views: int = 10,
    encoder: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Multi-view mean-pooled ResNet embedding at t=0."""
    from phys4d.visual_dynamics.feature_encoder import build_feature_encoder

    rgb_root = scene_dir / "rgb"
    mask_root = scene_dir / "masks"
    enc = encoder or build_feature_encoder(out_dim=64)
    enc.eval()
    device = torch.device(device)
    enc.to(device)

    feats: list[torch.Tensor] = []
    for cam in _list_cam_dirs(rgb_root, max_views):
        rgb_path = rgb_root / cam / f"frame{frame:05d}.png"
        mask_path = mask_root / cam / f"frame{frame:05d}.png"
        crop = load_masked_crop(rgb_path, mask_path, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            feats.append(enc(crop))
    if not feats:
        return np.zeros(64, dtype=np.float32)
    pooled = torch.stack(feats, dim=0).mean(dim=0).squeeze(0).cpu().numpy()
    return pooled.astype(np.float32)
