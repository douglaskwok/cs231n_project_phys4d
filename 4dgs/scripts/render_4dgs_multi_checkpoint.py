#!/usr/bin/env python
"""Render multiple separately trained 4DGS checkpoints into one raster sequence.

This is for object-wise 4DGS models that share the same world coordinate system
but have separate deformation fields. Each checkpoint is restored into its own
GaussianModel, rendered at the same camera/time, and composited on a black
background. Optional frame gates keep objects black outside their original
visible time range.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fourdgs_common as fc

FRAME_RE = re.compile(r"cam(\d+)_(\d+)")


def _frame_id(image_name: str) -> int:
    match = FRAME_RE.search(image_name)
    if not match:
        return -1
    return int(match.group(2))


def _parse_ints(text: str, n: int, default: int) -> list[int]:
    if not text:
        return [default] * n
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(vals)}: {text}")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourd-root", default=os.environ.get("FOURDGS_ROOT", "/opt/4dgs"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-starts", default="")
    parser.add_argument("--frame-ends", default="")
    parser.add_argument("--dry-run-max", type=int, default=0)
    parser.add_argument("--mode", choices=("max", "over"), default="max")
    args = parser.parse_args()

    fourd = Path(args.fourd_root).resolve()
    os.environ["FOURDGS_ROOT"] = str(fourd)
    if str(fourd) not in sys.path:
        sys.path.insert(0, str(fourd))

    import torch
    from gaussian_renderer import render
    from scene.dataset_readers import readCamerasFromTransforms
    from torchvision.utils import save_image
    from tqdm import tqdm
    from utils.camera_utils import cameraList_from_camInfos

    args_ns, lp, _, pp = fc.merge_config_yaml(args.config.resolve())
    dataset = args.dataset.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    starts = _parse_ints(args.frame_starts, len(args.checkpoints), -10**9)
    ends = _parse_ints(args.frame_ends, len(args.checkpoints), 10**9)

    all_infos: list = []
    for tf_name in ("transforms_train.json", "transforms_test.json"):
        if not (dataset / tf_name).is_file():
            continue
        infos = readCamerasFromTransforms(
            str(dataset),
            tf_name,
            lp.white_background,
            lp.extension,
            time_duration=None,
            frame_ratio=lp.frame_ratio,
            dataloader=lp.dataloader,
        )
        all_infos.extend(infos)

    all_infos.sort(key=lambda ci: (float(ci.timestamp), fc.cam_sort_key(ci.image_name)))
    if args.dry_run_max > 0:
        all_infos = all_infos[: args.dry_run_max]
    if not all_infos:
        raise FileNotFoundError(f"No camera viewpoints under {dataset}")

    camera_stack = cameraList_from_camInfos(all_infos, resolution_scale=1.0, args=lp)

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    gaussians = []
    checkpoint_paths = [p.resolve() for p in args.checkpoints]
    for ckpt in checkpoint_paths:
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        model = fc.instantiate_gaussian_model(args_ns, lp, pp)
        blob = torch.load(str(ckpt), map_location="cuda")
        if not isinstance(blob, (tuple, list)) or len(blob) != 2:
            raise ValueError(f"Bad checkpoint tuple: {ckpt}")
        model.restore(blob[0], None)
        gaussians.append(model)

    png_paths: list[str] = []
    active_counts = [0] * len(gaussians)
    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(camera_stack, desc="Rendering multi-4DGS")):
            v = viewpoint.cuda()
            fid = _frame_id(v.image_name)
            out = torch.zeros((3, v.image_height, v.image_width), device="cuda")
            for model_idx, model in enumerate(gaussians):
                if fid < starts[model_idx] or fid > ends[model_idx]:
                    continue
                active_counts[model_idx] += 1
                img = torch.clamp(render(v, model, pp, bg)["render"], 0.0, 1.0)
                if args.mode == "max":
                    out = torch.maximum(out, img)
                else:
                    mask = img.amax(dim=0, keepdim=True) > 8.0 / 255.0
                    out = torch.where(mask, img, out)

            stem = f"{idx:06d}_{v.image_name.replace('/', '_').replace('.', '_')}"
            pfp = out_dir / f"{stem}.png"
            save_image(torch.clamp(out, 0.0, 1.0), str(pfp))
            png_paths.append(str(pfp))

    meta = {
        "num_frames": len(png_paths),
        "mode": args.mode,
        "dataset": str(dataset),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "frame_starts": starts,
        "frame_ends": ends,
        "active_render_counts": active_counts,
    }
    (out_dir / "render_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"{len(png_paths)} PNGs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
