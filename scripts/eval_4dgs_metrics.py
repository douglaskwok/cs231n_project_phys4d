#!/usr/bin/env python
"""PSNR / MAE vs ground-truth RGB for a trained fudan 4DGS checkpoint.

Loads cameras with **full RGB** (``dataloader`` forced off) so GT matches training images.
Reports separate aggregates for ``transforms_train.json`` and ``transforms_test.json``.

Needs CUDA + fudan clone::

    export FOURDGS_ROOT=/path/to/4d-gaussian-splatting
    python scripts/eval_4dgs_metrics.py \\
      --config configs/sphere_bounce_4dgs.yaml \\
      --dataset outputs/.../dynerf_sphere_bounce \\
      --checkpoint 4dgs_sphere_bounce/chkpnt15000.pth \\
      --out-json metrics_4dgs.json

Modal::

    modal run modal_app.py --eval-4d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fourdgs_common as fc


def _psnr_mse(pred: "torch.Tensor", tgt: "torch.Tensor") -> tuple[float, float]:
    import math
    import torch

    mse = torch.mean((pred - tgt) ** 2).item()
    if mse < 1e-12:
        return 99.0, 0.0
    psnr = 10.0 * math.log10(1.0 / mse)
    return psnr, mse


def _eval_split(
    dataset: Path,
    transforms_name: str,
    *,
    gaussians,
    pp,
    lp_load,
    bg,
    dry_max: int,
) -> dict:
    import torch
    from gaussian_renderer import render
    from scene.dataset_readers import readCamerasFromTransforms
    from tqdm import tqdm
    from utils.camera_utils import cameraList_from_camInfos

    infos = readCamerasFromTransforms(
        str(dataset),
        transforms_name,
        lp_load.white_background,
        lp_load.extension,
        time_duration=None,
        frame_ratio=lp_load.frame_ratio,
        dataloader=False,
    )
    infos.sort(key=lambda ci: (float(ci.timestamp), fc.cam_sort_key(ci.image_name)))
    if dry_max > 0:
        infos = infos[:dry_max]

    if not infos:
        return {"n": 0}

    sum_abs = 0.0
    sum_sq = 0.0
    sum_psnr = 0.0

    with torch.no_grad():
        for ci in tqdm(infos, desc=f"eval:{transforms_name}"):
            cams = cameraList_from_camInfos([ci], resolution_scale=1.0, args=lp_load)
            v = cams[0].cuda()
            rp = render(v, gaussians, pp, bg)
            pred = torch.clamp(rp["render"], 0.0, 1.0)
            gt = v.image
            if pred.shape != gt.shape:
                raise RuntimeError(f"Shape mismatch pred {pred.shape} gt {gt.shape}")
            pred = pred.cuda()
            gt = gt.cuda()

            psnr, mse = _psnr_mse(pred, gt)
            sum_psnr += psnr
            sum_sq += mse
            sum_abs += float(torch.mean(torch.abs(pred - gt)).item())

    n = len(infos)
    return {
        "n": n,
        "psnr_mean": sum_psnr / n,
        "mse_mean": sum_sq / n,
        "mae_mean": sum_abs / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourd-root", default=os.environ.get("FOURDGS_ROOT", "/opt/4dgs"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("metrics_4dgs.json"))
    parser.add_argument("--dry-run-max", type=int, default=0)
    cli = parser.parse_args()

    fourd = Path(cli.fourd_root).resolve()
    os.environ["FOURDGS_ROOT"] = str(fourd)
    if str(fourd) not in sys.path:
        sys.path.insert(0, str(fourd))

    import torch

    args_ns, lp, _, pp = fc.merge_config_yaml(Path(cli.config).resolve())
    ckpt_path = cli.checkpoint.resolve()
    dataset = cli.dataset.resolve()

    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        return 1

    lp_load = SimpleNamespace(**{**vars(lp), "dataloader": False})

    bg = torch.tensor(
        [1.0, 1.0, 1.0] if lp.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    gaussians = fc.instantiate_gaussian_model(args_ns, lp, pp)
    blob = torch.load(str(ckpt_path), map_location="cuda")
    if not isinstance(blob, (tuple, list)) or len(blob) != 2:
        print("Bad checkpoint tuple.", file=sys.stderr)
        return 1
    gaussians.restore(blob[0], None)

    results: dict = {
        "checkpoint": str(ckpt_path),
        "dataset": str(dataset),
        "config": str(Path(cli.config).resolve()),
    }
    splits_seen = []
    for tf in ("transforms_train.json", "transforms_test.json"):
        if not (dataset / tf).is_file():
            continue
        splits_seen.append(tf)
        tag = "train" if "train" in tf else "test"
        results[tag] = _eval_split(
            dataset,
            tf,
            gaussians=gaussians,
            pp=pp,
            lp_load=lp_load,
            bg=bg,
            dry_max=cli.dry_run_max,
        )

    if not splits_seen:
        print("No transforms_train/test.json found.", file=sys.stderr)
        return 1

    n_tot = 0.0
    w_psnr = 0.0
    w_mse = 0.0
    w_mae = 0.0
    for tag in ("train", "test"):
        block = results.get(tag)
        if not block or block.get("n", 0) == 0:
            continue
        n = block["n"]
        n_tot += n
        w_psnr += block["psnr_mean"] * n
        w_mse += block["mse_mean"] * n
        w_mae += block["mae_mean"] * n

    if n_tot > 0:
        results["combined"] = {
            "n": int(n_tot),
            "psnr_mean": w_psnr / n_tot,
            "mse_mean": w_mse / n_tot,
            "mae_mean": w_mae / n_tot,
        }

    out_path = cli.out_json.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
