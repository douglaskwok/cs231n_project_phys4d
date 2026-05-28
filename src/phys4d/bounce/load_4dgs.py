"""Load fudan 4D Gaussian Splatting checkpoints for trajectory extraction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Loaded4DGS:
    """Restored ``GaussianModel`` plus training metadata from the YAML config."""

    gaussians: Any
    time_duration: tuple[float, float]
    config_path: Path
    checkpoint_path: Path
    device: str


def _validate_fourdgs_root(fourd_root: Path) -> None:
    """Fail early with setup instructions if the Fudan clone is missing."""

    fourd_root = fourd_root.resolve()
    args_pkg = fourd_root / "arguments"
    scene_pkg = fourd_root / "scene"
    if args_pkg.is_dir() and scene_pkg.is_dir():
        return
    raise FileNotFoundError(
        f"FOURDGS_ROOT does not look like fudan-zvg/4d-gaussian-splatting: {fourd_root}\n"
        "Clone once (sibling to this repo is fine):\n"
        "  git clone --recursive https://github.com/fudan-zvg/4d-gaussian-splatting.git\n"
        "  export FOURDGS_ROOT=/path/to/4d-gaussian-splatting\n"
        "On Modal, training/render already use /opt/4dgs; Phase 1 locally needs the same clone."
    )


def _ensure_fourdgs_on_path(fourd_root: Path) -> None:
    fourd_root = fourd_root.resolve()
    _validate_fourdgs_root(fourd_root)
    os.environ["FOURDGS_ROOT"] = str(fourd_root)
    root_str = str(fourd_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_4dgs_checkpoint(
    *,
    checkpoint: Path,
    config: Path,
    fourd_root: Path | None = None,
    device: str = "cuda",
) -> Loaded4DGS:
    """Instantiate fudan ``GaussianModel``, restore ``chkpnt*.pth``, move to ``device``."""

    checkpoint = checkpoint.resolve()
    config = config.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Missing config YAML: {config}")

    fourd = Path(fourd_root or os.environ.get("FOURDGS_ROOT", "/opt/4dgs"))
    _ensure_fourdgs_on_path(fourd)

    # Repo helper lives next to other 4DGS scripts.
    scripts_dir = Path(__file__).resolve().parents[3] / "4dgs" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import fourdgs_common as fc
    import torch

    args_ns, lp, _op, pp = fc.merge_config_yaml(config)
    gaussians = fc.instantiate_gaussian_model(args_ns, lp, pp)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass device='cpu'")
    map_location = "cuda" if device == "cuda" else "cpu"
    blob = torch.load(str(checkpoint), map_location=map_location)
    if not isinstance(blob, (tuple, list)) or len(blob) != 2:
        raise ValueError(f"Unexpected checkpoint tuple in {checkpoint}")

    gaussians.restore(blob[0], None)

    td = tuple(fc.model_params_override_time_duration(tuple(args_ns.time_duration), lp))
    return Loaded4DGS(
        gaussians=gaussians,
        time_duration=(float(td[0]), float(td[1])),
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
    )


def deformed_gaussians_at_time(
    loaded: Loaded4DGS,
    timestamp_s: float,
    *,
    marginal_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deformed xyz (N,3) and per-Gaussian weights at ``timestamp_s``.

  Weights are ``sigmoid(opacity) * marginal_t(t)`` for 4D models (same gating as
  the fudan renderer when ``compute_cov3D_python`` is enabled). Positions use the
  conditional mean offset: ``get_xyz + delta_mean(timestamp)``.
    """

    import torch

    gaussians = loaded.gaussians
    device = loaded.device
    ts = float(timestamp_s)

    with torch.no_grad():
        _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, ts)
        xyz = gaussians.get_xyz + delta_mean
        opacity = gaussians.get_opacity.squeeze(-1)
        if gaussians.gaussian_dim == 4:
            marginal = gaussians.get_marginal_t(ts).squeeze(-1)
            weights = opacity * marginal
            keep = marginal >= float(marginal_threshold)
            if torch.any(keep):
                xyz = xyz[keep]
                weights = weights[keep]
        else:
            weights = opacity

    return xyz.detach().cpu().numpy(), weights.detach().cpu().numpy()
