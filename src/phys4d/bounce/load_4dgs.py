"""Load Wu et al. 4DGaussians checkpoints for Step 4a trajectory extraction."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class LoadedWu4DGS:
    """Canonical Gaussians plus deformation network restored from a Wu training run."""

    gaussians: Any
    args: Namespace
    cfg_path: Path
    canonical_ply: Path
    deform_dir: Path
    device: str


def default_wu_root() -> Path:
    """Prefer env override, then the vendored submodule path from milestone3."""

    env = os.environ.get("WU_4DGS_ROOT")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "third_party" / "4DGaussians"


def _validate_wu_root(wu_root: Path) -> None:
    wu_root = wu_root.resolve()
    if (wu_root / "scene" / "gaussian_model.py").is_file():
        return
    raise FileNotFoundError(
        f"WU_4DGS_ROOT does not look like hustvl/4DGaussians: {wu_root}\n"
        "  git submodule add https://github.com/hustvl/4DGaussians third_party/4DGaussians\n"
        "  export WU_4DGS_ROOT=/path/to/4DGaussians"
    )


def _ensure_wu_on_path(wu_root: Path) -> None:
    wu_root = wu_root.resolve()
    _validate_wu_root(wu_root)
    root_str = str(wu_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ["WU_4DGS_ROOT"] = root_str


def load_cfg_args(cfg_args_path: Path) -> Namespace:
    """Parse Wu ``cfg_args`` (a repr of ``argparse.Namespace``)."""

    cfg_args_path = cfg_args_path.resolve()
    if not cfg_args_path.is_file():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_args_path}")

    text = cfg_args_path.read_text(encoding="utf-8")
    # Teammate wrappers may add a custom train-on-all-frames flag; fail before any eval.
    compact = text.replace(" ", "")
    if "all_train=True" in compact or "all-train" in text.lower():
        raise RuntimeError(
            f"{cfg_args_path} indicates --all-train was used. "
            "Retrain on train frames only before held-out evaluation."
        )

    try:
        ns = eval(text, {"__builtins__": {}}, {"Namespace": Namespace})  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Could not parse cfg_args at {cfg_args_path}: {exc}") from exc
    if not isinstance(ns, Namespace):
        raise ValueError(f"cfg_args did not evaluate to Namespace: {type(ns)}")
    return ns


def _deform_checkpoint_dir(deform_path: Path) -> Path:
    """Wu saves ``deformation.pth`` inside ``point_cloud/iteration_*``."""

    deform_path = deform_path.resolve()
    if deform_path.is_dir():
        return deform_path
    if deform_path.name == "deformation.pth" and deform_path.is_file():
        return deform_path.parent
    raise FileNotFoundError(f"Expected deformation.pth or its parent dir: {deform_path}")


def load_wu_4dgs(
    *,
    canonical_ply: Path,
    deform_path: Path,
    cfg_args_path: Path,
    wu_root: Path | None = None,
    device: str = "cuda",
    sh_degree: int = 3,
) -> LoadedWu4DGS:
    """Instantiate ``GaussianModel``, load PLY + deformation weights."""

    canonical_ply = canonical_ply.resolve()
    deform_dir = _deform_checkpoint_dir(deform_path)
    cfg_args_path = cfg_args_path.resolve()

    if not canonical_ply.is_file():
        raise FileNotFoundError(f"Missing canonical PLY: {canonical_ply}")
    if not (deform_dir / "deformation.pth").is_file():
        raise FileNotFoundError(f"Missing {deform_dir / 'deformation.pth'}")

    wu = Path(wu_root or default_wu_root())
    _ensure_wu_on_path(wu)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass device='cpu'")

    from scene.gaussian_model import GaussianModel

    args = load_cfg_args(cfg_args_path)
    gaussians = GaussianModel(sh_degree, args)

    map_location = "cuda" if device == "cuda" else "cpu"
    gaussians.load_ply(str(canonical_ply))
    gaussians.load_model(str(deform_dir))
    gaussians._deformation.eval()

    return LoadedWu4DGS(
        gaussians=gaussians,
        args=args,
        cfg_path=cfg_args_path,
        canonical_ply=canonical_ply,
        deform_dir=deform_dir,
        device=device,
    )


def deform_positions_at_norm_time(
    loaded: LoadedWu4DGS,
    t_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deform canonical xyz at normalized time in [0, 1]; return xyz and opacity weights."""

    g = loaded.gaussians
    with torch.no_grad():
        mu0 = g.get_xyz.detach()
        scales = g._scaling
        rotations = g._rotation
        opacity = g._opacity
        shs = g.get_features
        time_input = torch.full((mu0.shape[0], 1), float(t_norm), device=mu0.device, dtype=mu0.dtype)
        means3D_final, _, _, _, _ = g._deformation(
            mu0,
            scales,
            rotations,
            opacity,
            shs,
            time_input,
        )
        weights = g.get_opacity.detach().squeeze(-1)

    xyz = means3D_final.cpu().numpy()
    w = weights.cpu().numpy()
    return xyz, w


def opacity_weighted_centroid(xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """3D center of mass using activated opacities (Wu ``get_opacity`` path)."""

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    if pts.shape[0] == 0:
        raise ValueError("No Gaussians for centroid")
    total = float(w.sum())
    if total < 1e-12:
        return pts.mean(axis=0)
    return (pts * w[:, None]).sum(axis=0) / total
