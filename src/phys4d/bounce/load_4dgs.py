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
    """Prefer env override, then the vendored submodule path."""

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


def geometric_median(
    xyz: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    n_iter: int = 64,
    tol: float = 1e-9,
) -> np.ndarray:
    """Weighted geometric median (Weiszfeld). Parameter-free, robust to residual spill.

    Minimizes the sum of (optionally opacity-weighted) Euclidean distances, which is far
    less sensitive to a handful of far outliers than the mean centroid — and needs no
    distance threshold, so it stays scene-agnostic.
    """

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("No Gaussians for geometric median")
    if weights is None:
        w = np.ones(pts.shape[0], dtype=np.float64)
    else:
        w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    if float(w.sum()) < 1e-12:
        w = np.ones(pts.shape[0], dtype=np.float64)

    center = (pts * w[:, None]).sum(axis=0) / w.sum()
    for _ in range(int(n_iter)):
        d = np.linalg.norm(pts - center, axis=1)
        near = d < 1e-12
        if np.any(near):
            return pts[near][0].copy()
        ww = w / d
        new_center = (pts * ww[:, None]).sum(axis=0) / ww.sum()
        if float(np.linalg.norm(new_center - center)) < tol:
            center = new_center
            break
        center = new_center
    return center


def select_object_gaussian_set(
    xyz: np.ndarray,
    weights: np.ndarray,
    *,
    k_neighbors: int = 8,
    eps_scale: float = 3.0,
    min_size: int = 16,
) -> np.ndarray:
    """Pick the dense object cluster as a *fixed* Gaussian index set (scale-invariant).

    Object-only Wu exports carry high-opacity "spill" Gaussians scattered far from the
    object core. Opacity can't separate them (spill is opaque too), but *density* can: the
    ball is a tight dense cluster while spill is sparse. We build a neighbor graph whose
    linking radius is ``eps_scale * median(k-th NN distance)`` — derived from the data, so
    it auto-adapts to ball size / Gaussian density — and return the connected component
    with the largest total opacity (the object). No absolute thresholds, no per-scene knobs.

    Returns a boolean mask over the input Gaussians (consistent index order across time).
    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    n = pts.shape[0]
    if n == 0:
        raise ValueError("No Gaussians to cluster")
    if n <= min_size:
        return np.ones(n, dtype=bool)

    tree = cKDTree(pts)
    kk = int(min(max(2, k_neighbors), n - 1))
    knn_dist, _ = tree.query(pts, k=kk + 1)  # col 0 is self (dist 0)
    kth = knn_dist[:, -1]
    eps = float(eps_scale * np.median(kth))
    if not np.isfinite(eps) or eps <= 0:
        eps = float(np.median(kth[kth > 0])) if np.any(kth > 0) else 1.0

    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    if pairs.shape[0] == 0:
        # everything sparse: fall back to the single highest-opacity point neighborhood
        seed = int(np.argmax(w))
        d = np.linalg.norm(pts - pts[seed], axis=1)
        return d <= eps

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)

    # pick the component with the largest total opacity weight
    best_label, best_w = -1, -1.0
    for c in range(n_comp):
        m = labels == c
        if int(m.sum()) < min_size:
            continue
        tw = float(w[m].sum())
        if tw > best_w:
            best_w, best_label = tw, c
    if best_label < 0:
        # no component met min_size; take the largest by count
        counts = np.bincount(labels)
        best_label = int(np.argmax(counts))
    return labels == best_label


def robust_opacity_weighted_centroid(
    xyz: np.ndarray,
    weights: np.ndarray,
    *,
    n_iter: int = 8,
    k_mad: float = 3.0,
    min_inliers: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Opacity-weighted centroid with iterative spatial-outlier rejection.

    Object-only Wu exports can carry high-opacity "spill" Gaussians scattered far from
    the object core (seen out to ~200 units while the ball core spans <1 unit). Pure
    opacity weighting can't drop them (the spill is high-opacity too), so we iteratively
    keep Gaussians within ``median + k_mad * MAD`` of the running weighted centroid.

    Returns ``(centroid, inlier_mask)``.
    """

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    if pts.shape[0] == 0:
        raise ValueError("No Gaussians for centroid")

    mask = np.ones(pts.shape[0], dtype=bool)
    center = opacity_weighted_centroid(pts, w)
    for _ in range(max(1, int(n_iter))):
        dist = np.linalg.norm(pts - center, axis=1)
        med = float(np.median(dist[mask]))
        mad = float(np.median(np.abs(dist[mask] - med))) * 1.4826 + 1e-9
        new_mask = dist <= med + float(k_mad) * mad
        if int(new_mask.sum()) < int(min_inliers):
            break
        mask = new_mask
        center = opacity_weighted_centroid(pts[mask], w[mask])
    return center, mask
