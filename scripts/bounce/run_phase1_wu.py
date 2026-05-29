#!/usr/bin/env python
"""Phase 1 (Wu 4DGaussians backend) — extract the ball trajectory from a Wu checkpoint.

The bounce pipeline spec targets the Fudan 4DGS backend, but the
``wu_debug_ball_huge_120fps_0p5s`` scene was trained with Wu et al. 4DGaussians
(canonical ``point_cloud.ply`` + a HexPlane/MLP deformation field saved as
``deformation.pth``). This script reproduces Phase 1 for that backend.

It is pure-torch and runs on CPU: we read the canonical Gaussian means + opacity
from the PLY, instantiate ONLY the deformation network (no CUDA rasterizer), then
evaluate the deformed means at each frame's normalized time to get an
opacity-weighted centroid. Outputs match the standard Phase 1 contract so the
remaining phases run unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.extract import (  # noqa: E402
    plot_trajectory,
    procrustes_align,
    smooth_trajectory_savgol,
    write_trajectory_csv,
)
from phys4d.poses import load_object_poses_csv  # noqa: E402


def _import_wu_deform_network(fourd_root: Path):
    """Import Wu's ``deform_network`` without triggering the CUDA-heavy scene package.

    ``scene/__init__.py`` imports the Gaussian rasterizer (``simple_knn._C``) which
    is unavailable on a CPU/Mac box. We register a lightweight stub ``scene`` package
    so the pure-torch submodules (hexplane, grid, deformation) import in isolation.
    """

    fourd_root = fourd_root.resolve()
    if not (fourd_root / "scene" / "deformation.py").is_file():
        raise FileNotFoundError(
            f"Not a 4DGaussians clone (missing scene/deformation.py): {fourd_root}"
        )
    if str(fourd_root) not in sys.path:
        sys.path.insert(0, str(fourd_root))

    # Stub package so `import scene.deformation` skips scene/__init__.py side effects.
    if "scene" not in sys.modules:
        scene_pkg = types.ModuleType("scene")
        scene_pkg.__path__ = [str(fourd_root / "scene")]
        sys.modules["scene"] = scene_pkg

    import importlib

    importlib.import_module("scene.hexplane")
    importlib.import_module("scene.grid")
    deformation = importlib.import_module("scene.deformation")
    return deformation.deform_network


def _hidden_args_from_cfg(cfg_args_path: Path) -> types.SimpleNamespace:
    """Reconstruct the ModelHiddenParams needed by ``deform_network`` from cfg_args.

    ``cfg_args`` is a stringified ``argparse.Namespace(...)``; we eval it with a stub
    Namespace and pull the deformation-relevant fields out.
    """

    text = cfg_args_path.read_text(encoding="utf-8").strip()
    ns = eval(text, {"Namespace": lambda **kw: types.SimpleNamespace(**kw)})  # noqa: S307
    keys = (
        "net_width",
        "timebase_pe",
        "defor_depth",
        "posebase_pe",
        "scale_rotation_pe",
        "opacity_pe",
        "timenet_width",
        "timenet_output",
        "grid_pe",
        "bounds",
        "kplanes_config",
        "multires",
        "no_grid",
        "no_dx",
        "no_ds",
        "no_dr",
        "no_do",
        "no_dshs",
        "empty_voxel",
        "static_mlp",
        "apply_rotation",
    )
    return types.SimpleNamespace(**{k: getattr(ns, k) for k in keys})


def _read_ply_means_opacity(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read canonical Gaussian means (N,3) and raw opacity logits (N,) from a PLY."""

    from plyfile import PlyData

    el = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack(
        [np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1
    ).astype(np.float64)
    opacity = np.asarray(el["opacity"]).astype(np.float64)
    return xyz, opacity


def _deformed_centroid(
    deform_net,
    xyz_t: "object",
    scales_t: "object",
    rots_t: "object",
    weights: np.ndarray,
    t_norm: float,
):
    """Query the deformation field at ``t_norm`` and return opacity-weighted centroid."""

    import torch

    n = xyz_t.shape[0]
    times = torch.full((n, 1), float(t_norm), dtype=torch.float32)
    # Deformation forward references opacity for the (unused) static mask shape even
    # though no_do=True leaves opacity untouched; pass a placeholder (N,1) tensor.
    opacity_t = torch.zeros((n, 1), dtype=torch.float32)
    with torch.no_grad():
        means, _scales, _rots, _opacity, _shs = deform_net(
            xyz_t, scales_t, rots_t, opacity_t, None, times
        )
    pts = means.detach().cpu().numpy().astype(np.float64)
    w = np.clip(weights, 0.0, None)
    total = float(w.sum())
    if total < 1e-12:
        return pts.mean(axis=0)
    return (pts * w[:, None]).sum(axis=0) / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Wu model dir containing point_cloud/iteration_*/ and cfg_args.",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        required=True,
        help="Scene config.json (provides num_frames, fps, train/test split).",
    )
    parser.add_argument("--gt-poses", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fourd-root",
        type=Path,
        default=_REPO_ROOT / "third_party" / "4DGaussians",
    )
    parser.add_argument("--iteration", type=int, default=None, help="Which iteration_* to load.")
    parser.add_argument("--savgol-window", type=int, default=7)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument(
        "--train-frames-only",
        action="store_true",
        help="Only extract config train_frames (default extracts all frames).",
    )
    parser.add_argument("--frame-start", type=int, default=None, help="Override first frame to extract.")
    parser.add_argument("--frame-end", type=int, default=None, help="Override last frame (inclusive).")
    parser.add_argument(
        "--align-to-gt",
        action="store_true",
        help=(
            "Register the 4DGS world into the PyBullet/sim world by a single constant "
            "translation = mean(GT - raw) over the extracted window. Requires --gt-poses. "
            "This only fixes the arbitrary GS world origin; per-frame motion is untouched."
        ),
    )
    args = parser.parse_args()

    import torch

    model_dir = args.model_dir.resolve()
    pc_root = model_dir / "point_cloud"
    iters = sorted(int(p.name.split("_")[-1]) for p in pc_root.glob("iteration_*"))
    if not iters:
        raise FileNotFoundError(f"No point_cloud/iteration_* under {model_dir}")
    iteration = int(args.iteration) if args.iteration is not None else iters[-1]
    iter_dir = pc_root / f"iteration_{iteration}"
    ply_path = iter_dir / "point_cloud.ply"
    deform_path = iter_dir / "deformation.pth"

    cfg = json.loads(args.scene_config.read_text(encoding="utf-8"))
    sim = cfg.get("simulation", {})
    num_frames = int(sim.get("num_frames", 61))
    duration_s = float(sim.get("duration_s", 0.5))
    fps = (num_frames - 1) / duration_s if duration_s > 0 else 120.0
    train_lo, train_hi = sim.get("train_frames", [0, num_frames - 1])

    if args.frame_start is not None or args.frame_end is not None:
        lo = int(args.frame_start) if args.frame_start is not None else 0
        hi = int(args.frame_end) if args.frame_end is not None else num_frames - 1
        frame_ids = list(range(lo, hi + 1))
    elif args.train_frames_only:
        frame_ids = list(range(int(train_lo), int(train_hi) + 1))
    else:
        frame_ids = list(range(0, num_frames))

    hidden_args = _hidden_args_from_cfg(model_dir / "cfg_args")
    deform_network = _import_wu_deform_network(args.fourd_root)
    deform_net = deform_network(hidden_args)
    state = torch.load(str(deform_path), map_location="cpu")
    missing, unexpected = deform_net.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing deform keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected deform keys: {unexpected}", file=sys.stderr)
    deform_net.eval()

    xyz, opacity_logit = _read_ply_means_opacity(ply_path)
    weights = 1.0 / (1.0 + np.exp(-opacity_logit))  # sigmoid(opacity logits)
    n = xyz.shape[0]
    xyz_t = torch.from_numpy(xyz.astype(np.float32))
    # Deformation gates scale/rotation on these too; zeros are fine because Phase 1
    # only consumes the deformed means (no_do/no_dshs leave opacity/shs untouched).
    scales_t = torch.zeros((n, 3), dtype=torch.float32)
    rots_t = torch.zeros((n, 4), dtype=torch.float32)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, float | int]] = []
    raw_positions: list[np.ndarray] = []
    times: list[float] = []
    denom = max(num_frames - 1, 1)
    for frame in frame_ids:
        t_norm = float(frame) / float(denom)
        center = _deformed_centroid(deform_net, xyz_t, scales_t, rots_t, weights, t_norm)
        t_sec = float(frame) / float(fps)
        raw_positions.append(center)
        times.append(t_sec)
        raw_rows.append(
            {
                "frame": int(frame),
                "t_sec": t_sec,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            }
        )

    raw_arr = np.stack(raw_positions, axis=0)

    # Optionally register the (arbitrary) 4DGS world origin into the sim/camera world
    # by a constant translation estimated from the extracted window only.
    align_translation = np.zeros(3, dtype=np.float64)
    if args.align_to_gt:
        if args.gt_poses is None or not args.gt_poses.is_file():
            raise ValueError("--align-to-gt requires a valid --gt-poses CSV")
        traj_align = load_object_poses_csv(args.gt_poses)
        gt_align = np.stack(
            [traj_align.by_frame(int(f)).position for f in frame_ids], axis=0
        )
        align_translation = np.mean(gt_align - raw_arr, axis=0)
        raw_arr = raw_arr + align_translation[None, :]
        for i, row in enumerate(raw_rows):
            row["x"] = float(raw_arr[i, 0])
            row["y"] = float(raw_arr[i, 1])
            row["z"] = float(raw_arr[i, 2])

    smoothed_arr = smooth_trajectory_savgol(
        raw_arr, window_length=args.savgol_window, polyorder=args.savgol_polyorder
    )
    smoothed_rows = [
        {
            "frame": raw_rows[i]["frame"],
            "t_sec": raw_rows[i]["t_sec"],
            "x": float(smoothed_arr[i, 0]),
            "y": float(smoothed_arr[i, 1]),
            "z": float(smoothed_arr[i, 2]),
        }
        for i in range(len(raw_rows))
    ]

    raw_csv = out_dir / "trajectory_raw.csv"
    smoothed_csv = out_dir / "trajectory_smoothed.csv"
    plot_png = out_dir / "trajectory_plot.png"
    write_trajectory_csv(raw_csv, raw_rows)
    write_trajectory_csv(smoothed_csv, smoothed_rows)

    gt_arr = None
    gt_aligned = None
    train_mse = None
    train_mse_aligned = None
    if args.gt_poses is not None and args.gt_poses.is_file():
        traj = load_object_poses_csv(args.gt_poses)
        gt_arr = np.stack([traj.by_frame(int(f)).position for f in frame_ids], axis=0)
        train_mse = float(np.mean(np.sum((raw_arr - gt_arr) ** 2, axis=1)))
        gt_aligned, _ = procrustes_align(raw_arr, gt_arr)
        train_mse_aligned = float(np.mean(np.sum((raw_arr - gt_aligned) ** 2, axis=1)))

    plot_trajectory(
        plot_png,
        raw=raw_arr,
        smoothed=smoothed_arr,
        times_s=np.asarray(times),
        gt=gt_arr,
        gt_aligned=gt_aligned,
    )

    meta = {
        "backend": "wu_4dgaussians",
        "model_dir": str(model_dir),
        "iteration": iteration,
        "checkpoint_ply": str(ply_path),
        "deformation": str(deform_path),
        "num_gaussians": int(n),
        "num_frames": len(raw_rows),
        "frame_ids": [int(f) for f in frame_ids],
        "fps": float(fps),
        "time_normalization": "frame / (num_frames - 1)",
        "aligned_to_gt": bool(args.align_to_gt),
        "align_translation_xyz_m": align_translation.tolist(),
        "train_mse_m2": train_mse,
        "train_mse_aligned_m2": train_mse_aligned,
        "savgol_window": args.savgol_window,
        "savgol_polyorder": args.savgol_polyorder,
    }
    (out_dir / "phase1_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(raw_rows)} frames -> {out_dir}")
    print(f"  raw:      {raw_csv}")
    print(f"  smoothed: {smoothed_csv}")
    print(f"  plot:     {plot_png}")
    if train_mse is not None:
        print(f"  train MSE vs GT:            {train_mse:.6f} m^2")
        print(f"  train MSE vs GT (aligned):  {train_mse_aligned:.6f} m^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
