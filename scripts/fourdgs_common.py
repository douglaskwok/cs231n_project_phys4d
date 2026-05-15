"""Shared loaders for scripts that invoke fudan-zvg ``4d-gaussian-splatting`` locally or on Modal."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def cam_sort_key(image_name: str) -> tuple[int, int]:
    m = re.match(r"cam(\d+)_(\d+)", image_name)
    if not m:
        return (999, 999999)
    return (int(m.group(1)), int(m.group(2)))


def merge_config_yaml(cfg_path: Path):
    """Match ``train.py``: parse → append save_iterations → OmegaConf merge."""
    sys.path.insert(0, os.environ.get("FOURDGS_ROOT", "/opt/4dgs"))

    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")

    args = parser.parse_args(["--config", str(cfg_path)])
    args.save_iterations.append(args.iterations)

    cfg = OmegaConf.load(str(cfg_path))

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), f"YAML key {key!r} unknown to parser"
            setattr(args, key, host[key])

    for k in cfg.keys():
        recursive_merge(k, cfg)

    return args, lp.extract(args), op.extract(args), pp.extract(args)


def model_params_override_time_duration(td: tuple[float, float], lp_like) -> list[float]:
    """Apply optional ``frame_ratio`` to ``time_duration`` (matches training convention)."""
    td = list(td)
    if lp_like.frame_ratio and lp_like.frame_ratio > 1:
        td = [td[0] / lp_like.frame_ratio, td[1] / lp_like.frame_ratio]
    return td


def instantiate_gaussian_model(args_ns_like, lp_like, pp_like):
    """Build ``GaussianModel`` (fudan); caller must insert fudan root in ``sys.path`` first."""
    from scene.gaussian_model import GaussianModel

    td = model_params_override_time_duration(tuple(args_ns_like.time_duration), lp_like)

    return GaussianModel(
        lp_like.sh_degree,
        gaussian_dim=args_ns_like.gaussian_dim,
        time_duration=td,
        rot_4d=args_ns_like.rot_4d,
        force_sh_3d=args_ns_like.force_sh_3d,
        sh_degree_t=2 if pp_like.eval_shfs_4d else 0,
        prefilter_var=lp_like.prefilter_var,
    )
