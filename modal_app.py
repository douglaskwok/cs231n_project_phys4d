"""Modal GPU jobs for CS231N Phys4D.

Prerequisites:
  pip install -r requirements-modal.txt
  modal setup

Local prep (uses **your** PyBullet RGB under outputs/sphere_bounce_m2/rgb/):
  python scripts/generate_sphere_bounce_dataset.py
  python scripts/export_gs_blender_scene.py          # static 3DGS (one frame)
  python scripts/export_4dgs_dataset.py            # full video for 4DGS

Modal:
  modal run modal_app.py                    # GPU smoke
  modal run modal_app.py --tests            # remote unittest
  modal run modal_app.py --upload           # upload static Blender scene
  modal run modal_app.py --train            # 3DGS (frame 0, 6 cams)
  modal run modal_app.py --upload-4d        # upload DyNeRF folder
  modal run modal_app.py --train-4d         # fudan 4D Gaussian Splatting
  modal run modal_app.py --render-4d       # raster MP4 + PNGs (multi-cam + correct time t)
  modal run modal_app.py --render-4d-orbit # horizontal orbit MP4 (novel viewpoints)
  modal run modal_app.py --eval-4d        # PSNR/MAE vs GT (train + test splits)
  modal run modal_app.py --upload-batch          # sphere_bounce_batch -> volume
  modal run modal_app.py --extract-perception    # cache states + t=0 visual feats on volume
  modal run modal_app.py --train-visual-dynamics # project.md dynamics (GPU)
  modal run modal_app.py --upload-visual-pipeline
  modal run modal_app.py --visual-pipeline       # rollout + warp E2E
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal  # Modal 1.x: use Image.add_local_dir, not modal.Mount

REPO_ROOT = Path(__file__).resolve().parent

app = modal.App("cs231n-phys4d")

data_volume = modal.Volume.from_name("phys4d-gs-data", create_if_missing=True)
output_volume = modal.Volume.from_name("phys4d-gs-output", create_if_missing=True)

_torch_image = modal.Image.from_registry(
    "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
).pip_install("numpy<2", "imageio")

# Build CUDA extensions for official 3D Gaussian Splatting (slow first build).
# Image build has no GPU; TORCH_CUDA_ARCH_LIST must be set or nvcc arch detection crashes.
_gs_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0")
    .pip_install("plyfile", "tqdm", "opencv-python-headless", "numpy<2", "ninja")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6+PTX"})
    .run_commands(
        "git clone --recursive --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /opt/gs",
        "pip install /opt/gs/submodules/diff-gaussian-rasterization /opt/gs/submodules/simple-knn",
    )
)

# fudan-zvg/4d-gaussian-splatting (ICLR 2024): 4D primitives + temporal training.
_4dgs_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "plyfile",
        "tqdm",
        "opencv-python-headless",
        "numpy<2",
        "ninja",
        "omegaconf",
        "imagesize",
        "kornia",
        "torchmetrics",
        "torchvision",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.6+PTX"})
    .run_commands(
        # Rasterizer is JIT-built at train time (gaussian_renderer/diff_gaussian_rasterization.py).
        "git clone --recursive --depth 1 https://github.com/fudan-zvg/4d-gaussian-splatting.git /opt/4dgs",
        "pip install /opt/4dgs/simple-knn /opt/4dgs/pointops2",
    )
    .add_local_dir(
        str(REPO_ROOT / "configs"),
        remote_path="/repo_configs",
        ignore=["__pycache__", ".DS_Store"],
    )
    .add_local_dir(
        str(REPO_ROOT / "scripts"),
        remote_path="/repo/scripts",
        ignore=["__pycache__", ".DS_Store"],
    )
)

_repo_ignore = [
    ".git",
    "outputs",
    "phys_sim",
    ".venv",
    "__pycache__",
    "*.pyc",
    ".ipynb_checkpoints",
    ".DS_Store",
    "**/.DS_Store",
]

_test_image = _torch_image.add_local_dir(
    str(REPO_ROOT),
    remote_path="/repo",
    ignore=_repo_ignore,
)

# Visual dynamics: perception cache, transformer train, rollout + 3DGS warp.
_ml_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2",
        "imageio",
        "torchvision",
        "plyfile",
        "opencv-python-headless",
        "pybullet",
    )
    .add_local_dir(
        str(REPO_ROOT / "src"),
        remote_path="/repo/src",
        ignore=["__pycache__", ".DS_Store"],
    )
    .add_local_dir(
        str(REPO_ROOT / "scripts"),
        remote_path="/repo/scripts",
        ignore=["__pycache__", ".DS_Store"],
    )
    .add_local_dir(
        str(REPO_ROOT / "configs"),
        remote_path="/repo_configs",
        ignore=["__pycache__", ".DS_Store"],
    )
)


@app.function(image=_torch_image, gpu="T4", timeout=600)
def smoke_gpu() -> dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available inside container")
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }


@app.function(image=_test_image, gpu="T4", timeout=600)
def run_phys_tests() -> str:
    import os

    os.chdir("/repo")
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests/test_restitution_recovery.py",
            "tests/test_visual_dynamics.py",
            "-v",
        ],
        check=True,
        env=env,
    )
    return "tests_ok"


@app.function(
    image=_gs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 3,
)
def train_gs(iterations: int = 7000) -> str:
    """Train 3DGS on /data/scene (uploaded PyBullet RGB, one static timestep)."""
    scene = Path("/data/scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No scene at /data/scene. Run: modal run modal_app.py --upload"
        )

    model_dir = Path("/outputs/gs_sphere_bounce")
    if model_dir.exists():
        import shutil

        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "/opt/gs/train.py",
        "-s",
        str(scene),
        "-m",
        str(model_dir),
        "--iterations",
        str(iterations),
    ]
    # Older 3DGS builds expose --disable_viewer; ignore if absent.
    try:
        subprocess.run(
            [*cmd, "--disable_viewer"],
            check=True,
            cwd="/opt/gs",
        )
    except subprocess.CalledProcessError:
        subprocess.run(cmd, check=True, cwd="/opt/gs")

    output_volume.commit()
    ply = model_dir / "point_cloud"
    iters = sorted(ply.glob("iteration_*")) if ply.is_dir() else []
    last = iters[-1].name if iters else "none"
    return f"trained -> volume phys4d-gs-output:{model_dir} ({last})"


def _train_4dgs_on_paths(
    scene: Path,
    model_dir: Path,
    config_name: str,
) -> str:
    """Shared fudan 4DGS train + PLY export (used by single-scene and batch jobs)."""
    import shutil
    import sys

    from omegaconf import OmegaConf

    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(f"No DyNeRF scene at {scene}")

    cfg_src = Path("/repo_configs") / config_name
    if not cfg_src.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_src}")

    cfg = OmegaConf.load(cfg_src)
    cfg.ModelParams.source_path = str(scene)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    cfg.ModelParams.model_path = str(model_dir)

    cfg_path = Path("/tmp/train_4dgs.yaml")
    OmegaConf.save(cfg, cfg_path)

    # fudan train loads torch after building CUDA ext with libgomp; MKL_INTEL vs GNU OpenMP
    # can make train.py exit 1 even after "Training complete". Force Intel layer.
    train_env = {
        **os.environ,
        "MKL_SERVICE_FORCE_INTEL": "1",
        "MKL_THREADING_LAYER": "INTEL",
    }
    subprocess.run(
        ["python", "/opt/4dgs/train.py", "--config", str(cfg_path)],
        check=True,
        cwd="/opt/4dgs",
        env=train_env,
    )

    sys.path.insert(0, "/repo/scripts")
    from export_4dgs_ply import export_checkpoint_to_ply

    def _ckpt_iter(path: Path) -> int:
        if path.stem == "chkpnt_best":
            return -1
        return int(path.stem.replace("chkpnt", ""))

    ckpts = [p for p in model_dir.glob("chkpnt*.pth") if _ckpt_iter(p) >= 0]
    if not ckpts:
        ckpts = list(model_dir.glob("chkpnt*.pth"))
    ckpt = max(ckpts, key=_ckpt_iter) if ckpts else None

    ply_out = model_dir / "point_cloud" / "exported" / "point_cloud.ply"
    if ckpt is not None:
        try:
            meta = export_checkpoint_to_ply(ckpt, ply_out)
            ply_note = f"PLY @ iter {meta['iteration']} -> {ply_out.name}"
        except Exception as exc:  # noqa: BLE001 — optional SuperSplat export
            ply_note = f"PLY export skipped ({exc})"
    else:
        ply_note = "no checkpoint found for PLY export"

    output_volume.commit()
    on_vol = [p.name for p in model_dir.glob("chkpnt*.pth")]
    return (
        f"4dgs trained -> phys4d-gs-output:{model_dir} "
        f"checkpoints={on_vol}. {ply_note}"
    )


@app.function(
    image=_4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 6,
)
def train_4dgs(config_name: str = "sphere_bounce_4dgs.yaml") -> str:
    """Train fudan 4DGS on /data/4d_scene (DyNeRF export from PyBullet)."""
    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No 4D scene at /data/4d_scene. Run: modal run modal_app.py --upload-4d"
        )
    return _train_4dgs_on_paths(scene, Path("/outputs/4dgs_sphere_bounce"), config_name)


@app.function(
    image=_4dgs_image,
    gpu="T4",
    volumes={"/outputs": output_volume},
    timeout=30 * 60,
)
def export_4dgs_ply(
    checkpoint_name: str = "chkpnt_best.pth",
) -> str:
    """Write SuperSplat PLY from a trained 4DGS checkpoint on the output volume."""
    import sys

    sys.path.insert(0, "/repo/scripts")
    from export_4dgs_ply import export_checkpoint_to_ply

    model_dir = Path("/outputs/4dgs_sphere_bounce")
    ckpt = model_dir / checkpoint_name
    if not ckpt.is_file():
        available = [p.name for p in sorted(model_dir.glob("chkpnt*.pth"))]
        raise FileNotFoundError(
            f"No {checkpoint_name} under {model_dir}. Found: {available}"
        )

    out = model_dir / "point_cloud" / "exported" / "point_cloud.ply"
    meta = export_checkpoint_to_ply(ckpt, out)
    output_volume.commit()
    return (
        f"exported {meta['num_points']} points @ iter {meta['iteration']} -> "
        f"phys4d-gs-output:{out}"
    )


def _pick_4d_checkpoint(model_dir: Path, checkpoint_name: str | None) -> Path:
    if checkpoint_name:
        ckpt = model_dir / checkpoint_name
        if not ckpt.is_file():
            available = [p.name for p in sorted(model_dir.glob("chkpnt*.pth"))]
            raise FileNotFoundError(
                f"No {checkpoint_name} under {model_dir}. Found: {available}"
            )
        return ckpt

    best = model_dir / "chkpnt_best.pth"
    if best.is_file():
        return best

    def _ckpt_iter(path: Path) -> int:
        if path.stem == "chkpnt_best":
            return -1
        return int(path.stem.replace("chkpnt", ""))

    ckpts = [p for p in model_dir.glob("chkpnt*.pth") if _ckpt_iter(p) >= 0]
    if not ckpts:
        raise FileNotFoundError(f"No chkpnt*.pth under {model_dir}")
    return max(ckpts, key=_ckpt_iter)


@app.function(
    image=_4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 90,
)
def render_4d_trajectory(
    config_name: str = "sphere_bounce_4dgs.yaml",
    checkpoint_name: str | None = None,
    fps: float = 60.0,
    dry_run_max: int = 0,
) -> str:
    """Rasterize 4DGS at each frame's timestamp; sort by time then camera (motion + orbit)."""
    import shutil

    from omegaconf import OmegaConf

    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No 4D scene at /data/4d_scene. Run: modal run modal_app.py --upload-4d"
        )

    model_dir = Path("/outputs/4dgs_sphere_bounce")
    ckpt = _pick_4d_checkpoint(model_dir, checkpoint_name)

    cfg_src = Path("/repo_configs") / config_name
    if not cfg_src.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_src}")
    cfg = OmegaConf.load(cfg_src)
    cfg.ModelParams.source_path = str(scene)
    cfg_path = Path("/tmp/render_4dgs.yaml")
    OmegaConf.save(cfg, cfg_path)

    out_base = Path("/outputs/4dgs_renders/latest")
    out_base.mkdir(parents=True, exist_ok=True)
    for child in out_base.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    subprocess.run(
        [
            sys.executable,
            "/repo/scripts/render_4dgs_trajectory.py",
            "--mode",
            "dataset",
            "--fourd-root",
            "/opt/4dgs",
            "--config",
            str(cfg_path),
            "--dataset",
            str(scene),
            "--checkpoint",
            str(ckpt),
            "--out-dir",
            str(out_base),
            "--fps",
            str(fps),
            *(["--dry-run-max", str(dry_run_max)] if dry_run_max > 0 else []),
        ],
        check=True,
    )

    output_volume.commit()
    return (
        f"4dgs render -> phys4d-gs-output:{out_base} "
        f"(checkpoint {ckpt.name}, trajectory.mp4)"
    )


@app.function(
    image=_4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60,
)
def render_4d_orbit_job(
    config_name: str = "sphere_bounce_4dgs.yaml",
    checkpoint_name: str | None = None,
    fps: float = 60.0,
    orbit_frames: int = 180,
    orbit_time_start: float | None = None,
    orbit_time_end: float | None = None,
) -> str:
    """Novel horizontal orbit (intrinsics from dataset JSON); time sweeps along bounce range."""
    import shutil

    from omegaconf import OmegaConf

    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No 4D scene at /data/4d_scene. Run: modal run modal_app.py --upload-4d"
        )

    model_dir = Path("/outputs/4dgs_sphere_bounce")
    ckpt = _pick_4d_checkpoint(model_dir, checkpoint_name)

    cfg_src = Path("/repo_configs") / config_name
    if not cfg_src.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_src}")
    cfg = OmegaConf.load(cfg_src)
    cfg.ModelParams.source_path = str(scene)
    cfg_path = Path("/tmp/render_4dgs_orbit.yaml")
    OmegaConf.save(cfg, cfg_path)

    out_base = Path("/outputs/4dgs_renders/orbit_latest")
    out_base.mkdir(parents=True, exist_ok=True)
    for child in out_base.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    cmd = [
        sys.executable,
        "/repo/scripts/render_4dgs_trajectory.py",
        "--mode",
        "orbit",
        "--fourd-root",
        "/opt/4dgs",
        "--config",
        str(cfg_path),
        "--dataset",
        str(scene),
        "--checkpoint",
        str(ckpt),
        "--out-dir",
        str(out_base),
        "--fps",
        str(fps),
        "--mp4",
        "orbit.mp4",
        "--orbit-frames",
        str(orbit_frames),
    ]
    if orbit_time_start is not None:
        cmd += ["--orbit-time-start", str(orbit_time_start)]
    if orbit_time_end is not None:
        cmd += ["--orbit-time-end", str(orbit_time_end)]

    subprocess.run(cmd, check=True)

    output_volume.commit()
    return (
        f"4dgs orbit -> phys4d-gs-output:{out_base} "
        f"(checkpoint {ckpt.name}, orbit.mp4)"
    )


@app.function(
    image=_4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 120,
)
def eval_4dgs_metrics_remote(
    config_name: str = "sphere_bounce_4dgs.yaml",
    checkpoint_name: str | None = None,
    dry_run_max: int = 0,
) -> str:
    """PSNR / MSE / MAE on train+test transforms vs ground-truth PNGs."""
    from omegaconf import OmegaConf

    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No 4D scene at /data/4d_scene. Run: modal run modal_app.py --upload-4d"
        )

    model_dir = Path("/outputs/4dgs_sphere_bounce")
    ckpt = _pick_4d_checkpoint(model_dir, checkpoint_name)

    cfg_src = Path("/repo_configs") / config_name
    if not cfg_src.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_src}")
    cfg = OmegaConf.load(cfg_src)
    cfg.ModelParams.source_path = str(scene)
    cfg_path = Path("/tmp/eval_4dgs.yaml")
    OmegaConf.save(cfg, cfg_path)

    out_dir = Path("/outputs/4dgs_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metrics_4dgs.json"

    cmd = [
        sys.executable,
        "/repo/scripts/eval_4dgs_metrics.py",
        "--fourd-root",
        "/opt/4dgs",
        "--config",
        str(cfg_path),
        "--dataset",
        str(scene),
        "--checkpoint",
        str(ckpt),
        "--out-json",
        str(out_json),
    ]
    if dry_run_max > 0:
        cmd += ["--dry-run-max", str(dry_run_max)]

    subprocess.run(cmd, check=True)
    output_volume.commit()
    text = out_json.read_text(encoding="utf-8")
    return f"4dgs metrics -> phys4d-gs-output:{out_json}\n{text}"


@app.function(
    image=_ml_image,
    gpu="T4",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 3,
)
def extract_perception_remote(
    batch_rel: str = "sphere_bounce_batch",
    config_name: str = "visual_dynamics.json",
    limit: int = 0,
) -> str:
    """Phase 2 on Modal: states.npy + visual_feat_t0.npy per scene under /data/<batch>."""
    batch_root = Path("/data") / batch_rel
    if not batch_root.is_dir():
        raise FileNotFoundError(
            f"No batch at {batch_root}. Run: modal run modal_app.py --upload-batch"
        )
    cfg = Path("/repo_configs") / config_name
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    cmd = [
        sys.executable,
        "/repo/scripts/extract_perception.py",
        "--config",
        str(cfg),
        "--batch-root",
        str(batch_root),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    cmd += ["--device", "cuda", "--skip-existing"]
    subprocess.run(cmd, check=True, env=env)
    output_volume.commit()
    return f"perception caches written under /data/{batch_rel}/*/perception/"


@app.function(
    image=_ml_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 8,
)
def train_visual_dynamics_remote(
    manifest_rel: str = "sphere_bounce_batch/dataset_manifest.json",
    config_name: str = "visual_dynamics.json",
    no_visual: bool = False,
) -> str:
    """Train visually conditioned dynamics (project.md) on /data batch scenes."""
    manifest = Path("/data") / manifest_rel
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No manifest at {manifest}. Upload batch: modal run modal_app.py --upload-batch"
        )
    cfg_path = Path("/repo_configs") / config_name
    out_dir = Path("/outputs/visual_dynamics")
    if no_visual:
        out_dir = Path("/outputs/visual_dynamics/states_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    cmd = [
        sys.executable,
        "/repo/scripts/train_visual_dynamics.py",
        "--config",
        str(cfg_path),
        "--manifest",
        str(manifest),
        "--data-root",
        "/data",
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
    ]
    if no_visual:
        cmd.append("--no-visual")
    subprocess.run(cmd, check=True, env=env)
    output_volume.commit()
    ckpt = "visual_dynamics_states_only.pt" if no_visual else "visual_dynamics.pt"
    return f"visual dynamics -> phys4d-gs-output:{out_dir} ({ckpt})"


@app.function(
    image=_ml_image,
    gpu="T4",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60,
)
def run_visual_dynamics_pipeline_remote(
    scene_rel: str = "scene",
    config_name: str = "visual_dynamics.json",
    checkpoint_name: str = "visual_dynamics.pt",
    ply_rel: str = "gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply",
    gs_cameras_rel: str = "gs_sphere_bounce/cameras.json",
    out_rel: str = "visual_dynamics_pipeline",
) -> str:
    """Phase 5: perception + rollout + warp (project.md)."""
    scene_dir = Path("/data") / scene_rel
    if not (scene_dir / "rgb").is_dir():
        raise FileNotFoundError(
            f"No scene at {scene_dir}. Run: modal run modal_app.py --upload-visual-pipeline"
        )
    ckpt = Path("/outputs/visual_dynamics") / checkpoint_name
    if not ckpt.is_file():
        ckpt = Path("/outputs/visual_dynamics/states_only") / checkpoint_name
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"No checkpoint {checkpoint_name}. Run: modal run modal_app.py --train-visual-dynamics"
        )
    ply = Path("/outputs") / ply_rel
    if not ply.is_file():
        raise FileNotFoundError(f"No PLY at {ply}. Run --upload && --train (3DGS) first.")
    gs_cams = Path("/outputs") / gs_cameras_rel
    cfg = Path("/repo_configs") / config_name
    out_dir = Path("/outputs") / out_rel
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    cmd = [
        sys.executable,
        "/repo/scripts/run_visual_dynamics_pipeline.py",
        "--config",
        str(cfg),
        "--scene-dir",
        str(scene_dir),
        "--checkpoint",
        str(ckpt),
        "--ply",
        str(ply),
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
    ]
    if gs_cams.is_file():
        cmd += ["--gs-cameras", str(gs_cams)]
    subprocess.run(cmd, check=True, env=env)
    output_volume.commit()
    report = out_dir / "pipeline_report.json"
    text = report.read_text(encoding="utf-8") if report.is_file() else "(no report)"
    return f"visual pipeline -> phys4d-gs-output:{out_dir}\n{text}"


@app.local_entrypoint()
def main(
    tests: bool = False,
    upload: bool = False,
    train: bool = False,
    upload_4d: bool = False,
    train_4d: bool = False,
    export_4d_ply: bool = False,
    render_4d: bool = False,
    render_4d_orbit: bool = False,
    eval_4d: bool = False,
    upload_batch: bool = False,
    extract_perception: bool = False,
    train_visual_dynamics: bool = False,
    visual_states_only: bool = False,
    upload_visual_pipeline: bool = False,
    visual_pipeline: bool = False,
    batch_rel: str = "sphere_bounce_batch",
    vd_manifest: str = "sphere_bounce_batch/dataset_manifest.json",
    scene_rel: str = "scene",
    pipeline_out_rel: str = "visual_dynamics_pipeline",
    perception_limit: int = 0,
    render_4d_checkpoint: str | None = None,
    render_fps: float = 60.0,
    render_dry_run_max: int = 0,
    orbit_frames: int = 180,
    orbit_time_start: float | None = None,
    orbit_time_end: float | None = None,
    eval_4d_dry_run_max: int = 0,
    frame: int = 0,
    iterations: int = 7000,
) -> None:
    if tests:
        print(run_phys_tests.remote())
        return
    if upload:
        export_script = REPO_ROOT / "scripts" / "export_gs_blender_scene.py"
        out_scene = REPO_ROOT / "outputs" / "sphere_bounce_m2" / f"gs_blender_frame{frame:05d}"
        subprocess.run(
            [
                sys.executable,
                str(export_script),
                "--frame",
                str(frame),
                "--output",
                str(out_scene),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        # Copy local scene tree into Modal Volume (uses your generated PNGs).
        subprocess.run(
            ["modal", "volume", "rm", "phys4d-gs-data", "scene", "-r"],
            check=False,
        )
        subprocess.run(
            ["modal", "volume", "put", "phys4d-gs-data", str(out_scene), "scene"],
            check=True,
        )
        print(f"Uploaded {out_scene} -> volume phys4d-gs-data:/scene")
        return
    if train:
        print(train_gs.remote(iterations=iterations))
        print(
            "Download: modal volume get phys4d-gs-output gs_sphere_bounce . --force\n"
            "  PLY: gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply"
        )
        return
    if upload_4d:
        export_script = REPO_ROOT / "scripts" / "export_4dgs_dataset.py"
        out_scene = REPO_ROOT / "outputs" / "sphere_bounce_m2" / "dynerf_sphere_bounce"
        subprocess.run(
            [sys.executable, str(export_script), "--output", str(out_scene)],
            check=True,
            cwd=str(REPO_ROOT),
        )
        subprocess.run(
            ["modal", "volume", "rm", "phys4d-gs-data", "4d_scene", "-r"],
            check=False,
        )
        subprocess.run(
            ["modal", "volume", "put", "phys4d-gs-data", str(out_scene), "4d_scene"],
            check=True,
        )
        print(f"Uploaded {out_scene} -> volume phys4d-gs-data:/4d_scene")
        return
    if train_4d:
        print(train_4dgs.remote())
        print(
            "Raster (calibrated): modal run modal_app.py --render-4d\n"
            "Raster (orbit):      modal run modal_app.py --render-4d-orbit\n"
            "Metrics (PSNR/GT):   modal run modal_app.py --eval-4d\n"
            "Export PLY: modal run modal_app.py --export-4d-ply\n"
            "Download: modal volume get phys4d-gs-output 4dgs_sphere_bounce . --force\n"
            "  View: 4dgs_sphere_bounce/point_cloud/exported/point_cloud.ply"
        )
        return
    if export_4d_ply:
        print(export_4dgs_ply.remote())
        print(
            "Download: modal volume get phys4d-gs-output "
            "4dgs_sphere_bounce/point_cloud/exported . --force"
        )
        return
    if render_4d:
        print(
            render_4d_trajectory.remote(
                checkpoint_name=render_4d_checkpoint,
                fps=render_fps,
                dry_run_max=render_dry_run_max,
            )
        )
        print(
            "Download video: modal volume get phys4d-gs-output 4dgs_renders/latest . --force\n"
            "  Play trajectory.mp4; PNGs sorted by simulation time then camera."
        )
        return
    if render_4d_orbit:
        print(
            render_4d_orbit_job.remote(
                checkpoint_name=render_4d_checkpoint,
                fps=render_fps,
                orbit_frames=orbit_frames,
                orbit_time_start=orbit_time_start,
                orbit_time_end=orbit_time_end,
            )
        )
        print(
            "Download orbit: modal volume get phys4d-gs-output "
            "4dgs_renders/orbit_latest . --force  (orbit.mp4)"
        )
        return
    if eval_4d:
        print(
            eval_4dgs_metrics_remote.remote(
                checkpoint_name=render_4d_checkpoint,
                dry_run_max=eval_4d_dry_run_max,
            )
        )
        print(
            "Full JSON: modal volume get phys4d-gs-output 4dgs_eval/metrics_4dgs.json . --force"
        )
        return
    if upload_batch:
        batch_src = REPO_ROOT / "outputs" / batch_rel
        if not batch_src.is_dir():
            raise FileNotFoundError(
                f"Missing {batch_src}. Generate data locally first."
            )
        print(
            "Uploading batch via Modal Volume (local CLI, no remote GPU).\n"
            "First `modal run` of this repo can sit silent 10–30+ min while 3DGS/4DGS "
            "images build even for --upload-batch. Prefer:\n"
            "  bash scripts/upload_batch_to_modal.sh\n",
            flush=True,
        )
        subprocess.run(
            ["modal", "volume", "rm", "phys4d-gs-data", batch_rel, "-r"],
            check=False,
        )
        print(f"Putting {batch_src} -> phys4d-gs-data:/{batch_rel} ...", flush=True)
        subprocess.run(
            ["modal", "volume", "put", "phys4d-gs-data", str(batch_src), batch_rel],
            check=True,
        )
        manifest = batch_src / "dataset_manifest.json"
        if manifest.is_file():
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "put",
                    "phys4d-gs-data",
                    str(manifest),
                    f"{batch_rel}/dataset_manifest.json",
                ],
                check=True,
            )
        print(f"Uploaded {batch_src} -> phys4d-gs-data:/{batch_rel}")
        return
    if extract_perception:
        print(
            extract_perception_remote.remote(
                batch_rel=batch_rel,
                limit=perception_limit,
            )
        )
        return
    if train_visual_dynamics:
        print(
            train_visual_dynamics_remote.remote(
                manifest_rel=vd_manifest,
                no_visual=visual_states_only,
            )
        )
        print(
            "Download: modal volume get phys4d-gs-output visual_dynamics . --force"
        )
        return
    if upload_visual_pipeline:
        scene_src = REPO_ROOT / "outputs" / "sphere_bounce_m2"
        if not (scene_src / "rgb").is_dir():
            raise FileNotFoundError(f"Missing {scene_src}")
        subprocess.run(
            ["modal", "volume", "rm", "phys4d-gs-data", "scene", "-r"],
            check=False,
        )
        subprocess.run(
            ["modal", "volume", "put", "phys4d-gs-data", str(scene_src), "scene"],
            check=True,
        )
        for ckpt in (
            REPO_ROOT / "outputs" / "visual_dynamics" / "visual_dynamics.pt",
            REPO_ROOT / "visual_dynamics" / "visual_dynamics.pt",
        ):
            if not ckpt.is_file():
                continue
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "rm",
                    "phys4d-gs-output",
                    "visual_dynamics/visual_dynamics.pt",
                ],
                check=False,
            )
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "put",
                    "phys4d-gs-output",
                    str(ckpt),
                    "visual_dynamics/visual_dynamics.pt",
                ],
                check=True,
            )
            break
        gs_local = REPO_ROOT / "gs_sphere_bounce"
        if (gs_local / "point_cloud").is_dir():
            subprocess.run(
                ["modal", "volume", "rm", "phys4d-gs-output", "gs_sphere_bounce", "-r"],
                check=False,
            )
            subprocess.run(
                ["modal", "volume", "put", "phys4d-gs-output", str(gs_local), "gs_sphere_bounce"],
                check=True,
            )
        print("Uploaded scene + optional visual_dynamics ckpt + gs_sphere_bounce")
        return
    if visual_pipeline:
        print(run_visual_dynamics_pipeline_remote.remote(scene_rel=scene_rel))
        print(
            f"Download: modal volume get phys4d-gs-output {pipeline_out_rel} . --force"
        )
        return
    print(smoke_gpu.remote())
