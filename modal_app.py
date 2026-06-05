"""Modal GPU jobs for CS231N Phys4D.

Prerequisites:
  pip install -r requirements-modal.txt
  modal setup

Local prep (uses **your** PyBullet RGB under outputs/sphere_bounce_m2/rgb/):
  python scripts/generate_sphere_bounce_dataset.py
  python scripts/export_gs_blender_scene.py          # static 3DGS (one frame)
  python 4dgs/scripts/export_4dgs_dataset.py      # full video for 4DGS

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
import shutil
import subprocess
import sys
from pathlib import Path

import modal  # Modal 1.x: use Image.add_local_dir, not modal.Mount

REPO_ROOT = Path(__file__).resolve().parent

app = modal.App("cs231n-phys4d")

data_volume = modal.Volume.from_name("phys4d-gs-data", create_if_missing=True)
output_volume = modal.Volume.from_name("phys4d-gs-output", create_if_missing=True)


def _ignore_4dgs_mount(path: Path) -> bool:
    parts = path.parts
    return (
        "__pycache__" in parts
        or path.name in {".DS_Store"}
        or parts[:3] == ("experiments", "object_only", "runs")
        or parts[:3] == ("experiments", "wu4dgs", "runs")
        or parts[:1] == ("collision_janhavi",)
    )


def _ignore_repo_mount(path: Path) -> bool:
    parts = path.parts
    if not parts:
        return False
    top = parts[0]
    return (
        top in {
            ".git",
            ".venv",
            "external",
            "phys_sim",
            "outputs",
            "dataset_outputs",
        }
        or "__pycache__" in parts
        or path.name in {".DS_Store"}
        or path.suffix == ".pyc"
        or parts[:2] == ("dataset", "outputs")
        or parts[:4] == ("4dgs", "experiments", "object_only", "runs")
        or parts[:4] == ("4dgs", "experiments", "wu4dgs", "runs")
        or parts[:2] == ("4dgs", "collision_janhavi")
        or top.startswith("latest_")
        or top.endswith("_4dgs")
        or top.endswith("_render")
        or top.endswith("_frames")
    )

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
        str(REPO_ROOT / "4dgs"),
        remote_path="/repo/4dgs",
        ignore=_ignore_4dgs_mount,
    )
)

FOURDGS_CONFIGS = "/repo/4dgs/configs"
FOURDGS_SCRIPTS = "/repo/4dgs/scripts"

# hustvl/4DGaussians (Wu et al., CVPR 2024): global canonical Gaussians +
# deformation field. Keep this isolated from the Fudan image because it targets
# the older torch/cu116 stack used by the original project.
_wu4dgs_image = (
    modal.Image.from_registry("pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "numpy<2",
        "typing_extensions>=4.12",
        "tqdm",
        "plyfile",
        "imageio[ffmpeg]",
        "lpips",
        "pytorch_msssim",
        "matplotlib",
        "opencv-python-headless",
        "open3d",
        "ninja",
        "addict",
        "yapf==0.40.1",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.6+PTX"})
    .run_commands(
        "pip install mmcv==1.6.0",
        "git clone --recursive --depth 1 https://github.com/hustvl/4DGaussians.git /opt/wu4dgs",
        "pip install -e /opt/wu4dgs/submodules/depth-diff-gaussian-rasterization",
        "pip install -e /opt/wu4dgs/submodules/simple-knn",
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
    ignore=_ignore_repo_mount,
)

_pybullet_dataset_image = (
    modal.Image.from_registry("python:3.10-slim")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("numpy<2", "imageio[ffmpeg]", "pybullet")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/repo",
        ignore=_ignore_repo_mount,
    )
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

    cfg_src = Path(FOURDGS_CONFIGS) / config_name
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

    sys.path.insert(0, FOURDGS_SCRIPTS)
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
def train_4dgs(
    config_name: str = "sphere_bounce_4dgs.yaml",
    model_rel: str = "4dgs_sphere_bounce",
) -> str:
    """Train fudan 4DGS on /data/4d_scene (DyNeRF export from PyBullet)."""
    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No 4D scene at /data/4d_scene. Run: modal run modal_app.py --upload-4d"
        )
    return _train_4dgs_on_paths(scene, Path("/outputs") / model_rel, config_name)


def _wu4dgs_config_text(
    *,
    iterations: int,
    coarse_iterations: int,
    time_resolution: int,
    bounds: float,
    densify_until_iter: int = 0,
    opacity_reset_interval: int = 0,
) -> str:
    optional_optimization = ""
    if densify_until_iter > 0:
        optional_optimization += f"    densify_until_iter={int(densify_until_iter)},\n"
    if opacity_reset_interval > 0:
        optional_optimization += f"    opacity_reset_interval={int(opacity_reset_interval)},\n"
    return f"""_base_ = '/opt/wu4dgs/arguments/dnerf/dnerf_default.py'

OptimizationParams = dict(
    coarse_iterations={int(coarse_iterations)},
    iterations={int(iterations)},
    pruning_interval=8000,
    render_process=False,
    batch_size=1,
{optional_optimization.rstrip()}
)

ModelHiddenParams = dict(
    bounds={float(bounds)},
    kplanes_config={{
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, {int(time_resolution)}],
    }},
)
"""


def _patch_wu4dgs_aspect_preserving_loader() -> None:
    """Keep Phys4D frames at their native rectangular resolution in Wu's loader."""
    reader_py = Path("/opt/wu4dgs/scene/dataset_readers.py")
    text = reader_py.read_text(encoding="utf-8")
    old = "image = PILtoTorch(image,(800,800))"
    new = "image = PILtoTorch(image,None)"
    if old not in text and new not in text:
        raise RuntimeError("Could not find Wu Blender image resize line to patch")
    if old in text:
        reader_py.write_text(text.replace(old, new), encoding="utf-8")
        print("Patched Wu Blender loader to preserve native image aspect/resolution", flush=True)


def _patch_wu4dgs_forced_aabb(force_aabb: str = "") -> None:
    """Optionally force Wu's deformation AABB across independently trained objects.

    Format is xmin,ymin,zmin,xmax,ymax,zmax. Wu internally calls set_aabb(max, min),
    so the patch only swaps those two vectors before the original call.
    """
    if not force_aabb:
        return
    values = [float(v) for v in force_aabb.split(",")]
    if len(values) != 6:
        raise ValueError(
            "--wu-force-aabb must be xmin,ymin,zmin,xmax,ymax,zmax "
            f"(got {force_aabb!r})"
        )
    xmin, ymin, zmin, xmax, ymax, zmax = values
    scene_py = Path("/opt/wu4dgs/scene/__init__.py")
    text = scene_py.read_text(encoding="utf-8")
    old = "        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)\n"
    new = (
        "        force_aabb = os.environ.get('PHYS4D_WU_FORCE_AABB', '').strip()\n"
        "        if force_aabb:\n"
        "            import numpy as _phys4d_np\n"
        "            vals = [float(v) for v in force_aabb.split(',')]\n"
        "            if len(vals) != 6:\n"
        "                raise ValueError('PHYS4D_WU_FORCE_AABB must be xmin,ymin,zmin,xmax,ymax,zmax')\n"
        "            xyz_min = _phys4d_np.array([vals[0], vals[1], vals[2]], dtype=scene_info.point_cloud.points.dtype)\n"
        "            xyz_max = _phys4d_np.array([vals[3], vals[4], vals[5]], dtype=scene_info.point_cloud.points.dtype)\n"
        "            print('Phys4D forced Wu deformation AABB', xyz_max, xyz_min, flush=True)\n"
        "        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)\n"
    )
    if new in text:
        print(f"Wu deformation AABB already patched: {force_aabb}", flush=True)
    elif old in text:
        scene_py.write_text(text.replace(old, new), encoding="utf-8")
        print(f"Patched Wu deformation AABB override: {force_aabb}", flush=True)
    else:
        raise RuntimeError("Could not find Wu set_aabb line to patch")
    os.environ["PHYS4D_WU_FORCE_AABB"] = f"{xmin},{ymin},{zmin},{xmax},{ymax},{zmax}"


def _patch_wu4dgs_foreground_loss(
    weight: float,
    mask_weight: float = 0.0,
    bg_spill_weight: float = 0.0,
    area_weight: float = 0.0,
    compactness_weight: float = 0.0,
    scale_isotropy_weight: float = 0.0,
    max_scale_weight: float = 0.0,
    max_scale_value: float = 0.02,
    cloud_isotropy_weight: float = 0.0,
    silhouette_roundness_weight: float = 0.0,
    trajectory_anchor_weight: float = 0.0,
    camera_loss_weights: str = "",
) -> None:
    """Optionally focus Wu's loss on masked foreground pixels and compact objects."""
    camera_weight_pairs: list[tuple[int, float]] = []
    if camera_loss_weights.strip():
        for item in camera_loss_weights.replace("\\", "").split(","):
            if not item.strip():
                continue
            cam_s, weight_s = item.split(":", 1)
            camera_weight_pairs.append((int(cam_s), float(weight_s)))
    if (
        weight <= 0
        and mask_weight <= 0
        and bg_spill_weight <= 0
        and area_weight <= 0
        and compactness_weight <= 0
        and scale_isotropy_weight <= 0
        and max_scale_weight <= 0
        and cloud_isotropy_weight <= 0
        and silhouette_roundness_weight <= 0
        and trajectory_anchor_weight <= 0
        and not camera_weight_pairs
    ):
        return
    needs_alpha_render = mask_weight > 0 or area_weight > 0 or silhouette_roundness_weight > 0
    if needs_alpha_render:
        renderer_py = Path("/opt/wu4dgs/gaussian_renderer/__init__.py")
        renderer_text = renderer_py.read_text(encoding="utf-8")
        old = "    else:\n        colors_precomp = override_color\n"
        new = "    else:\n        colors_precomp = override_color\n        shs_final = None\n"
        if new not in renderer_text:
            if old not in renderer_text:
                raise RuntimeError("Could not patch Wu renderer override_color path")
            renderer_py.write_text(renderer_text.replace(old, new), encoding="utf-8")

    train_py = Path("/opt/wu4dgs/train.py")
    text = train_py.read_text(encoding="utf-8")
    if trajectory_anchor_weight > 0:
        old_import = "import copy\n"
        new_import = "import copy\nimport json\nfrom pathlib import Path\n"
        if new_import not in text:
            if old_import not in text:
                raise RuntimeError("Could not patch Wu train.py trajectory imports")
            text = text.replace(old_import, new_import)

        old_anchor_setup = (
            '    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]\n'
            '    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")\n'
        )
        new_anchor_setup = f"""    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    trajectory_anchor_times = None
    trajectory_anchor_centers = None
    trajectory_anchor_path = Path(dataset.source_path) / "trajectory_anchors.json"
    if {float(trajectory_anchor_weight)} > 0 and trajectory_anchor_path.is_file():
        with trajectory_anchor_path.open("r", encoding="utf-8") as anchor_file:
            trajectory_anchor_data = json.load(anchor_file)
        trajectory_anchor_frames = trajectory_anchor_data.get("frames", [])
        if trajectory_anchor_frames:
            trajectory_anchor_times = torch.tensor(
                [float(row["wu_time"]) for row in trajectory_anchor_frames],
                dtype=torch.float32,
                device="cuda",
            )
            trajectory_anchor_centers = torch.tensor(
                [row["center_world_m"] for row in trajectory_anchor_frames],
                dtype=torch.float32,
                device="cuda",
            )
            print(f"Loaded {{len(trajectory_anchor_frames)}} trajectory anchors from {{trajectory_anchor_path}}", flush=True)
"""
        if new_anchor_setup not in text:
            if old_anchor_setup not in text:
                raise RuntimeError("Could not patch Wu train.py trajectory anchor setup")
            text = text.replace(old_anchor_setup, new_anchor_setup)

    if needs_alpha_render:
        old = "        images = []\n        gt_images = []\n"
        new = "        images = []\n        alpha_images = []\n        gt_images = []\n"
        if new not in text:
            if old not in text:
                raise RuntimeError("Could not patch Wu train.py alpha image list")
            text = text.replace(old, new)

        old = (
            '            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], '
            'render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]\n'
            "            images.append(image.unsqueeze(0))\n"
        )
        new = (
            '            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], '
            'render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]\n'
            "            images.append(image.unsqueeze(0))\n"
            "            alpha_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type, override_color=torch.ones_like(gaussians.get_xyz))\n"
            '            alpha_images.append(alpha_pkg["render"].amax(dim=0, keepdim=True).unsqueeze(0))\n'
        )
        if new not in text:
            if old not in text:
                raise RuntimeError("Could not patch Wu train.py alpha render")
            text = text.replace(old, new)

        old = "        image_tensor = torch.cat(images,0)\n        gt_image_tensor = torch.cat(gt_images,0)\n"
        new = (
            "        image_tensor = torch.cat(images,0)\n"
            "        alpha_tensor = torch.cat(alpha_images,0)\n"
            "        gt_image_tensor = torch.cat(gt_images,0)\n"
        )
        if new not in text:
            if old not in text:
                raise RuntimeError("Could not patch Wu train.py alpha tensor")
            text = text.replace(old, new)

    old = "        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])\n"
    camera_weight_map = "{" + ", ".join(f"{cam}: {cam_weight}" for cam, cam_weight in camera_weight_pairs) + "}"
    camera_weight_setup = ""
    if camera_weight_pairs:
        camera_weight_setup = f"""        sample_loss_weights = torch.ones((gt_image_tensor.shape[0], 1, 1, 1), device=gt_image_tensor.device)
        camera_loss_weight_map = {camera_weight_map}
        for camera_weight_i, camera_weight_cam in enumerate(viewpoint_cams):
            camera_weight_name = str(getattr(camera_weight_cam, "image_name", ""))
            camera_weight_id = None
            if camera_weight_name.startswith("cam"):
                camera_weight_digits = []
                for camera_weight_ch in camera_weight_name[3:]:
                    if camera_weight_ch.isdigit():
                        camera_weight_digits.append(camera_weight_ch)
                    else:
                        break
                if camera_weight_digits:
                    camera_weight_id = int("".join(camera_weight_digits))
            if camera_weight_id in camera_loss_weight_map:
                sample_loss_weights[camera_weight_i] = float(camera_loss_weight_map[camera_weight_id])
"""
    else:
        camera_weight_setup = "        sample_loss_weights = torch.ones((gt_image_tensor.shape[0], 1, 1, 1), device=gt_image_tensor.device)\n"
    compactness_term = ""
    if compactness_weight > 0:
        compactness_term = f"""        xyz_compact = gaussians.get_xyz
        xyz_center = xyz_compact.mean(dim=0, keepdim=True).detach()
        Ll1 = Ll1 + {float(compactness_weight)} * (xyz_compact - xyz_center).pow(2).sum(dim=1).mean()
"""
    scale_isotropy_term = ""
    if scale_isotropy_weight > 0:
        scale_isotropy_term = f"""        scale_iso = torch.log(torch.clamp(gaussians.get_scaling, min=1e-6))
        Ll1 = Ll1 + {float(scale_isotropy_weight)} * (scale_iso - scale_iso.mean(dim=1, keepdim=True)).pow(2).mean()
"""
    max_scale_term = ""
    if max_scale_weight > 0:
        max_scale_term = f"""        scale_size = torch.log(torch.clamp(gaussians.get_scaling, min=1e-6))
        max_scale_log = torch.log(torch.tensor({float(max_scale_value)}, device=scale_size.device))
        Ll1 = Ll1 + {float(max_scale_weight)} * torch.relu(scale_size - max_scale_log).pow(2).mean()
"""
    cloud_isotropy_term = ""
    if cloud_isotropy_weight > 0:
        cloud_isotropy_term = f"""        xyz_iso = gaussians.get_xyz - gaussians.get_xyz.mean(dim=0, keepdim=True).detach()
        cov_iso = xyz_iso.transpose(0, 1).matmul(xyz_iso) / torch.clamp(torch.tensor(float(xyz_iso.shape[0]), device=xyz_iso.device), min=1.0)
        eig_iso = torch.linalg.eigvalsh(cov_iso + torch.eye(3, device=xyz_iso.device) * 1e-8)
        eig_iso = eig_iso / torch.clamp(eig_iso.mean().detach(), min=1e-8)
        Ll1 = Ll1 + {float(cloud_isotropy_weight)} * (eig_iso - 1.0).pow(2).mean()
"""
    silhouette_roundness_term = ""
    if silhouette_roundness_weight > 0:
        silhouette_roundness_term = f"""        alpha_for_round = torch.clamp(alpha_tensor, 0.0, 1.0)
        b_round, _, h_round, w_round = alpha_for_round.shape
        yy_round, xx_round = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h_round, device=alpha_for_round.device),
            torch.linspace(-1.0, 1.0, w_round, device=alpha_for_round.device),
            indexing='ij'
        )
        xx_round = xx_round.view(1, 1, h_round, w_round)
        yy_round = yy_round.view(1, 1, h_round, w_round)
        mass_round = alpha_for_round.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        mx_round = (alpha_for_round * xx_round).sum(dim=(2, 3), keepdim=True) / mass_round
        my_round = (alpha_for_round * yy_round).sum(dim=(2, 3), keepdim=True) / mass_round
        dx_round = xx_round - mx_round
        dy_round = yy_round - my_round
        cxx_round = (alpha_for_round * dx_round.pow(2)).sum(dim=(2, 3), keepdim=True) / mass_round
        cyy_round = (alpha_for_round * dy_round.pow(2)).sum(dim=(2, 3), keepdim=True) / mass_round
        cxy_round = (alpha_for_round * dx_round * dy_round).sum(dim=(2, 3), keepdim=True) / mass_round
        trace_round = cxx_round + cyy_round
        disc_round = torch.sqrt((cxx_round - cyy_round).pow(2) + 4.0 * cxy_round.pow(2) + 1e-10)
        anis_round = disc_round / torch.clamp(trace_round, min=1e-6)
        visible_round = (mass_round.flatten() > 4.0).float()
        Ll1 = Ll1 + {float(silhouette_roundness_weight)} * (anis_round.flatten().pow(2) * visible_round).sum() / visible_round.sum().clamp_min(1.0)
"""
    trajectory_anchor_term = ""
    if trajectory_anchor_weight > 0:
        trajectory_anchor_term = f"""        if trajectory_anchor_times is not None and stage == "fine":
            trajectory_loss = torch.tensor(0.0, device=gaussians.get_xyz.device)
            trajectory_count = 0
            for trajectory_cam in viewpoint_cams:
                trajectory_time = torch.tensor(float(trajectory_cam.time), dtype=torch.float32, device=trajectory_anchor_times.device)
                trajectory_idx = torch.argmin(torch.abs(trajectory_anchor_times - trajectory_time))
                trajectory_target = trajectory_anchor_centers[trajectory_idx]
                trajectory_means = gaussians.get_xyz
                trajectory_time_tensor = torch.tensor(float(trajectory_cam.time), device=trajectory_means.device).repeat(trajectory_means.shape[0], 1)
                trajectory_means, _, _, _, _ = gaussians._deformation(
                    trajectory_means,
                    gaussians._scaling,
                    gaussians._rotation,
                    gaussians._opacity,
                    gaussians.get_features,
                    trajectory_time_tensor,
                )
                trajectory_weights = gaussians.get_opacity.detach().squeeze(-1).clamp_min(1e-4)
                trajectory_center = (trajectory_means * trajectory_weights[:, None]).sum(dim=0) / trajectory_weights.sum().clamp_min(1e-6)
                trajectory_loss = trajectory_loss + (trajectory_center - trajectory_target).pow(2).mean()
                trajectory_count += 1
            Ll1 = Ll1 + {float(trajectory_anchor_weight)} * trajectory_loss / max(trajectory_count, 1)
"""
    area_term = ""
    if area_weight > 0:
        area_term = f"""        pred_area = alpha_clamped.sum(dim=(2, 3)).clamp_min(1.0)
        gt_area = fg_mask.sum(dim=(2, 3)).clamp_min(1.0)
        visible_area = (gt_area.flatten() > 4.0).float() * sample_loss_weights.flatten()
        rel_area_error = (torch.log(pred_area) - torch.log(gt_area)).pow(2).flatten()
        Ll1 = Ll1 + {float(area_weight)} * (rel_area_error * visible_area).sum() / visible_area.sum().clamp_min(1.0)
"""

    normalized_region_loss = f"""        gt_rgb_tensor = gt_image_tensor[:,:3,:,:]
{camera_weight_setup.rstrip()}
        fg_mask = (gt_rgb_tensor > (2.0 / 255.0)).any(dim=1, keepdim=True).float()
        bg_mask = 1.0 - fg_mask
        abs_rgb = torch.abs(image_tensor - gt_rgb_tensor)
        fg_mask_weighted = fg_mask * sample_loss_weights
        bg_mask_weighted = bg_mask * sample_loss_weights
        fg_count = (fg_mask_weighted.sum() * gt_rgb_tensor.shape[1]).clamp_min(1.0)
        bg_count = (bg_mask_weighted.sum() * gt_rgb_tensor.shape[1]).clamp_min(1.0)
        fg_l1 = (abs_rgb * fg_mask_weighted).sum() / fg_count
        bg_l1 = (abs_rgb * bg_mask_weighted).sum() / bg_count
        Ll1 = bg_l1 + {float(weight)} * fg_l1
"""
    bg_spill_term = ""
    if bg_spill_weight > 0:
        bg_spill_term = f"""        bg_rgb_spill_loss = (abs_rgb * bg_mask_weighted).sum() / fg_count
        Ll1 = Ll1 + {float(bg_spill_weight)} * bg_rgb_spill_loss
"""

    mask_term = ""
    if mask_weight > 0:
        mask_term = f"""        fg_alpha_norm = fg_mask_weighted.sum().clamp_min(1.0)
        fg_alpha_loss = (torch.abs(alpha_clamped - fg_mask) * fg_mask_weighted).sum() / fg_alpha_norm
        bg_alpha_loss = (torch.abs(alpha_clamped - fg_mask) * bg_mask_weighted).sum() / fg_alpha_norm
        Ll1 = Ll1 + {float(mask_weight)} * (fg_alpha_loss + bg_alpha_loss)
"""

    if needs_alpha_render:
        new = f"""        gt_rgb_tensor = gt_image_tensor[:,:3,:,:]
{camera_weight_setup.rstrip()}
        fg_mask = (gt_rgb_tensor > (2.0 / 255.0)).any(dim=1, keepdim=True).float()
        bg_mask = 1.0 - fg_mask
        abs_rgb = torch.abs(image_tensor - gt_rgb_tensor)
        fg_mask_weighted = fg_mask * sample_loss_weights
        bg_mask_weighted = bg_mask * sample_loss_weights
        fg_count = (fg_mask_weighted.sum() * gt_rgb_tensor.shape[1]).clamp_min(1.0)
        bg_count = (bg_mask_weighted.sum() * gt_rgb_tensor.shape[1]).clamp_min(1.0)
        fg_l1 = (abs_rgb * fg_mask_weighted).sum() / fg_count
        bg_l1 = (abs_rgb * bg_mask_weighted).sum() / bg_count
        Ll1 = bg_l1 + {float(weight)} * fg_l1
        alpha_clamped = torch.clamp(alpha_tensor, 0.0, 1.0)
{mask_term.rstrip()}
{bg_spill_term.rstrip()}
{area_term.rstrip()}
{compactness_term.rstrip()}
{scale_isotropy_term.rstrip()}
{max_scale_term.rstrip()}
{cloud_isotropy_term.rstrip()}
{silhouette_roundness_term.rstrip()}
{trajectory_anchor_term.rstrip()}
"""
    else:
        new = f"""{normalized_region_loss.rstrip()}
{bg_spill_term.rstrip()}
{compactness_term.rstrip()}
{scale_isotropy_term.rstrip()}
{max_scale_term.rstrip()}
{cloud_isotropy_term.rstrip()}
{silhouette_roundness_term.rstrip()}
{trajectory_anchor_term.rstrip()}
"""
    if new in text:
        return
    if old not in text:
        raise RuntimeError("Could not patch Wu train.py foreground-weighted loss")
    train_py.write_text(text.replace(old, new), encoding="utf-8")
    print(
        f"Patched Wu normalized foreground/background L1 loss with foreground_weight={weight} "
        f"mask_weight={mask_weight} bg_spill_weight={bg_spill_weight} "
        f"area_weight={area_weight} compactness_weight={compactness_weight} "
        f"scale_isotropy_weight={scale_isotropy_weight} "
        f"max_scale_weight={max_scale_weight} max_scale_value={max_scale_value} "
        f"cloud_isotropy_weight={cloud_isotropy_weight} "
        f"silhouette_roundness_weight={silhouette_roundness_weight} "
        f"trajectory_anchor_weight={trajectory_anchor_weight} "
        f"camera_loss_weights={camera_loss_weights}",
        flush=True,
    )


@app.function(
    image=_wu4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 8,
)
def train_wu_4dgs(
    model_rel: str = "wu4dgs_sphere_bounce",
    scene_rel: str = "4d_scene",
    iterations: int = 15000,
    coarse_iterations: int = 3000,
    time_resolution: int = 75,
    bounds: float = 1.6,
    foreground_loss_weight: float = 0.0,
    mask_loss_weight: float = 0.0,
    bg_spill_loss_weight: float = 0.0,
    area_loss_weight: float = 0.0,
    compactness_loss_weight: float = 0.0,
    scale_isotropy_loss_weight: float = 0.0,
    max_scale_loss_weight: float = 0.0,
    max_gaussian_scale: float = 0.02,
    cloud_isotropy_loss_weight: float = 0.0,
    silhouette_roundness_loss_weight: float = 0.0,
    trajectory_anchor_loss_weight: float = 0.0,
    camera_loss_weights: str = "",
    densify_until_iter: int = 0,
    opacity_reset_interval: int = 0,
    start_checkpoint: str = "",
    force_aabb: str = "",
) -> str:
    """Train hustvl/4DGaussians on a DyNeRF scene under /data."""
    import shutil
    import tarfile

    scene_rel = scene_rel.strip("/") or "4d_scene"
    scene = Path("/data") / scene_rel
    scene_archive = Path("/data") / f"{scene_rel}.tar.gz"
    if not (scene / "transforms_train.json").is_file():
        if scene_archive.is_file():
            extracted = Path("/tmp") / scene_rel.replace("/", "_")
            if extracted.exists():
                shutil.rmtree(extracted)
            extracted.mkdir(parents=True, exist_ok=True)
            with tarfile.open(scene_archive, "r:gz") as tar:
                tar.extractall(extracted)
            scene = extracted
            print(f"Extracted archived DyNeRF scene from {scene_archive} to {scene}", flush=True)
        else:
            raise FileNotFoundError(
                f"No 4D scene at {scene} or {scene_archive}. "
                "Upload a DyNeRF export first."
            )

    model_dir = Path("/outputs") / model_rel
    start_checkpoint_path: Path | None = None
    if start_checkpoint:
        start_checkpoint_path = Path(start_checkpoint)
        if not start_checkpoint_path.is_absolute():
            start_checkpoint_path = Path("/outputs") / start_checkpoint_path
        if not start_checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing Wu start checkpoint: {start_checkpoint_path}")
        print(f"Resuming Wu 4DGS from {start_checkpoint_path}", flush=True)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path("/tmp/wu4dgs_phys4d.py")
    cfg_path.write_text(
        _wu4dgs_config_text(
            iterations=iterations,
            coarse_iterations=coarse_iterations,
            time_resolution=time_resolution,
            bounds=bounds,
            densify_until_iter=densify_until_iter,
            opacity_reset_interval=opacity_reset_interval,
        ),
        encoding="utf-8",
    )

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    _patch_wu4dgs_aspect_preserving_loader()
    _patch_wu4dgs_forced_aabb(force_aabb)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    _patch_wu4dgs_foreground_loss(
        foreground_loss_weight,
        mask_loss_weight,
        bg_spill_loss_weight,
        area_loss_weight,
        compactness_loss_weight,
        scale_isotropy_loss_weight,
        max_scale_loss_weight,
        max_gaussian_scale,
        cloud_isotropy_loss_weight,
        silhouette_roundness_loss_weight,
        trajectory_anchor_loss_weight,
        camera_loss_weights,
    )
    print("Wu 4DGS import probe...", flush=True)
    subprocess.run(
        [
            "python",
            "-u",
            "-c",
            (
                "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, flush=True); "
                "import open3d; print('open3d ok', flush=True); "
                "from gaussian_renderer import render; print('gaussian_renderer ok', flush=True)"
            ),
        ],
        check=True,
        cwd="/opt/wu4dgs",
        env=env,
        timeout=180,
    )

    train_cmd = [
            "python",
            "-u",
            "/opt/wu4dgs/train.py",
            "--source_path",
            str(scene),
            "--model_path",
            str(model_dir),
            "--configs",
            str(cfg_path),
            "--expname",
            model_rel,
            "--save_iterations",
            str(iterations),
            "--checkpoint_iterations",
            str(iterations),
            "--test_iterations",
            str(iterations + 1),
    ]
    if start_checkpoint_path is not None:
        train_cmd.extend(["--start_checkpoint", str(start_checkpoint_path)])

    subprocess.run(
        train_cmd,
        check=True,
        cwd="/opt/wu4dgs",
        env=env,
    )

    output_volume.commit()
    fine_dir = model_dir / "point_cloud" / f"iteration_{iterations}"
    ckpt = model_dir / f"chkpnt_fine_{iterations}.pth"
    return (
        f"Wu 4DGS trained -> phys4d-gs-output:/{model_rel}\n"
        f"PLY: {fine_dir / 'point_cloud.ply'}\n"
        f"deformation: {fine_dir / 'deformation.pth'}\n"
        f"checkpoint: {ckpt.name if ckpt.is_file() else '(not found)'}"
    )


@app.function(
    image=_wu4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 90,
)
def render_wu_4dgs(
    model_rel: str = "wu4dgs_sphere_bounce",
    scene_rel: str = "4d_scene",
    iteration: int = 15000,
    time_resolution: int = 75,
    bounds: float = 1.6,
    force_aabb: str = "",
    skip_test: bool = True,
    skip_video: bool = True,
) -> str:
    """Render hustvl/4DGaussians train/test/video sets with its native renderer."""
    import shutil
    import tarfile

    scene_rel = scene_rel.strip("/") or "4d_scene"
    scene = Path("/data") / scene_rel
    scene_archive = Path("/data") / f"{scene_rel}.tar.gz"
    if not (scene / "transforms_train.json").is_file():
        if scene_archive.is_file():
            extracted = Path("/tmp") / scene_rel.replace("/", "_")
            if extracted.exists():
                shutil.rmtree(extracted)
            extracted.mkdir(parents=True, exist_ok=True)
            with tarfile.open(scene_archive, "r:gz") as tar:
                tar.extractall(extracted)
            scene = extracted
            print(f"Extracted archived DyNeRF scene from {scene_archive} to {scene}", flush=True)
        else:
            raise FileNotFoundError(
                f"No 4D scene at {scene} or {scene_archive}. "
                "Upload the same DyNeRF export used for training."
            )

    model_dir = Path("/outputs") / model_rel
    point_dir = model_dir / "point_cloud" / f"iteration_{iteration}"
    if not (point_dir / "point_cloud.ply").is_file():
        available = [p.name for p in sorted((model_dir / "point_cloud").glob("iteration_*"))]
        raise FileNotFoundError(
            f"No Wu iteration_{iteration} under {model_dir / 'point_cloud'}. Found: {available}"
        )

    cfg_path = Path("/tmp/wu4dgs_phys4d_render.py")
    cfg_path.write_text(
        _wu4dgs_config_text(
            iterations=iteration,
            coarse_iterations=min(3000, iteration),
            time_resolution=time_resolution,
            bounds=bounds,
        ),
        encoding="utf-8",
    )

    _patch_wu4dgs_aspect_preserving_loader()
    _patch_wu4dgs_forced_aabb(force_aabb)
    render_py = Path("/opt/wu4dgs/render.py")
    render_text = render_py.read_text(encoding="utf-8")
    render_text = render_text.replace(
        "        render_images.append(to8b(rendering).transpose(1,2,0))\n"
        "        render_list.append(rendering)\n",
        "        imageio.imwrite(os.path.join(render_path, '{0:05d}.png'.format(idx)), to8b(rendering).transpose(1,2,0))\n",
    )
    render_text = render_text.replace(
        "            gt_list.append(gt)\n",
        "            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}.png'.format(idx)))\n",
    )
    render_text = render_text.replace(
        "    multithread_write(gt_list, gts_path)\n\n"
        "    multithread_write(render_list, render_path)\n\n"
        "    \n"
        "    imageio.mimwrite(os.path.join(model_path, name, \"ours_{}\".format(iteration), 'video_rgb.mp4'), render_images, fps=30)\n",
        "    print('streamed PNG render complete:', render_path, flush=True)\n",
    )
    render_py.write_text(render_text, encoding="utf-8")

    subprocess.run(
        [
            "python",
            "-u",
            "/opt/wu4dgs/render.py",
            "--source_path",
            str(scene),
            "--model_path",
            str(model_dir),
            "--configs",
            str(cfg_path),
            "--iteration",
            str(iteration),
            *(["--skip_test"] if skip_test else []),
            *(["--skip_video"] if skip_video else []),
        ],
        check=True,
        cwd="/opt/wu4dgs",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    output_volume.commit()
    return (
        f"Wu 4DGS render -> phys4d-gs-output:/{model_rel}/train/ours_{iteration}\n"
        "Native render outputs are renders/*.png, gt/*.png, and video_rgb.mp4."
    )


@app.function(
    image=_wu4dgs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 90,
)
def render_wu_4dgs_composed(
    model_rels: str,
    output_model_rel: str = "wu4dgs_composed",
    iteration: int = 50000,
    time_resolution: int = 120,
    bounds: float = 1.2,
    force_aabb: str = "",
    opacity_threshold: float = 0.0,
    skip_test: bool = True,
    skip_video: bool = True,
) -> str:
    """Render multiple Wu 4DGS models through one rasterizer pass.

    This keeps each object's learned deformation network, deforms every model at
    the current camera timestamp, concatenates the resulting Gaussian tensors,
    and rasterizes them together. That is the practical Wu equivalent of
    composing per-object models in Gaussian splat space.
    """
    import shutil
    import subprocess
    import tarfile

    scene = Path("/data/4d_scene")
    scene_archive = Path("/data/4d_scene.tar.gz")
    if not (scene / "transforms_train.json").is_file():
        if scene_archive.is_file():
            extracted = Path("/tmp/4d_scene")
            if extracted.exists():
                shutil.rmtree(extracted)
            extracted.mkdir(parents=True, exist_ok=True)
            with tarfile.open(scene_archive, "r:gz") as tar:
                tar.extractall(extracted)
            scene = extracted
            print(f"Extracted archived DyNeRF scene from {scene_archive} to {scene}", flush=True)
        else:
            raise FileNotFoundError(
                "No 4D scene at /data/4d_scene or /data/4d_scene.tar.gz. "
                "Upload the same DyNeRF export used for training."
            )

    rels = [rel.strip() for rel in model_rels.split(",") if rel.strip()]
    if len(rels) < 2:
        raise ValueError("--render-wu-4d-compose-models must contain at least two comma-separated models")

    model_dirs = [Path("/outputs") / rel for rel in rels]
    for model_dir in model_dirs:
        point_dir = model_dir / "point_cloud" / f"iteration_{iteration}"
        if not (point_dir / "point_cloud.ply").is_file():
            available = [p.name for p in sorted((model_dir / "point_cloud").glob("iteration_*"))]
            raise FileNotFoundError(
                f"No Wu iteration_{iteration} under {model_dir / 'point_cloud'}. Found: {available}"
            )

    out_dir = Path("/outputs") / output_model_rel
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path("/tmp/wu4dgs_phys4d_compose_render.py")
    cfg_path.write_text(
        _wu4dgs_config_text(
            iterations=iteration,
            coarse_iterations=min(3000, iteration),
            time_resolution=time_resolution,
            bounds=bounds,
        ),
        encoding="utf-8",
    )

    _patch_wu4dgs_aspect_preserving_loader()
    _patch_wu4dgs_forced_aabb(force_aabb)

    script = Path("/tmp/render_wu_composed.py")
    script.write_text(
        r'''
import json
import math
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, "/opt/wu4dgs")

import imageio
import numpy as np
import torch
import torchvision
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


@torch.no_grad()
def render_multi(viewpoint_camera, gaussians, pipe, bg_color, cam_type, opacity_threshold):
    means2d = []
    means3d = []
    scales = []
    rotations = []
    opacities = []
    shs = []

    for pc in gaussians:
        screen = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screen.retain_grad()
        except Exception:
            pass
        # Match Wu's reference single-object renderer: deform opacity and SHs
        # at the camera timestamp before applying activations. The helper
        # utils.render_utils.get_state_at_time returns the base opacity, which
        # makes composed per-object models dark/occluding even when each object
        # renders correctly on its own.
        time = torch.tensor(viewpoint_camera.time).to(pc.get_xyz.device).repeat(pc.get_xyz.shape[0], 1)
        m, s, r, o, h = pc._deformation(
            pc.get_xyz,
            pc._scaling,
            pc._rotation,
            pc._opacity,
            pc.get_features,
            time,
        )
        s = pc.scaling_activation(s)
        r = pc.rotation_activation(r)
        o = pc.opacity_activation(o)
        if opacity_threshold > 0:
            keep = o.squeeze(-1) >= opacity_threshold
            if keep.any():
                screen = screen[keep]
                m = m[keep]
                s = s[keep]
                r = r[keep]
                o = o[keep]
                h = h[keep]
            else:
                continue
        means2d.append(screen)
        means3d.append(m)
        scales.append(s)
        rotations.append(r)
        opacities.append(o)
        shs.append(h)

    if not means3d:
        return torch.zeros((3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)), device="cuda")
    means2d = torch.cat(means2d, dim=0)
    means3d = torch.cat(means3d, dim=0)
    scales = torch.cat(scales, dim=0)
    rotations = torch.cat(rotations, dim=0)
    opacities = torch.cat(opacities, dim=0)
    shs = torch.cat(shs, dim=0)

    if cam_type == "PanopticSports":
        raster_settings = viewpoint_camera["camera"]
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=gaussians[0].active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii, depth = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    return rendered_image


def load_model(base_args, model_path, model, hyper, pipe, iteration):
    args = deepcopy(base_args)
    args.model_path = str(model_path)
    gaussians = GaussianModel(model.extract(args).sh_degree, hyper.extract(args))
    scene = Scene(model.extract(args), gaussians, load_iteration=iteration, shuffle=False)
    return gaussians, scene


def main():
    parser = ArgumentParser(description="Render multiple Wu 4DGS objects in one Gaussian rasterization pass")
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    hyper = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--opacity_threshold", type=float, default=0.0)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)

    model_paths = [Path(p) for p in args.model_paths.split(",") if p]
    if len(model_paths) < 2:
        raise ValueError("Pass at least two model paths")

    gaussians = []
    scenes = []
    for model_path in model_paths:
        g, s = load_model(args, model_path, model, hyper, pipe, args.iteration)
        gaussians.append(g)
        scenes.append(s)

    scene = scenes[0]
    cam_type = scene.dataset_type
    bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = pipe.extract(args)
    out_root = Path(args.output_path)

    manifest = {
        "iteration": args.iteration,
        "model_paths": [str(p) for p in model_paths],
        "num_models": len(model_paths),
        "opacity_threshold": args.opacity_threshold,
        "sets": {},
    }

    def render_set(name, views):
        render_path = out_root / name / f"ours_{args.iteration}" / "renders"
        gt_path = out_root / name / f"ours_{args.iteration}" / "gt"
        render_path.mkdir(parents=True, exist_ok=True)
        gt_path.mkdir(parents=True, exist_ok=True)
        video = []
        for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
            image = render_multi(view, gaussians, pipeline, background, cam_type, args.opacity_threshold)
            torchvision.utils.save_image(image, render_path / f"{idx:05d}.png")
            video.append(to8b(image).transpose(1, 2, 0))
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt = view["image"].cuda()
                torchvision.utils.save_image(gt, gt_path / f"{idx:05d}.png")
        if video:
            imageio.mimwrite(out_root / name / f"ours_{args.iteration}" / "video_rgb.mp4", video, fps=30)
        manifest["sets"][name] = {
            "num_views": len(views),
            "renders": str(render_path),
        }

    if not args.skip_train:
        render_set("train", scene.getTrainCameras())
    if not args.skip_test:
        render_set("test", scene.getTestCameras())
    if not args.skip_video:
        render_set("video", scene.getVideoCameras())

    (out_root / "compose_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote composed Wu render to {out_root}", flush=True)


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )

    subprocess.run(
        [
            "python",
            "-u",
            str(script),
            "--source_path",
            str(scene),
            "--model_path",
            str(model_dirs[0]),
            "--model_paths",
            ",".join(str(path) for path in model_dirs),
            "--output_path",
            str(out_dir),
            "--configs",
            str(cfg_path),
            "--iteration",
            str(iteration),
            "--opacity_threshold",
            str(opacity_threshold),
            *(["--skip_test"] if skip_test else []),
            *(["--skip_video"] if skip_video else []),
        ],
        check=True,
        cwd="/opt/wu4dgs",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    output_volume.commit()
    return (
        f"Composed Wu 4DGS render -> phys4d-gs-output:/{output_model_rel}/train/ours_{iteration}\n"
        f"Models: {', '.join(rels)}\n"
        f"Opacity threshold: {opacity_threshold}\n"
        "This render concatenates deformed Gaussian tensors before rasterization."
    )


@app.function(
    image=_4dgs_image,
    gpu="T4",
    volumes={"/outputs": output_volume},
    timeout=30 * 60,
)
def export_4dgs_ply(
    checkpoint_name: str = "chkpnt_best.pth",
    model_rel: str = "4dgs_sphere_bounce",
) -> str:
    """Write SuperSplat PLY from a trained 4DGS checkpoint on the output volume."""
    import sys

    sys.path.insert(0, FOURDGS_SCRIPTS)
    from export_4dgs_ply import export_checkpoint_to_ply

    model_dir = Path("/outputs") / model_rel
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


@app.function(
    image=_4dgs_image,
    gpu=None,
    volumes={"/outputs": output_volume},
    timeout=30 * 60,
)
def compose_4dgs_checkpoints_remote(
    model_rels: str,
    output_model_rel: str,
    checkpoint_name: str = "chkpnt15000.pth",
    output_checkpoint_name: str = "chkpnt_composed.pth",
) -> str:
    """Compose multiple named output-volume 4DGS models into one checkpoint."""
    import sys

    sys.path.insert(0, FOURDGS_SCRIPTS)
    from compose_4dgs_checkpoints import compose_checkpoints

    rels = [rel.strip().strip("/") for rel in model_rels.split(",") if rel.strip()]
    if len(rels) < 2:
        raise ValueError("--compose-4d-models must contain at least two comma-separated model paths")
    checkpoints = [Path("/outputs") / rel / checkpoint_name for rel in rels]
    output = Path("/outputs") / output_model_rel.strip("/") / output_checkpoint_name
    meta = compose_checkpoints(checkpoints, output)
    output_volume.commit()
    return json.dumps(meta, indent=2)


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
    model_rel: str = "4dgs_sphere_bounce",
) -> str:
    """Rasterize 4DGS at each frame's timestamp; sort by time then camera (motion + orbit)."""
    import shutil

    from omegaconf import OmegaConf

    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(
            "No 4D scene at /data/4d_scene. Run: modal run modal_app.py --upload-4d"
        )

    model_dir = Path("/outputs") / model_rel
    ckpt = _pick_4d_checkpoint(model_dir, checkpoint_name)

    cfg_src = Path(FOURDGS_CONFIGS) / config_name
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
            f"{FOURDGS_SCRIPTS}/render_4dgs_trajectory.py",
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
    timeout=60 * 90,
)
def render_4d_multi_trajectory(
    config_name: str = "room_physics_4dgs_5p0s.yaml",
    model_rels: str = "",
    checkpoint_name: str = "chkpnt15000.pth",
    frame_starts: str = "",
    frame_ends: str = "",
    dry_run_max: int = 0,
    mode: str = "max",
) -> str:
    """Render multiple separately trained 4DGS models into one PNG sequence."""
    import shutil

    from omegaconf import OmegaConf

    scene = Path("/data/4d_scene")
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError("No reference 4D scene at /data/4d_scene. Upload one first.")

    rels = [rel.strip().strip("/") for rel in model_rels.split(",") if rel.strip()]
    if len(rels) < 2:
        raise ValueError("--render-4d-multi-models must contain at least two comma-separated model paths")
    ckpts = [Path("/outputs") / rel / checkpoint_name for rel in rels]
    missing = [str(path) for path in ckpts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    cfg_src = Path(FOURDGS_CONFIGS) / config_name
    if not cfg_src.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_src}")
    cfg = OmegaConf.load(cfg_src)
    cfg.ModelParams.source_path = str(scene)
    cfg_path = Path("/tmp/render_4dgs_multi.yaml")
    OmegaConf.save(cfg, cfg_path)

    out_base = Path("/outputs/4dgs_renders/multi_latest")
    out_base.mkdir(parents=True, exist_ok=True)
    for child in out_base.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    subprocess.run(
        [
            sys.executable,
            f"{FOURDGS_SCRIPTS}/render_4dgs_multi_checkpoint.py",
            "--fourd-root",
            "/opt/4dgs",
            "--config",
            str(cfg_path),
            "--dataset",
            str(scene),
            "--out-dir",
            str(out_base),
            "--checkpoints",
            *[str(path) for path in ckpts],
            "--frame-starts",
            frame_starts,
            "--frame-ends",
            frame_ends,
            "--dry-run-max",
            str(dry_run_max),
            "--mode",
            mode,
        ],
        check=True,
    )

    output_volume.commit()
    return (
        f"multi-4dgs render -> phys4d-gs-output:{out_base}\n"
        f"models: {rels}\ncheckpoint: {checkpoint_name}"
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

    cfg_src = Path(FOURDGS_CONFIGS) / config_name
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
        f"{FOURDGS_SCRIPTS}/render_4dgs_trajectory.py",
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

    cfg_src = Path(FOURDGS_CONFIGS) / config_name
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
        f"{FOURDGS_SCRIPTS}/eval_4dgs_metrics.py",
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


@app.function(
    image=_torch_image,
    volumes={"/outputs": output_volume},
    timeout=60 * 30,
)
def cleanup_output_volume(paths_csv: str) -> str:
    """Delete exact remote output-volume paths to recover inode budget."""
    root = Path("/outputs").resolve()
    removed: list[str] = []
    missing: list[str] = []
    for raw in paths_csv.split(","):
        rel = raw.strip().strip("/")
        if not rel:
            continue
        path = (root / rel).resolve()
        if root not in path.parents:
            raise ValueError(f"Refusing to delete outside /outputs: {rel}")
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(rel)
        elif path.is_file():
            path.unlink()
            removed.append(rel)
        else:
            missing.append(rel)
    output_volume.commit()
    return json.dumps({"removed": removed, "missing": missing}, indent=2)


@app.function(
    image=_pybullet_dataset_image,
    volumes={"/outputs": output_volume},
    cpu=8.0,
    timeout=60 * 60 * 2,
)
def generate_collision_variants_dataset(
    *,
    output_rel: str,
    video_fps: float = 60.0,
    duration_sec: float = 2.6,
    masses: str = "0.22,0.44",
    velocity_modes: str = "asym1.4,equal1.05",
    restitutions: str = "0.98,0.90",
    workers: int = 8,
) -> str:
    """Generate the final 2x2x2 collision PyBullet dataset on Modal."""
    rel = output_rel.strip().strip("/")
    if not rel:
        raise ValueError("output_rel must be non-empty")
    out_dir = (Path("/outputs") / rel).resolve()
    if Path("/outputs").resolve() not in out_dir.parents:
        raise ValueError(f"Refusing to write outside /outputs: {output_rel}")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable,
        "/repo/dataset/generate_collision_variants_final.py",
        "--output-dir",
        str(out_dir),
        "--video-fps",
        str(video_fps),
        "--duration-sec",
        str(duration_sec),
        "--masses",
        masses,
        "--velocity-modes",
        velocity_modes,
        "--restitutions",
        restitutions,
        "--workers",
        str(workers),
    ]
    subprocess.run(cmd, check=True, cwd="/repo")

    report_path = out_dir / "collision_contact_validation.json"
    subprocess.run(
        [
            sys.executable,
            "/repo/scripts/validate_collision_contacts.py",
            str(out_dir),
            "--write-json",
            str(report_path),
        ],
        check=True,
        cwd="/repo",
    )
    output_volume.commit()
    manifest = out_dir / "scenario_manifest.json"
    return (
        f"generated collision dataset -> phys4d-gs-output:/{rel}\n"
        f"manifest: {manifest}\n"
        f"validation: {report_path}"
    )


@app.local_entrypoint()
def main(
    tests: bool = False,
    upload: bool = False,
    train: bool = False,
    upload_4d: bool = False,
    train_4d: bool = False,
    train_wu_4d: bool = False,
    compose_4d: bool = False,
    export_4d_ply: bool = False,
    render_4d: bool = False,
    render_wu_4d: bool = False,
    render_wu_4d_compose: bool = False,
    render_4d_multi: bool = False,
    render_4d_orbit: bool = False,
    eval_4d: bool = False,
    upload_batch: bool = False,
    extract_perception: bool = False,
    train_visual_dynamics: bool = False,
    visual_states_only: bool = False,
    generate_collision_variants: bool = False,
    cleanup_output: bool = False,
    cleanup_output_paths: str = "",
    upload_visual_pipeline: bool = False,
    visual_pipeline: bool = False,
    batch_rel: str = "sphere_bounce_batch",
    vd_manifest: str = "sphere_bounce_batch/dataset_manifest.json",
    scene_rel: str = "scene",
    pipeline_out_rel: str = "visual_dynamics_pipeline",
    perception_limit: int = 0,
    collision_output_rel: str = "generated_datasets/collision_scale1p75_elastic_2x2x2_60fps",
    collision_local_output: str = "dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps",
    collision_video_fps: float = 60.0,
    collision_duration_sec: float = 2.6,
    collision_masses: str = "0.22,0.44",
    collision_velocity_modes: str = "asym1.4,equal1.05",
    collision_restitutions: str = "0.98,0.90",
    collision_workers: int = 8,
    modal_cmd: str = "arch -arm64 modal",
    render_4d_checkpoint: str | None = None,
    train_4d_config: str = "sphere_bounce_4dgs.yaml",
    train_4d_model: str = "4dgs_sphere_bounce",
    train_wu_4d_model: str = "wu4dgs_sphere_bounce",
    wu_scene_rel: str = "4d_scene",
    wu_iterations: int = 15000,
    wu_coarse_iterations: int = 3000,
    wu_time_resolution: int = 75,
    wu_bounds: float = 1.6,
    wu_foreground_loss_weight: float = 0.0,
    wu_mask_loss_weight: float = 0.0,
    wu_bg_spill_loss_weight: float = 0.0,
    wu_area_loss_weight: float = 0.0,
    wu_compactness_loss_weight: float = 0.0,
    wu_scale_isotropy_loss_weight: float = 0.0,
    wu_max_scale_loss_weight: float = 0.0,
    wu_max_gaussian_scale: float = 0.02,
    wu_cloud_isotropy_loss_weight: float = 0.0,
    wu_silhouette_roundness_loss_weight: float = 0.0,
    wu_trajectory_anchor_loss_weight: float = 0.0,
    wu_camera_loss_weights: str = "",
    wu_densify_until_iter: int = 0,
    wu_opacity_reset_interval: int = 0,
    wu_start_checkpoint: str = "",
    wu_force_aabb: str = "",
    render_4d_config: str = "sphere_bounce_4dgs.yaml",
    render_4d_model: str = "4dgs_sphere_bounce",
    render_wu_4d_model: str = "wu4dgs_sphere_bounce",
    render_wu_4d_iteration: int = 15000,
    render_wu_4d_compose_models: str = "",
    render_wu_4d_compose_output_model: str = "wu4dgs_composed",
    render_wu_4d_compose_opacity_threshold: float = 0.0,
    render_4d_multi_models: str = "",
    render_4d_multi_frame_starts: str = "",
    render_4d_multi_frame_ends: str = "",
    render_4d_multi_mode: str = "max",
    compose_4d_models: str = "",
    compose_4d_output_model: str = "4dgs_composed",
    compose_4d_checkpoint: str = "chkpnt15000.pth",
    compose_4d_output_checkpoint: str = "chkpnt_composed.pth",
    render_fps: float = 60.0,
    render_dry_run_max: int = 0,
    upload_4d_path: str = "outputs/sphere_bounce_m2/dynerf_sphere_bounce",
    orbit_frames: int = 180,
    orbit_time_start: float | None = None,
    orbit_time_end: float | None = None,
    eval_4d_dry_run_max: int = 0,
    frame: int = 0,
    iterations: int = 7000,
) -> None:
    if generate_collision_variants:
        print(
            generate_collision_variants_dataset.remote(
                output_rel=collision_output_rel,
                video_fps=collision_video_fps,
                duration_sec=collision_duration_sec,
                masses=collision_masses,
                velocity_modes=collision_velocity_modes,
                restitutions=collision_restitutions,
                workers=collision_workers,
            )
        )
        local_out = (REPO_ROOT / collision_local_output).resolve()
        local_out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *modal_cmd.split(),
                "volume",
                "get",
                "phys4d-gs-output",
                collision_output_rel.strip("/"),
                str(local_out),
                "--force",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        print(f"Downloaded collision dataset -> {local_out}")
        return
    if cleanup_output:
        print(cleanup_output_volume.remote(paths_csv=cleanup_output_paths))
        return
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
        out_scene = (REPO_ROOT / upload_4d_path).resolve()
        default_scene = (REPO_ROOT / "outputs" / "sphere_bounce_m2" / "dynerf_sphere_bounce").resolve()
        if out_scene == default_scene:
            export_script = REPO_ROOT / "4dgs" / "scripts" / "export_4dgs_dataset.py"
            subprocess.run(
                [sys.executable, str(export_script), "--output", str(out_scene)],
                check=True,
                cwd=str(REPO_ROOT),
            )
        elif not (out_scene / "transforms_train.json").is_file():
            raise FileNotFoundError(
                f"Custom 4D scene is missing transforms_train.json: {out_scene}"
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
        print(train_4dgs.remote(config_name=train_4d_config, model_rel=train_4d_model))
        print(
            f"Model: phys4d-gs-output:/{train_4d_model}\n"
            f"Raster (calibrated): modal run modal_app.py --render-4d --render-4d-model {train_4d_model}\n"
            "Raster (orbit):      modal run modal_app.py --render-4d-orbit\n"
            "Metrics (PSNR/GT):   modal run modal_app.py --eval-4d\n"
            "Export PLY: modal run modal_app.py --export-4d-ply\n"
            f"Download: modal volume get phys4d-gs-output {train_4d_model} . --force"
        )
        return
    if train_wu_4d:
        print(
            train_wu_4dgs.remote(
                model_rel=train_wu_4d_model,
                scene_rel=wu_scene_rel,
                iterations=wu_iterations,
                coarse_iterations=wu_coarse_iterations,
                time_resolution=wu_time_resolution,
                bounds=wu_bounds,
                foreground_loss_weight=wu_foreground_loss_weight,
                mask_loss_weight=wu_mask_loss_weight,
                bg_spill_loss_weight=wu_bg_spill_loss_weight,
                area_loss_weight=wu_area_loss_weight,
                compactness_loss_weight=wu_compactness_loss_weight,
                scale_isotropy_loss_weight=wu_scale_isotropy_loss_weight,
                max_scale_loss_weight=wu_max_scale_loss_weight,
                max_gaussian_scale=wu_max_gaussian_scale,
                cloud_isotropy_loss_weight=wu_cloud_isotropy_loss_weight,
                silhouette_roundness_loss_weight=wu_silhouette_roundness_loss_weight,
                trajectory_anchor_loss_weight=wu_trajectory_anchor_loss_weight,
                camera_loss_weights=wu_camera_loss_weights,
                densify_until_iter=wu_densify_until_iter,
                opacity_reset_interval=wu_opacity_reset_interval,
                start_checkpoint=wu_start_checkpoint,
                force_aabb=wu_force_aabb,
            )
        )
        print(
            f"Model: phys4d-gs-output:/{train_wu_4d_model}\n"
            f"Render: arch -arm64 modal run modal_app.py --render-wu-4d "
            f"--render-wu-4d-model {train_wu_4d_model} "
            f"--render-wu-4d-iteration {wu_iterations}\n"
            f"Download: arch -arm64 modal volume get phys4d-gs-output "
            f"{train_wu_4d_model} . --force"
        )
        return
    if compose_4d:
        print(
            compose_4dgs_checkpoints_remote.remote(
                model_rels=compose_4d_models,
                output_model_rel=compose_4d_output_model,
                checkpoint_name=compose_4d_checkpoint,
                output_checkpoint_name=compose_4d_output_checkpoint,
            )
        )
        print(
            "Render composed model:\n"
            f"  modal run modal_app.py --render-4d --render-4d-model {compose_4d_output_model} "
            f"--render-4d-checkpoint {compose_4d_output_checkpoint}"
        )
        return
    if export_4d_ply:
        print(
            export_4dgs_ply.remote(
                checkpoint_name=render_4d_checkpoint,
                model_rel=render_4d_model,
            )
        )
        print(
            "Download: modal volume get phys4d-gs-output "
            f"{render_4d_model}/point_cloud/exported . --force"
        )
        return
    if render_4d:
        print(
            render_4d_trajectory.remote(
                config_name=render_4d_config,
                checkpoint_name=render_4d_checkpoint,
                fps=render_fps,
                dry_run_max=render_dry_run_max,
                model_rel=render_4d_model,
            )
        )
        print(
            "Download video: modal volume get phys4d-gs-output 4dgs_renders/latest . --force\n"
            "  Play trajectory.mp4; PNGs sorted by simulation time then camera."
        )
        return
    if render_wu_4d:
        print(
            render_wu_4dgs.remote(
                model_rel=render_wu_4d_model,
                scene_rel=wu_scene_rel,
                iteration=render_wu_4d_iteration,
                time_resolution=wu_time_resolution,
                bounds=wu_bounds,
                force_aabb=wu_force_aabb,
            )
        )
        print(
            "Download Wu render/model: arch -arm64 modal volume get phys4d-gs-output "
            f"{render_wu_4d_model} . --force"
        )
        return
    if render_wu_4d_compose:
        print(
            render_wu_4dgs_composed.remote(
                model_rels=render_wu_4d_compose_models,
                output_model_rel=render_wu_4d_compose_output_model,
                iteration=render_wu_4d_iteration,
                time_resolution=wu_time_resolution,
                bounds=wu_bounds,
                force_aabb=wu_force_aabb,
                opacity_threshold=render_wu_4d_compose_opacity_threshold,
            )
        )
        print(
            "Download composed Wu render: arch -arm64 modal volume get phys4d-gs-output "
            f"{render_wu_4d_compose_output_model} . --force"
        )
        return
    if render_4d_multi:
        print(
            render_4d_multi_trajectory.remote(
                config_name=render_4d_config,
                model_rels=render_4d_multi_models,
                checkpoint_name=render_4d_checkpoint or "chkpnt15000.pth",
                frame_starts=render_4d_multi_frame_starts,
                frame_ends=render_4d_multi_frame_ends,
                dry_run_max=render_dry_run_max,
                mode=render_4d_multi_mode,
            )
        )
        print(
            "Download Gaussian-rendered PNGs: modal volume get phys4d-gs-output "
            "4dgs_renders/multi_latest latest_multi_4dgs_render --force"
        )
        return
    if render_4d_orbit:
        print(
            render_4d_orbit_job.remote(
                config_name=render_4d_config,
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
                config_name=render_4d_config,
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
