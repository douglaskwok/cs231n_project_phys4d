"""Modal GPU jobs for the Phys4D bounce pipeline (wu-pred).

This app covers the two GPU-only gaps in Step 5 (real Gaussian compositing):
  1. train_bg          -> fit a STATIC background 3DGS (graphdeco) -> background.ply
  2. bounce_step5_remote -> translate object Gaussians by the predicted per-frame
                            delta, merge with the background, and rasterize the
                            held-out cameras with the Wu CUDA renderer.

Setup (once):
  pip install modal
  python -m modal setup

Typical flow (paths are LOCAL; uploads land on the phys4d-gs-data volume):
  # (a) background 3DGS
  modal run modal_app.py --upload-bg --bg-dir outputs/bounce_pipeline/<scene>/background_export
  modal run modal_app.py --train-bg --iterations 7000
  modal volume get phys4d-gs-output background_3dgs/point_cloud/iteration_7000/point_cloud.ply \
      <scene>/background_3dgs/background.ply --force

  # (b) real Step 5 render (after object PLY + step4a/step4c + export are uploaded)
  modal run modal_app.py --upload-step5 --step5-dir outputs/bounce_pipeline/<scene>
  modal run modal_app.py --step5
  modal volume get phys4d-gs-output step5 outputs/bounce_pipeline/<scene>/step5 --force

NOTE: the Wu CUDA image (_wu_image) is built from hustvl/4DGaussians submodules; the
first build is slow and may need an arch/ABI tweak. We debug that on the first GPU run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

app = modal.App("phys4d-bounce")

data_volume = modal.Volume.from_name("phys4d-gs-data", create_if_missing=True)
output_volume = modal.Volume.from_name("phys4d-gs-output", create_if_missing=True)

_repo_ignore = [".git", "outputs", ".venv", "__pycache__", "*.pyc", ".DS_Store", "**/.DS_Store"]

# graphdeco 3DGS (static background). Proven image: builds diff-gaussian-rasterization
# + simple-knn from the official repo. Image build has no GPU, so pin TORCH_CUDA_ARCH_LIST.
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

# Wu / hustvl 4DGaussians rasterizer for Step 5 compositing. Provides the
# `diff_gaussian_rasterization` + `simple_knn` packages that render_merged_ply_wu needs.
# WU_4DGS_ROOT points at our pinned third_party copy (matches the trained checkpoints).
_wu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0")
    .pip_install(
        "plyfile", "tqdm", "opencv-python-headless", "numpy<2", "ninja",
        "imageio", "scipy", "matplotlib", "mmcv", "torchvision", "open3d",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.6+PTX"})
    .run_commands(
        "git clone https://github.com/hustvl/4DGaussians /opt/wu_src",
        "cd /opt/wu_src && git submodule update --init --recursive",
        "pip install /opt/wu_src/submodules/depth-diff-gaussian-rasterization",
        "pip install /opt/wu_src/submodules/simple-knn",
    )
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/repo/src", ignore=["__pycache__", ".DS_Store"])
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/repo/scripts", ignore=["__pycache__", ".DS_Store"])
    .add_local_dir(
        str(REPO_ROOT / "third_party" / "4DGaussians"),
        remote_path="/repo/third_party/4DGaussians",
        ignore=["__pycache__", ".DS_Store", ".git", "submodules/**"],
    )
)


@app.function(image=_gs_image, gpu="T4", timeout=600)
def smoke_gpu() -> dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available inside container")
    return {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}


@app.function(
    image=_gs_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 3,
)
def train_bg(iterations: int = 7000, scene_rel: str = "bg_scene", out_rel: str = "background_3dgs") -> str:
    """Train a static background 3DGS on /data/<scene_rel> (ball masked out)."""
    import shutil

    scene = Path("/data") / scene_rel
    if not (scene / "transforms_train.json").is_file():
        raise FileNotFoundError(f"No background scene at {scene}. Run --upload-bg first.")

    model_dir = Path("/outputs") / out_rel
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "/opt/gs/train.py",
        "-s", str(scene), "-m", str(model_dir),
        "--iterations", str(iterations), "--white_background",
    ]
    try:
        subprocess.run([*cmd, "--disable_viewer"], check=True, cwd="/opt/gs")
    except subprocess.CalledProcessError:
        subprocess.run(cmd, check=True, cwd="/opt/gs")

    output_volume.commit()
    plys = sorted((model_dir / "point_cloud").glob("iteration_*")) if (model_dir / "point_cloud").is_dir() else []
    last = plys[-1].name if plys else "none"
    return f"background 3DGS -> phys4d-gs-output:{model_dir} ({last})"


@app.function(
    image=_wu_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 3,
)
def bounce_step5_remote(
    canonical_rel: str,
    cfg_args_rel: str,
    bg_ply_rel: str = "background_3dgs/point_cloud/iteration_7000/point_cloud.ply",
    out_rel: str = "step5",
    object_scale: float = 0.0,
    object_crop_radius: float = 0.0,
    cameras: str = "",
    skip_existing: bool = False,
) -> str:
    """Run the real Wu-rasterizer Step 5 compositing on uploaded /data/step5 inputs."""
    base = Path("/data/step5")
    if not (base / "export" / "transforms_test.json").is_file():
        raise FileNotFoundError("No /data/step5/export/transforms_test.json. Run --upload-step5 first.")

    out_dir = Path("/outputs") / out_rel
    env = {
        **os.environ,
        "PYTHONPATH": "/repo/src:/repo/third_party/4DGaussians",
        "WU_4DGS_ROOT": "/repo/third_party/4DGaussians",
    }
    cmd = [
        sys.executable, "/repo/scripts/bounce/step5_render.py",
        "--canonical", str(base / canonical_rel),
        "--bg-ply", str(Path("/outputs") / bg_ply_rel),
        "--predicted", str(base / "step4c" / "trajectory_predicted.csv"),
        "--ref-traj", str(base / "step4a" / "trajectory_smoothed.csv"),
        "--dynerf-export", str(base / "export"),
        "--cfg-args", str(base / cfg_args_rel),
        "--scene-dir", str(base / "scene"),
        "--out", str(out_dir),
    ]
    if object_scale and object_scale > 0:
        cmd += ["--object-scale", str(object_scale)]
    if object_crop_radius and object_crop_radius > 0:
        cmd += ["--object-crop-radius", str(object_crop_radius)]
    if cameras:
        cmd += ["--cameras", cameras]
    if skip_existing:
        cmd += ["--skip-existing"]
    subprocess.run(cmd, check=True, env=env)
    output_volume.commit()
    meta = out_dir / "step5_meta.json"
    text = meta.read_text(encoding="utf-8") if meta.is_file() else "(no meta)"
    return f"step5 -> phys4d-gs-output:{out_dir}\n{text}"


@app.function(
    image=_wu_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/outputs": output_volume},
    timeout=60 * 60 * 2,
)
def bounce_step4a_remote(
    out_rel: str = "step4a",
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> str:
    """Run Step 4a trajectory extraction on uploaded /data/step4a inputs (needs Wu CUDA exts).

    Expected layout under /data/step4a (from --upload-step4a):
      object/point_cloud.ply, object/deformation.pth, object/deformation_table.pth,
      object/cfg_args
      export/frame_map.json
      scene/object_poses.csv
    """
    base = Path("/data/step4a")
    if not (base / "object" / "point_cloud.ply").is_file():
        raise FileNotFoundError("No /data/step4a/object/point_cloud.ply. Run --upload-step4a first.")
    if not (base / "export" / "frame_map.json").is_file():
        raise FileNotFoundError("No /data/step4a/export/frame_map.json.")

    out_dir = Path("/outputs") / out_rel
    env = {
        **os.environ,
        "PYTHONPATH": "/repo/src:/repo/third_party/4DGaussians",
        "WU_4DGS_ROOT": "/repo/third_party/4DGaussians",
        "MPLCONFIGDIR": "/tmp/mpl",
    }
    cmd = [
        sys.executable, "/repo/scripts/bounce/step4a_extract.py",
        "--canonical", str(base / "object" / "point_cloud.ply"),
        "--deform", str(base / "object"),
        "--cfg-args", str(base / "object" / "cfg_args"),
        "--dynerf-export", str(base / "export"),
        "--gt-poses", str(base / "scene" / "object_poses.csv"),
        "--out", str(out_dir),
        "--device", "cuda",
        "--savgol-window", str(savgol_window),
        "--savgol-polyorder", str(savgol_polyorder),
    ]
    subprocess.run(cmd, check=True, env=env)
    output_volume.commit()
    meta = out_dir / "step4a_meta.json"
    text = meta.read_text(encoding="utf-8") if meta.is_file() else "(no meta)"
    return f"step4a -> phys4d-gs-output:{out_dir}\n{text}"


def _volume_put(local: Path, remote: str, *, volume: str = "phys4d-gs-data") -> None:
    subprocess.run(["modal", "volume", "rm", volume, remote, "-r"], check=False)
    subprocess.run(["modal", "volume", "put", volume, str(local), remote], check=True)


@app.local_entrypoint()
def main(
    upload_bg: bool = False,
    train_bg_job: bool = False,
    upload_step5: bool = False,
    step5: bool = False,
    upload_step4a: bool = False,
    step4a: bool = False,
    step4a_dir: str = "",
    out_rel: str = "step4a",
    bg_dir: str = "",
    bg_name: str = "huge",
    step5_dir: str = "",
    canonical_rel: str = "object/point_cloud.ply",
    cfg_args_rel: str = "object/cfg_args",
    bg_ply_rel: str = "background_3dgs/point_cloud/iteration_7000/point_cloud.ply",
    iterations: int = 7000,
    object_scale: float = 0.0,
    object_crop_radius: float = 0.0,
    cameras: str = "",
    skip_existing: bool = False,
) -> None:
    if upload_bg:
        src = (REPO_ROOT / bg_dir).resolve()
        if not (src / "transforms_train.json").is_file():
            raise FileNotFoundError(f"--bg-dir missing transforms_train.json: {src}")
        _volume_put(src, f"bg_{bg_name}")
        print(f"Uploaded {src} -> phys4d-gs-data:/bg_{bg_name}")
        return
    if train_bg_job:
        out_rel = f"bg_{bg_name}_3dgs"
        print(train_bg.remote(iterations=iterations, scene_rel=f"bg_{bg_name}", out_rel=out_rel))
        print(
            "Download: modal volume get phys4d-gs-output "
            f"{out_rel}/point_cloud/iteration_{iterations}/point_cloud.ply background.ply --force"
        )
        return
    if upload_step4a:
        src = (REPO_ROOT / step4a_dir).resolve()
        if not (src / "object" / "point_cloud.ply").is_file():
            raise FileNotFoundError(f"--step4a-dir missing object/point_cloud.ply: {src}")
        if not (src / "export" / "frame_map.json").is_file():
            raise FileNotFoundError(f"--step4a-dir missing export/frame_map.json: {src}")
        _volume_put(src, "step4a")
        print(f"Uploaded {src} -> phys4d-gs-data:/step4a")
        return
    if step4a:
        print(bounce_step4a_remote.remote(out_rel=out_rel))
        print(f"Download: modal volume get phys4d-gs-output {out_rel} <local> --force")
        return
    if upload_step5:
        src = (REPO_ROOT / step5_dir).resolve()
        if not (src / "export" / "transforms_test.json").is_file():
            raise FileNotFoundError(f"--step5-dir missing export/transforms_test.json: {src}")
        _volume_put(src, "step5")
        print(f"Uploaded {src} -> phys4d-gs-data:/step5")
        return
    if step5:
        print(
            bounce_step5_remote.remote(
                canonical_rel=canonical_rel,
                cfg_args_rel=cfg_args_rel,
                bg_ply_rel=bg_ply_rel,
                object_scale=object_scale,
                object_crop_radius=object_crop_radius,
                cameras=cameras,
                skip_existing=skip_existing,
            )
        )
        print("Download: modal volume get phys4d-gs-output step5 step5 --force")
        return
    print(smoke_gpu.remote())
