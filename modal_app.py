"""Modal GPU jobs for CS231N Phys4D.

Prerequisites:
  pip install -r requirements-modal.txt
  modal setup

Local prep (uses **your** PyBullet RGB under outputs/sphere_bounce_m2/rgb/):
  python scripts/generate_sphere_bounce_dataset.py
  python scripts/export_gs_blender_scene.py

Modal:
  modal run modal_app.py                    # GPU smoke
  modal run modal_app.py --tests            # remote unittest
  modal run modal_app.py --upload           # upload Blender scene to Volume
  modal run modal_app.py --train            # 3DGS on uploaded scene (uses your images)

Video note: default export is **one timestep, all cameras** (static 3DGS). Full
multi-frame video needs 4DGS or per-frame training — not this train job yet.
"""

from __future__ import annotations

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

_repo_ignore = [
    ".git",
    "outputs",
    "phys_sim",
    ".venv",
    "__pycache__",
    "*.pyc",
    ".ipynb_checkpoints",
]

_test_image = _torch_image.add_local_dir(
    str(REPO_ROOT),
    remote_path="/repo",
    ignore=_repo_ignore,
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
        [sys.executable, "-m", "unittest", "tests/test_restitution_recovery.py", "-v"],
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


@app.local_entrypoint()
def main(
    tests: bool = False,
    upload: bool = False,
    train: bool = False,
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
    print(smoke_gpu.remote())
