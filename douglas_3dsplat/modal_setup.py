"""Modal setup for Douglas' 3D Gaussian Splatting experiments.

Run from the repository root:

    modal run douglas_3dsplat/modal_setup.py

This builds a CUDA/PyTorch image with gsplat installed as a reference backend.
Your own implementation can live in the ``douglas_3dsplat`` package and be
uploaded into the Modal container by this app.
"""

from __future__ import annotations

from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("douglas-3dsplat")
output_volume = modal.Volume.from_name("douglas-3dsplat-output", create_if_missing=True)

# Match the CUDA/PyTorch family already used by modal_app.py in this repo.
# gsplat's prebuilt wheel URL must match this PyTorch/CUDA pair: pt22cu121.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2",
        "ninja",
        "jaxtyping",
        "rich",
        "pillow",
        "imageio",
        "matplotlib",
        "plyfile",
        "tqdm",
    )
    .run_commands(
        "pip install gsplat --index-url https://docs.gsplat.studio/whl/pt22cu121"
    )
    .add_local_dir(
        str(REPO_ROOT / "douglas_3dsplat"),
        remote_path="/repo/douglas_3dsplat",
        ignore=["__pycache__", ".DS_Store", "*.egg-info", "build", "dist"],
    )
    .env({"PYTHONPATH": "/repo/douglas_3dsplat"})
)


@app.function(image=image, gpu="T4", timeout=10 * 60)
def check_gpu_and_gsplat() -> dict[str, str]:
    """Smoke-test the remote environment."""
    import gsplat
    import torch
    import douglas_3dsplat

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal container")

    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gsplat": getattr(gsplat, "__version__", "unknown"),
        "douglas_3dsplat": douglas_3dsplat.__version__,
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/outputs": output_volume},
    timeout=60 * 60,
)
def train_gsplat_baseline(
    steps: int = 300,
    num_gaussians: int = 1500,
    downscale: int = 4,
) -> dict:
    """Train a small static 3DGS baseline using gsplat directly."""
    from pathlib import Path

    from douglas_3dsplat.baseline_gsplat import train_baseline

    metrics = train_baseline(
        data_dir=Path("/repo/douglas_3dsplat/data/ping_pong_12view_frame_0000"),
        output_dir=Path("/outputs/gsplat_baseline_frame_0000"),
        steps=steps,
        num_gaussians=num_gaussians,
        downscale=downscale,
    )
    output_volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    train_baseline: bool = False,
    steps: int = 300,
    num_gaussians: int = 1500,
    downscale: int = 4,
) -> None:
    if train_baseline:
        result = train_gsplat_baseline.remote(
            steps=steps,
            num_gaussians=num_gaussians,
            downscale=downscale,
        )
    else:
        result = check_gpu_and_gsplat.remote()
    print(result)
