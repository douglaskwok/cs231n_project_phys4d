"""Packaging setup for Douglas' 3D Gaussian Splatting experiments.

From the repository root:

    pip install -e douglas_3dsplat

If you want gsplat as a reference implementation/backend, install the extra:

    pip install -e "douglas_3dsplat[ref]"

For Modal's current PyTorch 2.2 / CUDA 12.1 image, a matching prebuilt gsplat
wheel can be installed with:

    pip install ninja numpy jaxtyping rich
    pip install gsplat --index-url https://docs.gsplat.studio/whl/pt22cu121

The project itself should stay independent enough that you can implement the
math yourself and only use gsplat for comparison, profiling, or sanity checks.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="douglas-3dsplat",
    version="0.1.0",
    description="A from-scratch learning implementation of 3D Gaussian Splatting.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8")
    if (ROOT / "README.md").is_file()
    else "",
    long_description_content_type="text/markdown",
    author="Douglas",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "torch>=2.2",
        "torchvision>=0.17",
        "pillow",
        "imageio",
        "matplotlib",
        "plyfile",
        "tqdm",
        "rich",
        "jaxtyping",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
        ],
        "ref": [
            "gsplat>=1.5",
            "ninja",
        ],
    },
)
