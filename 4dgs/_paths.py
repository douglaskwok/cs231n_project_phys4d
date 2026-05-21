"""Shared paths for the 4dgs package (repo root + local configs/scripts)."""

from pathlib import Path

FOURDGS_DIR = Path(__file__).resolve().parent
REPO_ROOT = FOURDGS_DIR.parent
CONFIGS_DIR = FOURDGS_DIR / "configs"
SCRIPTS_DIR = FOURDGS_DIR / "scripts"
EXPERIMENTS_DIR = FOURDGS_DIR / "experiments"
