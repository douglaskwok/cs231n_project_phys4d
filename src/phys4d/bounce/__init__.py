"""Ball-bounce pipeline: 4DGS trajectory extraction and physics extrapolation."""

from .extract import Phase1Result, run_phase1
from .extrapolate import Phase3Result, run_phase3
from .load_4dgs import Loaded4DGS, deformed_gaussians_at_time, load_4dgs_checkpoint
from .physics import Phase2Result, PhysicsParams, run_phase2

__all__ = [
    "Loaded4DGS",
    "Phase1Result",
    "Phase2Result",
    "Phase3Result",
    "PhysicsParams",
    "deformed_gaussians_at_time",
    "load_4dgs_checkpoint",
    "run_phase1",
    "run_phase2",
    "run_phase3",
]
