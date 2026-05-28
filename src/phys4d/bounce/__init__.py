"""Ball-bounce pipeline: 4DGS trajectory extraction and physics extrapolation."""

from .extract import Phase1Result, run_phase1
from .extrapolate import Phase3Result, run_phase3
from .load_4dgs import Loaded4DGS, deformed_gaussians_at_time, load_4dgs_checkpoint
from .metrics import Phase5Result, run_phase5
from .physics import Phase2Result, PhysicsParams, run_phase2
from .render_compose import Phase4Result, run_phase4

__all__ = [
    "Loaded4DGS",
    "Phase1Result",
    "Phase2Result",
    "Phase3Result",
    "Phase4Result",
    "Phase5Result",
    "PhysicsParams",
    "deformed_gaussians_at_time",
    "load_4dgs_checkpoint",
    "run_phase1",
    "run_phase2",
    "run_phase3",
    "run_phase4",
    "run_phase5",
]
