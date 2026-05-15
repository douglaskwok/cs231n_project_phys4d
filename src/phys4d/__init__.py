"""Lightweight physics utilities for the CS231N Phys4D milestone."""

from .differentiable_bounce import (
    BounceConfig,
    RecoveryResult,
    fit_restitution,
    restitution_from_raw,
    simulate_bounce,
)

__all__ = [
    "BounceConfig",
    "RecoveryResult",
    "fit_restitution",
    "restitution_from_raw",
    "simulate_bounce",
]
