"""Multi-body collision physics (collision pipeline, Step 4b/4c).

Generalises the single-body floor-bounce integrator in ``phys4d.bounce.physics``
to N rigid spheres that can collide with each other. Each body keeps the bounce
model's floor contact (vertical restitution, frictionless horizontal), and pairs
of bodies resolve contact with an impulse along the line of centres using
conservation of momentum + a pair restitution coefficient.

The integrator sub-steps between sampled times so fast bodies cannot tunnel
through each other at the dataset frame rate (the #1 robustness risk here).

CSV-only: this module never imports Wu/4DGS code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Re-export the trajectory CSV loader so callers can use a single import surface.
from phys4d.bounce.physics import load_trajectory_csv  # noqa: F401


@dataclass(frozen=True)
class Body:
    """A single rigid sphere in the collision scene."""

    p0: np.ndarray  # (3,) initial position at the anchor time
    v0: np.ndarray  # (3,) initial velocity
    radius: float
    mass: float
    restitution_floor: float


@dataclass(frozen=True)
class SceneParams:
    """Fitted multi-body simulator parameters."""

    bodies: tuple[Body, ...]
    gravity_z: float
    ground_z: float
    restitution_pair: float


def simulate_scene(
    times: np.ndarray,
    *,
    bodies: list[Body] | tuple[Body, ...],
    gravity_z: float,
    ground_z: float,
    restitution_pair: float,
    substeps: int = 8,
    wall_x: tuple[float, float] | None = None,
    wall_y: tuple[float, float] | None = None,
    restitution_wall: float = 1.0,
) -> np.ndarray:
    """Forward-simulate N spheres with floor + optional walls + pairwise collisions.

    Returns an ``(n_times, n_bodies, 3)`` array of positions sampled at ``times``.
    Gravity acts only on ``z``; horizontal velocity is conserved across floor
    contacts (frictionless). Pairwise contacts apply a normal impulse only.

    ``wall_x`` / ``wall_y`` are ``(min, max)`` wall-surface positions (metric). A
    body reflects its ``x`` (resp. ``y``) velocity with ``restitution_wall`` when
    its surface (center ± radius) reaches a wall, modelling the room enclosure.
    Pass ``None`` to disable that axis' walls.
    """

    times = np.asarray(times, dtype=np.float64)
    n = int(times.shape[0])
    nb = len(bodies)
    if nb == 0:
        raise ValueError("simulate_scene needs at least one body")
    substeps = max(1, int(substeps))

    radius = np.array([b.radius for b in bodies], dtype=np.float64)
    inv_mass = np.array([1.0 / float(b.mass) for b in bodies], dtype=np.float64)
    e_floor = np.array([b.restitution_floor for b in bodies], dtype=np.float64)

    p = np.array([np.asarray(b.p0, dtype=np.float64).reshape(3) for b in bodies], dtype=np.float64)
    v = np.array([np.asarray(b.v0, dtype=np.float64).reshape(3) for b in bodies], dtype=np.float64)

    out = np.empty((n, nb, 3), dtype=np.float64)
    e_pair = float(restitution_pair)
    e_wall = float(restitution_wall)
    walls = {0: wall_x, 1: wall_y}

    def _reflect_walls(axis: int, bound: tuple[float, float] | None) -> None:
        if bound is None:
            return
        lo_b, hi_b = float(bound[0]), float(bound[1])
        lo_eff = lo_b + radius
        hi_eff = hi_b - radius
        hit_lo = (p[:, axis] < lo_eff) & (v[:, axis] < 0.0)
        if np.any(hit_lo):
            v[hit_lo, axis] = -e_wall * v[hit_lo, axis]
            p[hit_lo, axis] = lo_eff[hit_lo]
        hit_hi = (p[:, axis] > hi_eff) & (v[:, axis] > 0.0)
        if np.any(hit_hi):
            v[hit_hi, axis] = -e_wall * v[hit_hi, axis]
            p[hit_hi, axis] = hi_eff[hit_hi]

    for i in range(n):
        out[i] = p
        if i + 1 >= n:
            break
        dt = float(times[i + 1] - times[i])
        if dt <= 0.0:
            raise ValueError("times must be strictly increasing")
        h = dt / substeps
        for _ in range(substeps):
            v[:, 2] += float(gravity_z) * h
            p = p + v * h

            # --- floor contact (per body, vertical only) ---
            below = (p[:, 2] < ground_z) & (v[:, 2] < 0.0)
            if np.any(below):
                v[below, 2] = -e_floor[below] * v[below, 2]
                p[below, 2] = ground_z

            # --- wall contact (axis-aligned room enclosure) ---
            _reflect_walls(0, walls[0])
            _reflect_walls(1, walls[1])

            # --- pairwise sphere-sphere contact ---
            for a in range(nb):
                for b in range(a + 1, nb):
                    d = p[a] - p[b]
                    dist = float(np.linalg.norm(d))
                    r_sum = float(radius[a] + radius[b])
                    if dist >= r_sum or dist <= 1e-12:
                        continue
                    nrm = d / dist
                    inv_sum = float(inv_mass[a] + inv_mass[b])
                    v_rel = float(np.dot(v[a] - v[b], nrm))
                    if v_rel < 0.0:  # approaching
                        j = -(1.0 + e_pair) * v_rel / inv_sum
                        v[a] = v[a] + (j * inv_mass[a]) * nrm
                        v[b] = v[b] - (j * inv_mass[b]) * nrm
                    # positional de-penetration (split by inverse mass)
                    corr = (r_sum - dist) / inv_sum
                    p[a] = p[a] + (corr * inv_mass[a]) * nrm
                    p[b] = p[b] - (corr * inv_mass[b]) * nrm

    return out


def count_pair_collisions(
    positions: np.ndarray,
    radii: np.ndarray | list[float],
) -> int:
    """Count frames where any sphere pair is in contact (overlap onset)."""

    pos = np.asarray(positions, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    n, nb, _ = pos.shape
    in_contact_prev = np.zeros((nb, nb), dtype=bool)
    count = 0
    for i in range(n):
        for a in range(nb):
            for b in range(a + 1, nb):
                dist = float(np.linalg.norm(pos[i, a] - pos[i, b]))
                touching = dist < float(r[a] + r[b]) + 1e-6
                if touching and not in_contact_prev[a, b]:
                    count += 1
                in_contact_prev[a, b] = touching
    return count
