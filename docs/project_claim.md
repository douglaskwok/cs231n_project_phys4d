# Project Claim

## One-Sentence Claim

We model dynamic object-centric Gaussian scenes as an inverse system identification problem: from observed multi-view video, recover appearance, pose trajectories, and physical parameters that support extrapolation and counterfactual editing.

## What Is New

The central claim is not that Gaussians can be moved by physics. Prior work already shows that Gaussian scene representations can be simulated or regularized with physical structure. Our contribution is narrower and more testable:

1. Recover object-level physical parameters from video, starting with restitution in a sphere-ground bounce.
2. Use the recovered parameters to extrapolate beyond the observed training horizon.
3. Edit physical parameters and render counterfactual futures, such as a more elastic or less elastic bounce.

## Positioning Against Related Work

### 4DGS

4D Gaussian Splatting methods represent time-varying radiance fields and can render dynamic scenes well inside the observed motion distribution. They do not, by default, recover editable physical parameters. Our project uses dynamic Gaussian rendering as the visual substrate, but evaluates whether a physics-constrained object model improves future rollout and exposes interpretable controls.

### PhysGaussian

PhysGaussian connects Gaussian representations with physics-based simulation. Its emphasis is forward simulation and physically plausible Gaussian motion. Our project instead emphasizes inverse system identification from observed video: estimating parameters such as restitution from image evidence and then using those estimates for prediction and edits.

### GASP

GASP also explores physics-aware Gaussian scene representations. We treat it as related work for simulating or constraining Gaussian dynamics. Our narrower milestone target is a controlled inverse problem with known masks and poses, where the first success criterion is accurate recovery of a physical parameter from a PyBullet bounce.

## Milestone 2 Scope

Milestone 2 should demonstrate that the project has a concrete path from video observations to recoverable physics. The first experiment is deliberately small:

- One sphere bouncing on one ground plane.
- Known object masks, camera poses, and object pose supervision from PyBullet.
- Restitution as the first recovered parameter.
- A lightweight PyTorch differentiable simulator before integrating full Gaussian rendering.

This scope makes the claim falsifiable: if recovered restitution predicts held-out future heights better than unconstrained fitting, the project has evidence for physics-aware extrapolation.
