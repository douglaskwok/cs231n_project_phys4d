# Milestone 2 Slide Outline

## Slide 1: Problem and Claim

- Dynamic Gaussian methods can render observed motion but usually do not recover editable physical parameters.
- Claim: object-centric Gaussian scenes plus differentiable physics can identify parameters from video, then extrapolate and support counterfactual edits.
- First test case: recover restitution for a sphere bouncing on a plane.

## Slide 2: Related Work

- 4DGS: strong dynamic scene rendering, limited explicit physical interpretability.
- PhysGaussian: connects Gaussian scene representations to forward physical simulation.
- GASP: physics-aware Gaussian scene modeling and simulation.
- Our focus: inverse system identification from observed multi-view video, starting from a controlled PyBullet scene.

## Slide 3: Method Overview

```text
multi-view RGB video
  -> known masks and object poses for M2
  -> object-centric canonical Gaussians
  -> differentiable rigid-body simulator
  -> Gaussian pose transforms over time
  -> rendered frames and physics losses
  -> recovered parameters, extrapolation, counterfactual rollout
```

## Slide 4: Model and Losses

- Object `o` owns canonical Gaussians `G_o = {mu, Sigma, alpha, c}`.
- Simulator predicts rigid state `x_t = (p_t, v_t, R_t, omega_t)`.
- Transform Gaussians by `mu_t = R_t mu + p_t` and `Sigma_t = R_t Sigma R_t^T`.
- Optimize rendering loss, pose supervision, non-penetration, and contact consistency.
- M2 demo isolates the physics term by optimizing restitution from observed `z_t`.

## Slide 5: Data and Evaluation

- Data: PyBullet sphere-ground bounce, 6 camera ring, known masks, known camera poses, known object poses.
- Train split: first 60 frames and 4 cameras.
- Test split: final 30 frames and 2 held-out cameras.
- Metrics: restitution absolute error, trajectory MSE, future-frame PSNR, masked MSE by prediction horizon.
- Baselines: constant velocity, unconstrained pose fit, planned vanilla 4DGS extrapolation.

## Slide 6: Implementation Progress

- Existing repo contains PyBullet notebooks and trajectory outputs.
- Added focused experiment spec for sphere-bounce M2.
- Added minimal differentiable PyTorch bounce simulator.
- Added restitution recovery script and unittest coverage for gradient flow and recovery accuracy.
- Next integration target: connect PyBullet generated poses and masks to the Gaussian rendering path.

## Slide 7: Risks and Next Steps

- Risk: object discovery can dominate the project. Mitigation: use PyBullet masks for M2.
- Risk: full Gaussian rendering integration may be slower than expected. Mitigation: validate physics on pose trajectories first.
- Risk: baselines can expand scope. Mitigation: use simple extrapolation baselines first and cite PhysGaussian/GASP as related work.
- Next: generate locked sphere-bounce dataset, recover restitution from PyBullet poses, render held-out futures, then add counterfactual restitution edits.
