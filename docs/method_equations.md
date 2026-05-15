# Method Equations

These equations are written for Milestone 2 slides. The first implementation target is a single bouncing sphere, but the notation leaves room for multiple rigid objects.

## Object-Centric Gaussians

Each object `o` owns a set of canonical 3D Gaussians:

```text
G_o = {g_{o,k}}_{k=1}^{K_o}
g_{o,k} = (mu_{o,k}, Sigma_{o,k}, alpha_{o,k}, c_{o,k})
```

where `mu_{o,k}` is the canonical mean, `Sigma_{o,k}` is the covariance, `alpha_{o,k}` is opacity, and `c_{o,k}` denotes color or spherical harmonic coefficients.

## Per-Object Rigid-Body State

At frame `t`, object `o` has state:

```text
x_{o,t} = (p_{o,t}, v_{o,t}, R_{o,t}, omega_{o,t})
```

with position `p`, linear velocity `v`, rotation `R in SO(3)`, and angular velocity `omega`. Physical parameters are collected in:

```text
theta_o = (m_o, r_o, e_o, mu_f,o, ...)
```

For the first experiment, `theta = e`, the sphere-ground restitution.

## Gaussian Pose Transform

The simulator predicts object pose. Canonical Gaussians are transformed into world coordinates by:

```text
mu_{o,k,t} = R_{o,t} mu_{o,k} + p_{o,t}
Sigma_{o,k,t} = R_{o,t} Sigma_{o,k} R_{o,t}^T
alpha_{o,k,t} = alpha_{o,k}
c_{o,k,t} = c_{o,k}
```

For a sphere with isotropic appearance, rotation can be ignored in the first milestone while preserving the same transform interface.

## Differentiable Sphere-Ground Dynamics

Let `z_t` and `u_t` be vertical position and velocity, with sphere radius `r`, timestep `dt`, gravity `g < 0`, ground height `z_g`, and restitution `e in [0, 1]`.

Free fall:

```text
u^-_{t+1} = u_t + g dt
z^-_{t+1} = z_t + u^-_{t+1} dt
```

Contact event:

```text
h^-_{t+1} = z^-_{t+1} - (z_g + r)
I_{t+1} = 1[h^-_{t+1} < 0 and u^-_{t+1} < 0]
```

Velocity and position update:

```text
u_{t+1} = (1 - I_{t+1}) u^-_{t+1} + I_{t+1} (-e u^-_{t+1})
z_{t+1} = (1 - I_{t+1}) z^-_{t+1} + I_{t+1} (z_g + r)
```

This update is piecewise differentiable with respect to `e`. The Milestone 2 simulator uses this direct PyTorch formulation.

## Rendering Loss

Given camera `c`, frame `t`, observed image `I_{c,t}`, and rendered image `R_c({G_{o,t}})`, the photometric objective is:

```text
L_render = sum_{t in T_train} sum_{c in C_train}
           || M_{c,t} * (R_c({G_{o,t}}) - I_{c,t}) ||_2^2
```

where `M_{c,t}` is a known PyBullet object mask for the first experiment.

## Physics Losses

The full project can combine rendered evidence with physics consistency:

Non-penetration:

```text
L_np = sum_t max(0, z_g + r - z_t)^2
```

Trajectory consistency with supervised PyBullet poses:

```text
L_pose = sum_t || p_t - p_t^* ||_2^2 + lambda_R d_SO3(R_t, R_t^*)^2
```

Collision energy dissipation bound for sphere-ground impact:

```text
L_energy = sum_{t in contacts} max(0, E_{t+1} - E_t)^2
```

Momentum or impulse consistency for future multi-object collisions:

```text
L_impulse = sum_{t in contacts} || v_{t+1} - v_t - J_t / m ||_2^2
```

## Staged Optimization Objective

Stage 1, initialize appearance and per-frame poses:

```text
min_{G, {x_t}} L_render + lambda_pose L_pose
```

Stage 2, identify physical parameters from observed states:

```text
min_{theta, x_0} sum_{t in T_train} || Phi_t(x_0, theta) - x_t^* ||_2^2
```

where `Phi_t` is the differentiable simulator rollout.

Stage 3, joint refinement:

```text
min_{G, x_0, theta}
    L_render
  + lambda_pose L_pose
  + lambda_np L_np
  + lambda_energy L_energy
```

Evaluation uses held-out frames and cameras:

```text
MSE_z(h) = || z_{T_train+h}^{pred} - z_{T_train+h}^{gt} ||_2^2
param_error = | e_hat - e_gt |
```
