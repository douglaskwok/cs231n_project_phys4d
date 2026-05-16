# Project direction (reframe, May 2026)

## One paragraph (team circulation)

We take multi-view video of a dynamic scene, reconstruct it with **4D Gaussian Splatting** (perception + differentiable renderer), extract per-object states (position, orientation, velocity) and appearance features from the training views, then train a **visually conditioned dynamics model** that autoregressively predicts future object states from past states and visual context. Predicted rigid transforms are applied to the reconstructed Gaussians and re-rendered; evaluation is **image-based on held-out future frames** (second half of each clip), not just trajectory MSE. The contribution is **learned visual dynamics for Gaussian-rendered scenes**: extrapolating motion beyond the observed video, which vanilla 4DGS does not target and PhysGaussian does not invert from video. Physics enters as optional regularizers (energy, momentum, non-penetration) with ablations, not as the headline. **Cross-scene training** on many PyBullet bounces with varied mass and restitution is the default regime; 4DGS + Modal stay the perception stack, using upstream **git** implementations ([graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)) on our exported data.

## What is actually novel

| Prior work | What it does | What we add |
|------------|--------------|-------------|
| **4DGS** | Reconstructs and renders the motion that was observed | Predicts **unobserved future** frames after a temporal train/test split |
| **PhysGaussian** | Forward physics on a static Gaussian scene | **Inverse + learned** rollout from video, with visual conditioning |
| **Pure system ID** (fit \(e\), \(g\) on COM tracks) | Interpretable but not a CV story | Kept as baseline / ablation, not the main claim |

The publishable sentence: **from video → Gaussian reconstruction → visually grounded prediction of what happens next.**

## Chosen path (primary)

**Two-stage pipeline**

1. **Perception** — existing repo work: PyBullet → `export_4dgs_dataset.py` → Modal `train-4d` (fudan 4DGS). Optional static 3DGS at frame 0 for warp baselines.
2. **State + appearance** — `object_poses.csv` (sim) or clustered Gaussian COMs (real); CNN (ResNet18 / frozen DINO) on masked crops → per-object visual features.
3. **Prediction** — small dynamics head (Transformer or GNN): inputs = \(K\) past SE(3) states + visual features; output = \(\Delta\) state at \(t{+}1\); autoregressive rollout over the held-out window.
4. **Render** — warp Gaussians with predicted poses (`warp_gaussians_to_frame.py` pattern); compare to GT RGB (SSIM/L1).

**Training:** cross-scene on 50–200 PyBullet clips (`scripts/generate_sphere_bounce_batch.py`). Hold out scenes by restitution / mass buckets for generalization claims.

**Losses**

- \(L_\text{render}\): SSIM/L1 on held-out views (main CV metric)
- \(L_\text{state}\): MSE on positions / orientations (when poses available)
- \(L_\text{physics}\): optional energy / momentum / non-penetration residuals (ablated)

## Alternatives (ablations / baselines, not headline)

1. **System ID via splats** — 4DGS → cluster Gaussians → fit \(g, e, \mu\) on COM tracks → PyBullet forward on test half. Fast, interpretable; cite as baseline.
2. **Physics-informed losses only** — add concrete terms during 4DGS or warp optimization; ablate each. Supports the main story but weak alone.

## Answers to the two gating questions

### (1) Most stable 4DGS scenario today?

**Single sphere bounce** on a known ground plane.

- Simulator gives masks, poses, and multi-view RGB with no segmentation risk.
- Static 3DGS at frame 0 already trains reliably (~46 PSNR @ 7k iters).
- 4DGS **train** reconstruction is strong (~40 PSNR on cams 0–3, frames 0–59); **current DyNeRF split** holds out cams 4–5 on frames 60–89 (novel camera + late time), where test PSNR ~10 — that measures **view generalization**, not temporal extrapolation. For the new story, add a **temporal** eval: train 4DGS on frames 0–59 (all cams), predict 60–89 via the dynamics head (4DGS alone will not extrapolate meaningfully).

Pose-only physics extrapolation on the test half already beats constant-velocity and linear baselines (`extrapolation_report.json`: test \(z\) MSE 0.0039 vs 0.073 / 0.012).

### (2) Can we script ~100 PyBullet videos overnight?

**Yes.** One scene is ~90 frames × 6 cams × 256² RGB — seconds on CPU. `scripts/generate_sphere_bounce_batch.py` sweeps restitution / mass / drop height; 100 scenes is on the order of tens of minutes locally. Cross-scene training is feasible before investing in a heavy dynamics architecture.

## Stack policy (now)

Use **upstream git** splatting only — no fork maintenance in this repo:

- **3DGS:** cloned in Modal image to `/opt/gs` ([graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)); input via `export_gs_blender_scene.py`.
- **4DGS:** cloned to `/opt/4dgs` ([fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)); input via `export_4dgs_dataset.py` + `configs/sphere_bounce_4dgs.yaml`.

Our code owns: data generation, export formats, pose warp, metrics, and (new) dynamics training — not rasterizer internals.

## Near-term milestones

| Week | Deliverable |
|------|-------------|
| 1 | Batch dataset manifest + 10-scene smoke; temporal train/test eval script |
| 2 | State sequences + ResNet18 feature cache per scene |
| 3 | Dynamics head + autoregressive rollout; \(L_\text{state}\) on sim poses |
| 4 | Gaussian warp render + \(L_\text{render}\) on held-out frames; baselines table |

## Eval splits (explicit)

| Split | Purpose |
|-------|---------|
| **Temporal** | Frames 0–59 train, 60–89 test — **primary** for extrapolation claim |
| **Camera** (current 4DGS yaml) | Cams 0–3 train, 4–5 test — reconstruction / NVS, not extrapolation |
| **Cross-scene** | Train dynamics on scenes A–N, test on held-out physics params |

See also: `docs/WORKFLOW.md`, `docs/project_claim.md` (legacy M2 inverse-ID framing).
