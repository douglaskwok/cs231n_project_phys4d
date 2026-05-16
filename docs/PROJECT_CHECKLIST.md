# Checklist — project.md (visual dynamics)

**Spec:** [`../project.md`](../project.md)

## Phase 1 — Data

| Item | Status |
|------|--------|
| PyBullet video + masks + poses | [x] `generate_sphere_bounce_dataset.py` |
| Params saved for ablations only | [x] `physics_params.json` / config |
| 45 scenes batch + manifest | [x] `sphere_bounce_batch` |
| 150 scenes, K=10 | [~] script ready; batch is 6 cams |
| T_future=30, train 0–59 / test 60–89 | [x] `configs/visual_dynamics.json` |

## Phase 2 — Perception

| Item | Status |
|------|--------|
| 4DGS reconstruction | [x] Modal `--train-4d` |
| Object→Gaussian clustering | [~] `select_sphere_gaussians` |
| States + finite-diff velocity | [x] `perception/states.py` |
| t=0 visual features (fixed) | [x] `extract_perception.py` |

## Phase 3 — Dynamics NN

| Item | Status |
|------|--------|
| Object token = history + visual | [x] `ObjectTokenDynamics` |
| Transformer over objects | [x] (N=1 sphere) |
| SE(3) delta in state space | [x] 10D delta |
| Autoregressive rollout | [x] `rollout.py` |

## Phase 4 — Training

| Item | Status |
|------|--------|
| Stage A teacher forcing | [x] `train_visual_dynamics.py` |
| Stage B rollout curriculum | [x] horizons in config |
| L_state (pos + quat geodesic + vel) | [x] `losses.py` |
| L_render | [~] mask coverage proxy in pipeline |
| L_physics optional | [x] flag `w_physics` |
| AdamW 3e-4 + grad clip | [x] |

## Phase 5 — Inference

| Item | Status |
|------|--------|
| E2E pipeline script | [x] `run_visual_dynamics_pipeline.py` |
| Demo run with ckpt + PLY | [ ] needs train + 3DGS |

## Phase 6 — Eval

| Item | Status |
|------|--------|
| Horizon error curve | [x] `eval_visual_dynamics.py` |
| Velocity R² | [x] `trajectory_eval.py` |
| PSNR/SSIM render | [ ] full rasterizer TBD |
| Ablation: no visual | [x] `--no-visual` |
| Ablation: horizon sweep | [x] |
| Cross-scenario collision | [ ] |
