# Phys4D project checklist (living)

**Reframe (May 2026):** video → 4DGS perception → **learned visual dynamics** → predict motion **beyond observed frames** (temporal split 0–59 train / 60–89 test). See [`project_direction.md`](project_direction.md).

**Agents:** Read this file at the **start** of any Phys4D task. After meaningful work, **update statuses** and add a dated note under [Changelog](#changelog). Rule: `.cursor/rules/phys4d-project-checklist.mdc`.

**Legend:** `[x]` done · `[~]` partial · `[ ]` not started · `[—]` explicitly deferred / not chosen

---

## A. Planning, git, and framing

| # | Item | Status | Notes / artifacts |
|---|------|--------|-------------------|
| A1 | Problem overview vs 4DGS / PhysGaussian / system-ID | [x] | `docs/project_direction.md`, chat history |
| A2 | Novelty = **temporal extrapolation**, not vague “physics in loss” | [x] | `docs/project_direction.md` |
| A3 | Team circulation paragraph | [x] | Top of `project_direction.md` |
| A4 | Git branch + merge with `origin/main` | [x] | `feat/visual-dynamics-reframe` → `main` |
| A5 | Upstream **git** 3DGS (graphdeco) on our RGB | [x] | Modal `/opt/gs`, `export_gs_blender_scene.py` |
| A6 | Upstream **git** 4DGS (fudan) on our DyNeRF export | [x] | Modal `/opt/4dgs`, `export_4dgs_dataset.py` |

---

## B. Data (PyBullet, cross-scene)

| # | Item | Status | Notes / artifacts |
|---|------|--------|-------------------|
| B1 | Single-scene sphere bounce (m2) | [x] | `outputs/sphere_bounce_m2/` |
| B2 | Batch generator (varied e, m, drop_z) | [x] | `scripts/generate_sphere_bounce_batch.py` |
| B3 | **45-scene** batch (default grid) | [x] | `outputs/sphere_bounce_batch/` |
| B4 | Scale to **50–200** scenes | [ ] | Increase grid in `configs/cross_scene_batch.json` |
| B5 | Gating Q: ball bounce = stable first scenario | [x] | Documented in `project_direction.md` |
| B6 | Gating Q: overnight batch feasible | [x] | ~minutes per scene on CPU |

---

## C. Perception (3DGS / 4DGS)

| # | Item | Status | Notes / artifacts |
|---|------|--------|-------------------|
| C1 | Static 3DGS train (frame 0, 6 cams) | [x] | Modal; ~46 PSNR; `gs_sphere_bounce/` |
| C2 | Single-scene 4DGS train (full bounce) | [x] | Modal; train ~40 / test ~10 PSNR (camera split) |
| C3 | DyNeRF export for **all 45** batch scenes | [x] | `export_4dgs_batch.py` → `outputs/sphere_bounce_batch_dynerf/` |
| C4 | Modal **batch 4DGS smoke** (5 scenes, 5k iters) | [~] | Volume has **no** `4dgs_batch/` yet — re-run `--batch-4d-smoke` if needed |
| C5 | Batch 4DGS at full 15k / more scenes | [ ] | After smoke validates pipeline |
| C6 | Clarify eval: **temporal** vs **camera** holdout in writeup | [~] | Documented; don’t confuse test PSNR ~10 with extrapolation |

---

## D. Prediction (learned visual dynamics)

| # | Item | Status | Notes / artifacts |
|---|------|--------|-------------------|
| D1 | Per-object state (pose, vel) from sim | [x] | `object_poses.csv`, `object_state.py` |
| D2 | Visual encoder (masked crops) | [x] | ResNet18 in `visual_dynamics/feature_encoder.py` |
| D3 | Dynamics head (autoregressive Δstate) | [~] | MLP in `dynamics_head.py` (not Transformer/GNN) |
| D4 | Cross-scene train script | [x] | `scripts/train_visual_dynamics.py` |
| D5 | Checkpoint | [x] | `outputs/visual_dynamics/visual_dynamics.pt` |
| D6 | Train on **full 45** scenes + strict physics holdout | [~] | Trained; val split needs e=0.75,m=1.0 scenes |
| D7 | Frozen DINO / stronger visual backbone | [—] | Optional stretch |
| D8 | Per-scene-only training regime | [—] | **Deferred** — cross-scene chosen |

---

## E. Evaluation and baselines

| # | Item | Status | Notes / artifacts |
|---|------|--------|-------------------|
| E1 | Temporal split eval (frames 60–89) | [x] | `eval_temporal_extrapolation_split.py` |
| E2 | Baselines: constant, linear, physics bounce | [x] | `run_extrapolation_baselines.py` → `outputs/extrapolation_baselines.md` |
| E3 | Learned model in baseline table | [x] | z MSE competitive on batch |
| E4 | Warp 3DGS with predicted poses | [x] | `warp_gaussians_to_frame.py`, `eval_extrapolation_render.py` |
| E5 | **L_state** (pose MSE) | [x] | In train + eval |
| E6 | **L_render proxy** (mask coverage, crop MAE) | [~] | `eval_extrapolation_render.py` — not full rasterizer |
| E7 | **True L_render** (PSNR/SSIM on held-out RGB) | [ ] | **Headline gap** — needs Modal GS rasterizer pass |
| E8 | System-ID via splats (cluster → fit e,g → sim) | [ ] | Alternative path; physics-on-poses only so far |
| E9 | Physics-informed losses + ablations | [ ] | Optional regularizers not wired |
| E10 | Counterfactual restitution render edits | [ ] | Stretch in `sphere_bounce_m2.json` |

---

## F. Modal commands (operator)

| Step | Command |
|------|---------|
| Smoke | `modal run modal_app.py` |
| Static 3DGS | `modal run modal_app.py --upload` then `--train` |
| Single 4DGS | `modal run modal_app.py --upload-4d` then `--train-4d` |
| **Batch 4DGS smoke** | `modal run modal_app.py --batch-4d-smoke` |
| Reconstruction metrics | `modal run modal_app.py --eval-4d` |
| Download batch | `modal volume get phys4d-gs-output 4dgs_batch . --force` |

---

## G. Publishable story readiness

| Milestone | Status |
|-----------|--------|
| “We reconstruct dynamic scenes with 4DGS” | [x] single scene (train+render+eval on volume); batch smoke not on volume |
| “We predict **future** motion from past video” | [~] poses yes; **images** weak |
| “Visually conditioned, cross-scene generalization” | [~] scaffold yes; full sweep no |
| “Image-based eval on held-out frames” | [ ] **blocked on E7** |

**Rough overall progress (chosen two-stage path): ~55–65%.**

---

## H. Next actions (priority order)

1. [ ] Download single-scene 4DGS artifacts (if not local): `4dgs_sphere_bounce`, `4dgs_renders`, `metrics_4dgs.json`.
2. [ ] **Or** run `modal run modal_app.py --batch-4d-smoke` if batch perception was the goal (not on volume yet).
3. [ ] Implement **true L_render** (Modal rasterize warped 3DGS → PSNR frames 60–89).
4. [ ] Re-train / eval dynamics on all 45 scenes with holdout `e=0.75, m=1.0`.
5. [ ] Optional: system-ID baseline row; physics loss ablations.

---

## Changelog

| Date | Agent / human | Update |
|------|----------------|--------|
| 2026-05-16 | Agent | Initial checklist from first-message proposal vs work through commits `96e5add`, `83bb0c4`. |
| 2026-05-16 | Agent | User Modal run finished; volume has `4dgs_sphere_bounce`, `4dgs_renders`, `4dgs_eval` — **not** `4dgs_batch`. C4 still open for batch smoke. |
