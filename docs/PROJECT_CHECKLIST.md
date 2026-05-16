# Phys4D — Milestone 2 checklist (6 phases)

**Spine of the M2 deck.** Phases 1–3 are must-haves; 4–5 are the contribution; 6 is stretch.

**Last updated:** 2026-05-15

---

## Phase 1 — Data pipeline

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1.1 | PyBullet generator: randomized physics → multi-view video + GT trajectories + param vector | [~] | `generate_sphere_bounce_dataset.py`; writes `physics_params.json` |
| 1.2 | Param ranges: restitution ∈ [0.3, 0.95], mass ∈ [0.1, 2.0] kg, drop_z ∈ [1.0, 2.2] | [x] | `configs/param_id_dataset.json` |
| 1.3 | K=10 views per scene | [x] | `sphere_bounce_m2.json` `num_cameras: 10` |
| 1.4 | 100–200 scenarios batch script | [~] | `generate_param_id_batch.py` (target 150); run on cluster/overnight |
| 1.5 | 80/10/10 instance split manifest | [x] | `build_param_id_splits.py` → `dataset_manifest.json` |
| 1.6 | Clean tuple format (video, param vector, trajectory) | [x] | per-scene folder + manifest rows |

**Commands:**
```bash
python scripts/generate_param_id_batch.py --dry-run
python scripts/generate_param_id_batch.py --limit 5   # smoke
python scripts/build_param_id_splits.py
```

---

## Phase 2 — CNN architecture

| # | Item | Status | Notes |
|---|------|--------|-------|
| 2.1 | Visual encoder (ResNet18) | [x] | `param_ident/encoder.py` |
| 2.2 | Temporal Transformer (3 layers, d=128) | [x] | `param_ident/model.py` |
| 2.3 | Multi-view aggregation (attention pool) | [x] | `MultiheadAttention` on last timestep views |
| 2.4 | MLP head → param vector (v1: e, m, drop_z) | [x] | No friction in v1 |
| 2.5 | Supervised MSE on GT params | [x] | `train_param_predictor.py` |
| 2.6 | Frozen DINOv2 (stretch) | [—] | Optional |

---

## Phase 3 — Training

| # | Item | Status | Notes |
|---|------|--------|-------|
| 3.1 | Adam lr=1e-4, batch 8, ~50 epochs | [x] | `train_param_predictor.py` defaults |
| 3.2 | Augmentations: random view subsets, brightness jitter | [x] | `ParamIdDataset(augment=True)` |
| 3.3 | Val metric: per-param MSE | [x] | logged each epoch |
| 3.4 | Best checkpoint | [x] | `outputs/param_predictor/param_predictor_best.pt` |
| 3.5 | Train on full 150-scene set | [ ] | After batch gen completes |

---

## Phase 4 — Inference pipeline (end-to-end story)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 4.1 | Object→Gaussian mapping | [~] | mask filter @ ref frame in pipeline |
| 4.2 | video → CNN → params | [x] | `param_ident/inference.py`, `run_param_id_pipeline.py` |
| 4.3 | params → PyBullet rollout → trajectories | [x] | `sim_rollout.py` → `object_poses_predicted.csv` |
| 4.4 | SE(3) warp Gaussian clusters → render proxy | [x] | warped PLYs + mask coverage vs GT traj |
| 4.5 | One full demo E2E | [~] | local or `modal run modal_app.py --pipeline` |
| 4.6 | Full RGB render (PSNR) | [ ] | graphdeco render on Modal (stretch) |

**Commands:**
```bash
python scripts/run_param_id_pipeline.py
modal run modal_app.py --upload-pipeline
modal run modal_app.py --pipeline
```

**4DGS perception (substrate):** `export_4dgs_dataset.py`, Modal `--train-4d`, `--render-4d`

---

## Phase 5 — Evaluation & ablations

| # | Item | Status | Notes |
|---|------|--------|-------|
| 5.1 | Param prediction MSE | [~] | train script val; test split eval TBD |
| 5.2 | Trajectory MSE + velocity R² (held-out 2nd half) | [ ] | |
| 5.3 | Rendered PSNR/SSIM on held-out frames | [ ] | `eval_4dgs_metrics.py` + warp |
| 5.4 | Ablation: CNN vs MLP(first-N-states) | [ ] | |
| 5.5 | Ablation: cross-scenario (bounce → collision) | [ ] | stretch data |
| 5.6 | Ablation: view count K=4/10/20 | [ ] | |

---

## Phase 6 — Real-world (stretch)

| # | Item | Status |
|---|------|--------|
| 6.1 | 2–3 tabletop phone captures | [ ] |
| 6.2 | Qualitative pipeline run | [ ] |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Object→Gaussian mapping | Masks at train time; cluster t=0; solve before M2 demo |
| Friction unidentifiable | **Dropped from v1** predict set |
| Cross-scene generalization fails | Per-scene fine-tune fallback |

---

## Changelog

| Date | Update |
|------|--------|
| 2026-05-15 | Pivot from visual-dynamics extrapolation to 6-phase param-ID + render pipeline |
