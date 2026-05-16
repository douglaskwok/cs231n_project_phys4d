# Phys4D — Milestone 2 checklist (6 phases)

**Spine of the M2 deck.** Phases 1–3 are must-haves; 4–5 are the contribution; 6 is stretch.

**Full walkthrough:** [`PIPELINE_FROM_SCRATCH.md`](PIPELINE_FROM_SCRATCH.md)  
**Overnight automation:** `scripts/overnight_run.sh` → `outputs/overnight/run.log`

**Last updated:** 2026-05-16

---

## Phase 1 — Data pipeline

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1.1 | PyBullet generator + param vector | [~] | 45 scenes in `sphere_bounce_batch`; 150 via `generate_param_id_batch.py` |
| 1.2 | Param ranges in config | [x] | `param_id_dataset.json` |
| 1.3 | K=10 views | [~] | Config=10; existing batch=6 cams |
| 1.4 | 100–200 scenarios | [~] | Script ready; full run optional |
| 1.5 | 80/10/10 manifest | [x] | `sphere_bounce_batch/dataset_manifest.json` |
| 1.6 | Tuple format | [x] | manifest + per-scene folders |

---

## Phase 2 — CNN architecture

| # | Item | Status |
|---|------|--------|
| 2.1–2.5 | ResNet + Transformer + MLP + MSE | [x] |
| 2.6 | DINOv2 | [—] |

**Also:** pose-only MLP baseline (`param_ident/mlp_baseline.py`)

---

## Phase 3 — Training

| # | Item | Status | Notes |
|---|------|--------|-------|
| 3.1–3.4 | Train loop + aug + val + ckpt | [x] | `--model cnn\|mlp` |
| 3.5 | Full 150-scene train | [~] | Run `overnight_run.sh` or Modal `--train-param-id` |

---

## Phase 4 — Inference pipeline

| # | Item | Status | Notes |
|---|------|--------|-------|
| 4.1 | Object→Gaussian | [~] | `select_sphere_gaussians()` mask+opacity+centroid |
| 4.2 | video → CNN → params | [x] | |
| 4.3 | rollout → trajectories | [x] | |
| 4.4 | warp + render proxy | [x] | mask coverage + `projection_error.json` |
| 4.5 | E2E demo | [~] | needs trained ckpt + `gs_sphere_bounce` PLY |
| 4.6 | Full RGB PSNR | [ ] | Modal `--eval-4d` / graphdeco render on warped PLY |

---

## Phase 5 — Evaluation & ablations

| # | Item | Status | Notes |
|---|------|--------|-------|
| 5.1 | Param MSE (test) | [x] | `eval_param_predictor.py` |
| 5.2 | Traj MSE + velocity R² | [x] | in eval rollout metrics |
| 5.3 | Render PSNR/SSIM | [~] | 2D projection px error proxy; full PSNR TBD |
| 5.4 | CNN vs pose-MLP | [x] | `run_ablations.py` |
| 5.5 | Cross-scenario | [ ] | no collision data |
| 5.6 | View sweep K=4/6/10 | [x] | `--max-views` + `run_ablations.py --view-sweep` |

---

## Phase 6 — Real-world

| # | Item | Status |
|---|------|--------|
| 6.1–6.2 | Phone captures + qual | [ ] |

---

## Overnight (2026-05-16)

```bash
conda activate phys4d
OVERNIGHT_EPOCHS=40 bash scripts/overnight_run.sh
# log: outputs/overnight/run.log
```

---

## Changelog

| Date | Update |
|------|--------|
| 2026-05-16 | Eval, ablations, overnight script, improved Gaussian selection |
| 2026-05-15 | Pivot to param-ID pipeline |
