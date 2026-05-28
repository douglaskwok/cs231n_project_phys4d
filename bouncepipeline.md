# Ball bounce — physics extrapolation pipeline (spec)

**Repo root:** `cs231n_project_phys4d.nosync/` (run commands from here).

**Goal:** Given multi-view PyBullet ball-drop video, use **object-only 4DGS** (trained on the observed window) to estimate a 3D trajectory, fit a simple bounce simulator, **predict** positions on held-out frames, render object + background, and score against PyBullet GT.

**Scope:** `phys4d_final` **ball_drop** scenes only (single sphere, no inter-object collision). Rigid translation at render time (no spin).

**4DGS backend:** [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) (ICLR 2024) — same as Modal `--train-4d` / `--render-4d`. Not Wu et al. 4DGaussians.

**Status:** Phases 1–5 implemented (`run_phase1.py` … `run_phase5.py`). See [Current progress](#current-progress), [Reuse vs build](#reuse-vs-build), and **[scripts/bounce/INSTRUCTIONS.md](scripts/bounce/INSTRUCTIONS.md)** (“How to view the latest results”).

Related docs: [`dataset/README.md`](dataset/README.md) (datagen), [`4dgs/README.md`](4dgs/README.md) (train/render), [`docs/PROGRESS.md`](docs/PROGRESS.md) (team log).

---

## Current progress

| Area | Status | Notes |
|------|--------|--------|
| PyBullet ball-drop data | **Done** | `dataset/generate_phys4d_final.py` → `dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_*` |
| GT poses | **Done** | Per scene: `object_poses.csv` (`x_m`, `y_m`, `z_m`, …) |
| Object-only DyNeRF export | **Done** | `4dgs/experiments/object_only/export_object_only_dynerf.py` → `4dgs/experiments/object_only/runs/<RUN_NAME>/` |
| Object-only 4DGS train | **In progress** | Many runs under `runs/`; Modal via `4dgs/scripts/train_one_final_scene_4dgs.sh` |
| Train/test split in exports | **Mixed** | Scene `config.json` has `train_frames` / `test_frames`; many runs used `--all-train` (full clip) — need a **split-respecting** train for held-out eval |
| Background 3DGS | **Not started** | Phase 4 blocker |
| Phases 1–5 (this doc) | **Phases 1–3 done** | `src/phys4d/bounce/` + `scripts/bounce/run_phase{1,2,3}.py`; runbook: `scripts/bounce/INSTRUCTIONS.md` |
| Split eval (e0p78) | **Smoke-tested** | `ball_drop_e0p78_a0p0_object`: 60 train Phase 1, 30 test Phase 3 predictions |

**Reference scene (first timing run):**

```text
SCENE_DIR=dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p90_a0p0
RUN_NAME=ball_drop_e0p90_a0p0_object
EXPORT_DIR=4dgs/experiments/object_only/runs/ball_drop_e0p90_a0p0_object_vis6_px50_all12
```

**Assumed starting state for Phase 1+:** A finished object-only 4DGS checkpoint for one ball-drop scene (Modal download or local `chkpnt*.pth`), DyNeRF export with `transforms_train.json`, `frame_map.json`, and matching `object_poses.csv` on disk.

---

## How this fits the repo

The main CS231N story in [`docs/WORKFLOW_COMMANDS.md`](docs/WORKFLOW_COMMANDS.md) is **visual dynamics** (nested copy under `cs231n_project_phys4d/`). **This pipeline is a parallel track:** classical trajectory fit + rigid Gaussian warp, not the Transformer rollout.

```
dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_*/
  config.json, rgb/, masks/, object_poses.csv, cameras.json
       │
       ▼
4dgs/experiments/object_only/runs/<RUN_NAME>/     # DyNeRF export (local)
       │
       ▼  Modal train
dataset/.../scene_*/4dgs/<RUN_NAME>/              # chkpnt30000.pth, exported PLY
       │
       ▼  [Phases 1–5 — to implement]
outputs/bounce_pipeline/<RUN_NAME>/phase{1..5}/
```

**Compute:** Heavy steps (4DGS train, rasterize, **Phase 1**) → **Modal** (`modal_app.py --bounce-phase1`). Phases 2–3 are CPU-friendly locally once Phase 1 CSVs exist.

**Source code today:**

- Root `src/phys4d/__init__.py` is a thin stub; full utilities live in `cs231n_project_phys4d/src/phys4d/` (`gaussian_ply`, `poses`, `differentiable_bounce`, `trajectory_metrics`, …). **Before implementing phases, either copy/sync those modules to root `src/phys4d/` or add `sys.path` to the nested tree** (prefer syncing to root for one import path).

---

## Train / test split (not generic 80%)

The spec originally said “first 80% of video.” The repo uses **time-based splits** from PyBullet export:

- Exporter: `dataset/export_ping_pong_12view.py` — `TRAIN_DURATION_SEC = 1.0`, `TEST_DURATION_SEC = 0.5` at `video_fps` (default **60**, not 120).
- Per scene `config.json`: `train_frames`, `test_frames` (inclusive frame index ranges on the **sim timeline**).
- Object-only export may **drop** weak timestamps (`--drop-invisible-frames`, `--min-visible-cameras`, …). Use `frame_map.json` (`original_frame` ↔ `kept_index`) when aligning 4DGS time to GT poses.

**Implementation rule:** `T_obs` = number of **kept** train timestamps in the DyNeRF export used for 4DGS training; `T_total` = train + test kept timestamps (or full sim range if evaluating in sim frame space). Do not hard-code 96/120 unless the scene manifest says so.

For a strict **80/20** ablation later, add an export flag; default to **scene `config.json` split** for consistency with existing data.

---

## Pipeline overview

```
4DGS chkpnt + DyNeRF export ──► [Phase 1] ──► trajectory CSV
                                        │
                                        ▼
                               [Phase 2] fit physics
                                        │
                                        ▼
                               [Phase 3] roll forward (held-out)
                                        │
                    PLY (canonical) + BG 3DGS ──► [Phase 4] merge & render
                                        │
                                        ▼
                               [Phase 5] evaluate
```

Each phase: one CLI under `scripts/bounce/`, outputs under `outputs/bounce_pipeline/<RUN_NAME>/phaseN/`. Inputs are paths only so phases re-run independently.

---

# Phase 1 — Trajectory extraction from 4DGS

**Goal:** Per-frame 3D ball centroid `p̂(t)` over the **training** window (and optionally full clip for plots).

### Inputs

| Input | Typical path |
|-------|----------------|
| 4DGS checkpoint | `dataset/.../4dgs/<RUN_NAME>/chkpnt30000.pth` (or `chkpnt1000.pth` for `--quick`) |
| DyNeRF export (time + cameras) | `4dgs/experiments/object_only/runs/<RUN_NAME>/` |
| Train frame mapping | `frame_map.json`, `transforms_train.json` |
| GT (validation) | `<SCENE_DIR>/object_poses.csv` |
| FPS | `export_meta.json` → `fps` (usually **60**) |

Fudan checkpoints store deformation in **`chkpnt*.pth`**, not a static `point_cloud.ply` alone. Load via `4dgs/scripts/fourdgs_common.py` + upstream `GaussianModel`, or export once with `4dgs/scripts/export_4dgs_ply.py`.

### Outputs (`outputs/bounce_pipeline/<RUN_NAME>/phase1/`)

- `trajectory_raw.csv` — `frame, t_sec, x, y, z` (sim frame indices where possible)
- `trajectory_smoothed.csv` — Savitzky–Golay per axis
- `trajectory_plot.png` — raw / smoothed / GT

### Steps (implementation sketch)

1. **Load 4DGS** — canonical Gaussians + deformation field; `t_norm ∈ [0,1]` over the **training** time range used at train time (`time_duration` in YAML / export meta).
2. **Opacity-weighted centroid** per kept train timestamp (apply **sigmoid** to PLY opacity logits).
3. **Smooth** — start `window_length=5`, `polyorder=2`; reduce order if bounces round off.
4. **Validate** — overlay `object_poses.csv`; optional Procrustes if global offset only.

### Gotchas

- Opacity in PLY is **logit**; use sigmoid before weights.
- `t_norm` is normalized over the **4DGS training window**, not the full sim clip.
- 3DGS world frame ≠ PyBullet absolute frame; Phase 2 can fit in **shape + gravity direction**; Phase 4 should use **pose deltas** (see existing `warp_gaussians_to_frame`).

### Acceptance criteria

- [x] CSVs: `T_obs` rows, no NaNs, monotonic `t_sec` — `run_phase1.py`
- [ ] ≥1 visible bounce on z in plot — check `trajectory_plot.png` after a real checkpoint run
- [ ] Smoothed curve keeps bounce cusps — tune `--savgol-window` / `--savgol-polyorder` if needed
- [ ] MSE vs GT on train frames &lt; 0.05 m (after optional alignment) — see `phase1_meta.json` (`train_mse_aligned_m2`)

---

# Phase 2 — Physics parameter fitting

**Goal:** Fit `(p₀, v₀, g, e, z_g)` so a discrete bounce simulator matches `trajectory_smoothed.csv` on the train window.

### Inputs

- `trajectory_smoothed.csv` (Phase 1)
- `fps`, `T_obs`

### Outputs (`phase2/`)

- `physics_params.json`
- `physics_fit_plot.png`

### Steps

1. **Detect bounces** — `v_z` sign flip (− → +); sub-frame zero crossing; ignore apex noise below a `|v_z|` threshold.
2. **Initial guesses** — `p₀` first row; `v₀` from first ~5 frames; `g ≈ -9.81`; `z_g ≈ min(z)`; `e` from mean `-v_after/v_before` at bounces.
3. **Joint least squares** — `scipy.optimize.least_squares` on stacked `(sim - obs)`; optional fix `g = -9.81`.
4. Save JSON + overlay plot.

**Existing code (1D vertical toy):** `cs231n_project_phys4d/src/phys4d/differentiable_bounce.py`, `trajectory_metrics.py`, `scripts/eval_physics_trajectory_split.py` (poses + train-end frame). Phase 2 here is **3D translation** with floor bounce — reuse bounce detection ideas, not the 1D-only simulator as-is.

### Acceptance criteria

- [ ] `e ∈ [0.3, 0.95]`
- [ ] `g` within ±15% of −9.81 if free
- [ ] `fit_mse` &lt; 0.01 m² on train
- [ ] Bounce count matches plot

---

# Phase 3 — Extrapolation on held-out frames

**Goal:** Predict ball positions for **test** frame indices (per `config.json` / `frame_map.json`).

### Inputs

- `physics_params.json`
- `T_obs`, `T_total` (kept or sim indices — document which in CSV)
- `fps`

### Outputs (`phase3/`)

- `trajectory_predicted.csv` — test frames only
- `trajectory_full_plot.png` — train + pred + GT

### Steps

1. Advance simulator from **state at end of train window** (do not reset to `p₀`, `v₀`).
2. Write rows for test frames only.
3. Sanity: continuity at boundary; `z ≥ z_g − ε`.

### Acceptance criteria

- [ ] Row count = number of held-out frames
- [ ] Boundary gap &lt; 1 cm
- [ ] No floor penetration

---

# Phase 4 — Scene assembly & rendering

**Goal:** Render held-out views: translated object Gaussians + static background.

### Inputs

| Input | Path |
|-------|------|
| Canonical object Gaussians | Exported PLY or `chkpnt*.pth` at `t_norm = 0` |
| Background 3DGS | **TBD** — `runs/<RUN_NAME>/background.ply` or full-scene 4DGS |
| Predicted trajectory | `phase3/trajectory_predicted.csv` |
| Reference centroid | `phase1/trajectory_smoothed.csv` frame 0 or canonical centroid |
| Cameras | `transforms_train.json` / `transforms_test.json` in DyNeRF export |

### Outputs (`phase4/`)

- `renders/<frame>_cam<idx>.png` (match fudan naming for `eval_4dgs_metrics.py`)
- `composite_video.mp4` (optional)

### Steps

1. **Background (one-time):** train static 3DGS on room views with inverted mask, or full-scene 4DGS export — not in repo yet.
2. **Per frame:** `delta = p_pred[t] - p_ref`; translate canonical xyz (no rotation); **do not** also apply 4DGS deformation (double motion).
3. **Rasterize** — `4dgs/scripts/render_4dgs_trajectory.py` (dataset mode) or warp-then-render via `warp_gaussians_to_frame` + compositor.
4. **Composite** — `4dgs/scripts/composite_4dgs_renders.py` if object/background are separate render passes.

### Gotchas

- Prefer **pose-delta warping** pattern from `warp_gaussians_to_frame.py` if sim ↔ GS frame offset persists.
- Camera convention must match fudan DyNeRF (Blender → OpenCV flip in `render_4dgs_trajectory.py`).

### Acceptance criteria

- [ ] One PNG per (frame, camera) on test set
- [ ] Ball visible at plausible location
- [ ] No double ball (canonical only)

---

# Phase 5 — Evaluation

**Goal:** Trajectory + rendering metrics on held-out frames.

### Inputs

- `trajectory_predicted.csv`, `<SCENE_DIR>/object_poses.csv`
- `phase4/renders/`, GT RGB under `<SCENE_DIR>/rgb/` (test frames)

### Outputs (`phase5/`)

- `metrics.json`, `metrics_plot.png`

### Metrics

- Position RMSE (3D)
- Velocity / acceleration correlation (finite differences on train+test or test-only — document choice)
- **Rendering:** reuse `4dgs/scripts/eval_4dgs_metrics.py` (PSNR, MAE) where filenames align

### Milestone targets (tune per scene)

- [ ] `pos_rmse_m` &lt; 0.05
- [ ] `vel_r2` &gt; 0.9
- [ ] `psnr_db` &gt; 25, `ssim` &gt; 0.85 (if SSIM added)

---

## Reuse vs build

| Phase | Reuse now | Build |
|-------|-----------|--------|
| 1 | `fourdgs_common`, `export_4dgs_ply`, `export_meta` / `frame_map` | `load_4dgs` + centroid loop in `src/phys4d/bounce/extract.py` |
| 2 | `differentiable_bounce` (1D reference), `eval_physics_trajectory_split` pattern | 3D `simulate` + `scipy` fit in `src/phys4d/bounce/physics.py` |
| 3 | — | `src/phys4d/bounce/extrapolate.py` |
| 4 | `gaussian_ply`, `warp_gaussians_to_frame`, `render_4dgs_trajectory`, `composite_4dgs_renders` | Background train + merge entrypoint |
| 5 | `eval_4dgs_metrics`, `trajectory_metrics` | Held-out pose RMSE + report script |

---

## Planned repo layout (implementation)

```text
src/phys4d/
  bounce/
    load_4dgs.py      # Phase 1 — checkpoint + deformation
    extract.py        # centroid, smooth, validate
    physics.py        # Phases 2–3 — detect, fit, simulate
    render_compose.py # Phase 4 — translate + merge hooks
    metrics.py        # Phase 5
  # sync from cs231n_project_phys4d/src/phys4d/: gaussian_ply, poses, rigid, ...

scripts/bounce/
  run_phase1.py … run_phase5.py

outputs/bounce_pipeline/
  <RUN_NAME>/
    phase1/ … phase5/
```

Do **not** add top-level `scripts/run_phase1.py` without `bounce/` prefix — avoids clashing with visual-dynamics and 4dgs scripts.

---

## End-to-end CLI (target — not implemented)

```bash
# From repo root; set paths for your scene.
SCENE_DIR="dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p90_a0p0"
RUN_NAME="ball_drop_e0p90_a0p0_object"
CKPT="${SCENE_DIR}/4dgs/${RUN_NAME}/chkpnt30000.pth"
EXPORT="4dgs/experiments/object_only/runs/${RUN_NAME}"
OUT="outputs/bounce_pipeline/${RUN_NAME}"

# Modal (recommended):
arch -arm64 modal run modal_app.py --bounce-phase1 \
  --bounce-phase1-model 4dgs_ball_drop_e0p78_a0p0_object_all12_allframes \
  --bounce-phase1-checkpoint chkpnt15000.pth \
  --bounce-phase1-config room_physics_4dgs_4p0s.yaml \
  --bounce-phase1-gt-poses "${SCENE_DIR}/object_poses.csv" \
  --bounce-phase1-out-rel bounce_pipeline/${RUN_NAME}/phase1

# Local (needs Fudan clone + GPU/CUDA ops):
python scripts/bounce/run_phase1.py \
  --checkpoint "$CKPT" \
  --dynerf-export "$EXPORT" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --out "${OUT}/phase1"

python scripts/bounce/run_phase2.py \
  --traj "${OUT}/phase1/trajectory_smoothed.csv" \
  --fps 60 --out "${OUT}/phase2"

python scripts/bounce/run_phase3.py \
  --params "${OUT}/phase2/physics_params.json" \
  --scene-config "${SCENE_DIR}/config.json" \
  --frame-map "$EXPORT/frame_map.json" \
  --out "${OUT}/phase3"

python scripts/bounce/run_phase4.py \
  --checkpoint "$CKPT" \
  --bg-ply "${SCENE_DIR}/4dgs/${RUN_NAME}_background/point_cloud/exported/point_cloud.ply" \
  --predicted "${OUT}/phase3/trajectory_predicted.csv" \
  --ref-traj "${OUT}/phase1/trajectory_smoothed.csv" \
  --dynerf-export "$EXPORT" \
  --out "${OUT}/phase4"

python scripts/bounce/run_phase5.py \
  --predicted "${OUT}/phase3/trajectory_predicted.csv" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --rendered "${OUT}/phase4/renders" \
  --gt-rgb-root "${SCENE_DIR}/rgb" \
  --out "${OUT}/phase5"
```

**Prerequisites before running:** sync `src/phys4d` modules; train 4DGS with **train-only** timestamps for extrapolation eval; implement background path for Phase 4.

---

## Immediate next steps (implementation order)

1. Sync `cs231n_project_phys4d/src/phys4d/{gaussian_ply,poses,rigid,...}` → root `src/phys4d/`.
2. Re-export / re-train one scene **without** `--all-train` so `transforms_test.json` is meaningful.
3. Phase 1 script + manual check against `object_poses.csv` on `ball_drop_e0p90_a0p0_object_vis6_px50_all12`.
4. Phase 2–3 on CPU; Phase 4 blocked on background 3DGS design.
5. Phase 5 wire to `eval_4dgs_metrics.py` for renders.

Update this file when a phase lands or when split/checkpoint conventions change.
