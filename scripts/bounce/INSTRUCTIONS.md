# Bounce Pipeline Instructions (Phases 1–5)

Practical runbook for the ball-bounce pipeline in `cs231n_project_phys4d.nosync`.

Scope: `phys4d_final` ball-drop scenes with Fudan 4DGS.

## Progress (last updated)

| Phase | Status | CLI / code |
|-------|--------|------------|
| **1** Trajectory from 4DGS | **Done** | `run_phase1.py`, Modal `--bounce-phase1` |
| **2** Physics fit | **Done** | `run_phase2.py`, `physics.py` |
| **3** Held-out extrapolation | **Done** | `run_phase3.py`, `extrapolate.py` |
| **4** Render predicted poses | **Done** | `run_phase4.py`, `render_compose.py` |
| **5** Metrics / eval | **Done** | `run_phase5.py`, `metrics.py` |

**Validated split run:** `ball_drop_e0p78_a0p0_object` (scene `scene_0004_e0p78_a0p0`, 60 train + 30 test frames @ 60 fps).

Outputs live under `outputs/bounce_pipeline/<RUN_NAME>/phase{1..5}/`.

Spec: [`bouncepipeline.md`](../../bouncepipeline.md).

---

## How to view the latest results

All paths below assume repo root and:

```bash
export RUN_NAME="ball_drop_e0p78_a0p0_object"
export OUT="outputs/bounce_pipeline/${RUN_NAME}"
export SCENE_DIR="dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p78_a0p0"
export EXPORT="4dgs/experiments/object_only/runs/${RUN_NAME}"
```

**Note:** After Modal Phase 1 download, trajectories often live at  
`$OUT/phase1/phase1/trajectory_smoothed.csv` (nested `phase1/`), not `$OUT/phase1/trajectory_smoothed.csv`.

### Quick open (macOS Finder)

```bash
open "$OUT/phase2/physics_fit_plot.png"
open "$OUT/phase3/trajectory_full_plot.png"
open "$OUT/phase5/metrics_plot.png"
open "$OUT/phase4/renders"
```

### Phase-by-phase artifacts

| Phase | What to open | Path |
|-------|----------------|------|
| 1 | Train trajectory + plot | `$OUT/phase1/phase1/trajectory_smoothed.csv`, `trajectory_plot.png` (same folder if flat download) |
| 2 | Physics fit | `$OUT/phase2/physics_params.json`, `physics_fit_plot.png` |
| 3 | Held-out prediction | `$OUT/phase3/trajectory_predicted.csv`, `trajectory_full_plot.png`, `phase3_meta.json` |
| 4 | Predicted renders (PNG) | `$OUT/phase4/renders/*.png` |
| 5 | Scores | `$OUT/phase5/metrics.json`, `metrics_plot.png` |

Tuning variants (if you ran `phase2_tune_*`, etc.) use the same layout, e.g. `$OUT/phase4_tune_a/renders`, `$OUT/phase5_tune_a/metrics.json`.

### Interactive viewer (recommended for Phase 4)

Build once (or rebuild after new renders):

```bash
python 4dgs/scripts/view_4dgs_time.py build \
  --render-dir "$OUT/phase4/renders" \
  --dataset "$EXPORT" \
  --out "$OUT/phase4/viewer"
```

Serve (images load reliably vs `file://`):

```bash
python 4dgs/scripts/view_4dgs_time.py serve \
  --dir "$OUT/phase4/viewer" \
  --open
```

Controls: time slider, camera dropdown, play/spacebar, optional GT side-by-side (needs `--dataset` with `images/`).

**What you see in Phase 4 today:** masked scene background + soft red predicted Gaussian splat (not full object+background 3DGS raster yet). One ball only if GT ball was masked out.

### Compare to ground truth video

```bash
open "$SCENE_DIR/videos/rgb/cam10_front.mp4"   # example test camera
open "$SCENE_DIR/rgb/cam10/frame00060.png"     # single frame
```

### Metrics at a glance

```bash
python -m json.tool "$OUT/phase5/metrics.json"
```

Key fields: `trajectory.pos_rmse_m`, `trajectory.vel_r2`, `render.psnr_db_mean`, `render.mae_mean`.

---

## 0) Prerequisites

- Repo root:
  - `cd /Users/simoncasper/Desktop/CS231N/cs231n_project_phys4d.nosync`
- Environment:
  - `conda activate phys4d`
- You need:
  - A trained 4DGS checkpoint (`chkpnt*.pth`) on Modal (`phys4d-gs-output:/4dgs_<RUN_NAME>`)
  - A DyNeRF object-only export: `4dgs/experiments/object_only/runs/<RUN_NAME>/`
  - Split metadata: `export_meta.json` (or `frame_map.json`) with non-empty `test_timestamps`
  - Scene GT poses: `<SCENE_DIR>/object_poses.csv`

Important:

- Use a **split-respecting** export for held-out Phase 3 (`test_timestamps` 60–89, etc.).
- `_all12_allframes` / `--all-train` exports are fine for reconstruction and Phase 1–2 plots, but Phase 3 needs a test split in `export_meta.json`.
- Phase 1 on Modal extracts **train timestamps only** (~60 frames). Phase 3 builds the test time grid from `export_meta.json`; you do **not** need test rows in the Phase 1 CSV.

### One-shot env vars (split example)

```bash
export RUN_NAME="ball_drop_e0p78_a0p0_object"
export SCENE_DIR="dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p78_a0p0"
export OUT_ROOT="outputs/bounce_pipeline/${RUN_NAME}"
```

---

## 1) Phase 1 — Trajectory extraction (implemented)

Goal: Per-frame 3D centroid from trained 4DGS on **train** timestamps (opacity × marginal weighted).

Outputs (on Modal volume and after download):

- `trajectory_raw.csv`
- `trajectory_smoothed.csv`
- `trajectory_plot.png`
- `phase1_meta.json`

### 1a) Train 4DGS on Modal (if not done)

Use the same `<RUN_NAME>` export you uploaded to `/data/4d_scene`. See `4dgs/scripts/train_one_final_scene_4dgs.sh` or `modal_app.py --train-4d`.

### 1b) Upload DyNeRF export (if needed)

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py \
  "4dgs/experiments/object_only/runs/${RUN_NAME}" \
  --modal-cmd "arch -arm64 modal"
```

### 1c) Run Phase 1 on Modal

```bash
arch -arm64 modal run modal_app.py --bounce-phase1 \
  --bounce-phase1-config room_physics_4dgs_4p0s.yaml \
  --bounce-phase1-model "4dgs_${RUN_NAME}" \
  --bounce-phase1-checkpoint chkpnt15000.pth \
  --bounce-phase1-gt-poses "${SCENE_DIR}/object_poses.csv" \
  --bounce-phase1-out-rel "bounce_pipeline/${RUN_NAME}/phase1"
```

Expect `"num_frames": 60` for the default split (train window only).

### 1d) Download

```bash
mkdir -p "${OUT_ROOT}/phase1"

arch -arm64 modal volume get phys4d-gs-output \
  "bounce_pipeline/${RUN_NAME}/phase1" \
  "${OUT_ROOT}/phase1" \
  --force
```

**Nested path:** `modal volume get` often creates `phase1/phase1/trajectory_smoothed.csv`. Use that path in Phases 2–3, or flatten:

```bash
# optional: mv "${OUT_ROOT}/phase1/phase1"/* "${OUT_ROOT}/phase1/" && rmdir "${OUT_ROOT}/phase1/phase1"
```

Set trajectory path for later steps:

```bash
# pick whichever exists
TRAJ="${OUT_ROOT}/phase1/phase1/trajectory_smoothed.csv"
[ -f "$TRAJ" ] || TRAJ="${OUT_ROOT}/phase1/trajectory_smoothed.csv"
```

### Local Phase 1 (only if Fudan CUDA ops work)

Mac usually fails on `pointops2_cuda` — use Modal.

```bash
python scripts/bounce/run_phase1.py \
  --checkpoint <CKPT_PATH> \
  --config <4DGS_CONFIG_YAML> \
  --dynerf-export "4dgs/experiments/object_only/runs/${RUN_NAME}" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --out "${OUT_ROOT}/phase1" \
  --device cuda \
  --fourd-root <FOURDGS_ROOT>
```

Optional `--include-test` adds test timestamps to the CSV (debug plots only; not required for Phase 3).

Checks:

- Monotonic `t_sec`, no NaNs
- `trajectory_plot.png` shows plausible bounce motion on the train window (~1 s)

---

## 2) Phase 2 — Physics fitting (implemented)

Goal: Fit discrete bounce simulator to `trajectory_smoothed.csv`.

Outputs:

- `physics_params.json`
- `physics_fit_plot.png`

```bash
python scripts/bounce/run_phase2.py \
  --traj "${TRAJ}" \
  --out "${OUT_ROOT}/phase2" \
  --fix-gravity
```

Checks:

- `restitution` in `[0.3, 0.95]` (roughly)
- `fit_mse_m2` small on train window (split run ~`0.01–0.05` is common)
- `bounce_count` vs `physics_fit_plot.png` (split train ~1 s → often **1** bounce in fit; use longer train export for more bounces)

---

## 3) Phase 3 — Held-out extrapolation (implemented)

Goal: Roll fitted physics forward on **test** timestamps from the export split. Phase 1 CSV supplies train observations; `export_meta.json` supplies test frame ids and times.

Outputs:

- `trajectory_predicted.csv` (test-only rows)
- `trajectory_full_plot.png`
- `phase3_meta.json`

```bash
python scripts/bounce/run_phase3.py \
  --params "${OUT_ROOT}/phase2/physics_params.json" \
  --traj "${TRAJ}" \
  --dynerf-export "4dgs/experiments/object_only/runs/${RUN_NAME}" \
  --out "${OUT_ROOT}/phase3" \
  --fps 60
```

Checks:

- `num_test_frames` = 30 for default split (frames 60–89)
- `phase3/trajectory_predicted.csv` has 30 rows
- `boundary_gap_m` in `phase3_meta.json` — gap between last train centroid and first test prediction (~`0.01–0.05` m is typical with a short train window)
- `min_clearance_to_ground_m >= -epsilon` (no floor penetration)

### Troubleshooting Phase 3

| Error | Cause | Fix |
|-------|--------|-----|
| `No held-out test timestamps` | `--all-train` export | Re-export with train/test split |
| `Missing trajectory CSV` | Wrong path after Modal download | Use `phase1/phase1/...` or set `TRAJ` as above |
| `No overlap` (old code) | Outdated `extrapolate.py` | Pull latest; Phase 3 no longer requires test rows in Phase 1 CSV |
| `frame_map.json` not found | Some exports only have `export_meta.json` | OK — Phase 3 reads `export_meta.json` |

---

## 4) Phase 4 — Render predicted trajectory (implemented)

Goal: Held-out test views with predicted ball position overlaid on scene RGB.

Outputs:

- `phase4/renders/<seq>_cam<idx>_<frame>.png`
- `phase4_meta.json` (`projection_mode`, `render_style`, etc.)
- optional `phase4/viewer/` (from `view_4dgs_time.py`)

```bash
python scripts/bounce/run_phase4.py \
  --predicted "$OUT/phase3/trajectory_predicted.csv" \
  --dynerf-export "$EXPORT" \
  --scene-dir "$SCENE_DIR" \
  --out "$OUT/phase4" \
  --render-style gaussian_splat
```

See **How to view the latest results** above for the viewer.

---

## 5) Phase 5 — Evaluate trajectory/render quality (implemented)

Goal: Held-out trajectory + render metrics vs GT.

Outputs: `metrics.json`, `metrics_plot.png`

```bash
python scripts/bounce/run_phase5.py \
  --predicted "$OUT/phase3/trajectory_predicted.csv" \
  --gt-poses "$SCENE_DIR/object_poses.csv" \
  --rendered "$OUT/phase4/renders" \
  --gt-rgb-root "$SCENE_DIR/rgb" \
  --out "$OUT/phase5" \
  --fps 60
```

---

## Suggested split for bounce-count experiments

For ~2 bounces in train and ~2 in test at 60 fps:

- split export with train boundary near `t = 1.0 s` (`frame ~60` train, `60–89` test)
- or extend `train_frame_range` in export / scene config before retraining 4DGS

---

## Quick file map

| Piece | Path |
|-------|------|
| Phase 1 | `src/phys4d/bounce/load_4dgs.py`, `extract.py`, `scripts/bounce/run_phase1.py` |
| Phase 2 | `src/phys4d/bounce/physics.py`, `scripts/bounce/run_phase2.py` |
| Phase 3 | `src/phys4d/bounce/extrapolate.py`, `scripts/bounce/run_phase3.py` |
| Phase 4 | `src/phys4d/bounce/render_compose.py`, `scripts/bounce/run_phase4.py` |
| Phase 5 | `src/phys4d/bounce/metrics.py`, `scripts/bounce/run_phase5.py` |
| Viewer | `4dgs/scripts/view_4dgs_time.py` |
| Modal Phase 1 | `modal_app.py` → `bounce_phase1_remote` |
| Spec | `bouncepipeline.md` |
| **View results** | **This file — section “How to view the latest results”** |
