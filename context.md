# Bounce Pipeline — Context & Reproduction Guide

Physics-grounded prediction + Gaussian-splat re-rendering of a bouncing ball.
A Wu 4DGaussians object reconstruction is used to (a) extract the ball's metric
trajectory, (b) fit a ballistic+bounce physics model, (c) forward-predict the
held-out future, and (d) re-render that prediction as a 2D-matte composite over a
static 3DGS background, scored against ground truth.

Worked example below: **`wu_ball12_2s_blue_e93`, object 4DGS trained to 50k iters**
(scene `0601_scene_0000_e0p93_a0p0`, restitution e=0.93, 120 fps, 361 frames).

---

## 1. Data inputs

### 1a. Object 4DGS checkpoint (the only checkpoint-specific input)
A Wu/hustvl 4DGaussians **object-only** reconstruction, trained on the first 2 s
(frames 0–240). Four files are needed:

| File | Purpose |
|---|---|
| `object/point_cloud.ply` | canonical Gaussians (xyz, scale, rot, opacity, SH) |
| `object/deformation.pth` | time-deformation MLP weights |
| `object/deformation_table.pth` | deformation bookkeeping |
| `object/cfg_args` | Wu training args (SH degree, bounds, etc.) |

In this repo the 50k object lives in `./object/` (≈93 MB ply).
The 75k variant lives in `./75k_first_run/_step4a_stage/object/`.

### 1b. Scene export (checkpoint-independent — shared across all object iters)
Same physical capture, so reused verbatim between 50k/75k runs:

| File | Purpose |
|---|---|
| `export/frame_map.json` | maps Wu normalized time → original PyBullet frame index |
| `export/transforms_train.json` | train camera poses (Blender/OpenGL convention) |
| `export/transforms_test.json` | **held-out** test cameras (cams 0,2,4,6 × frames 241–360 = 480 views) |
| `scene/object_poses.csv` | PyBullet ground-truth object poses (position + quaternion + velocities) |

Canonical copies are staged under `_gdrive_staging/CS231N_Project/0601_scene_0000_e0p93_a0p0/`
(`object_poses.csv`, `rgb/cam00..cam11/frameNNNNN.png`).

### 1c. Static background 3DGS
A vanilla 3DGS of the empty scene (ball region temporal-median-filled to avoid a
hole). On the Modal volume: `bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply`.

### Train / test split
- **Train:** frames 0–240 (first 2 s) — fit physics here.
- **Test (held-out):** frames 241–360 — pure forward prediction, scored vs GT.

---

## 2. What to run

All GPU steps run on Modal (A10G). `export HOME=/Users/<you>` first so the
`modal` CLI finds its token; CPU steps use the local `.venv_pipeline`.

### Step 4a — extract object trajectory (Modal GPU)
Stage `object/` + reuse `export/` + `scene/` into one dir, upload, extract:

```bash
modal run modal_app.py --upload-step4a --step4a-dir 50k_rerun/_step4a_stage
modal run modal_app.py --step4a --bounce-out-rel step4a_50k
```

The per-frame object center is the weighted geometric median of the fixed,
time-tracked object Gaussian set, followed by light Savitzky–Golay smoothing in
Step 4b (window chosen there, not here). This is the only extraction method in the
pipeline — it makes no assumption about the object's mass distribution and was the
most robust across checkpoints we tried (see §3).

### Step 4b/4c — fit physics + predict (local CPU)
Umeyama-align the unscaled 4DGS trajectory to metric GT, fit gravity(fixed)+
restitution+ground-z on frames 0–240, then forward-integrate 241–360:

```bash
python scripts/bounce/refit_metric_gt.py --traj <trajectory_*.csv> \
  --gt-poses <scene>/object_poses.csv --out <OUT>/step4b_metric \
  --train-start 0 --train-end 240 --test-start 241 --test-end 360
python scripts/bounce/predict_metric_from_refit.py \
  --refit <OUT>/step4b_metric/refit_metric.json --out <OUT>/step4c/pred.csv \
  --frame-start 241 --frame-end 360 --anchor-frame 0 --fps 120 --zero-horizontal
```

The refit emits `similarity.scale` — this **is** the `object_scale` used in Step 5.

### Step 5 — composite render (Modal GPU)
Upload object ply + cfg, the predicted CSV, a reference trajectory, then render the
object as a separate Gaussian cloud and **alpha-composite it in 2D** over the
background (avoids cross-occlusion from background floaters):

```bash
# uploads: object_50k/{point_cloud.ply,cfg_args}, step4c/trajectory_predicted.csv,
#          step4a/trajectory_smoothed.csv  (all under phys4d-gs-data:/step5)
modal run modal_app.py --step5 \
  --canonical-rel object/point_cloud.ply --cfg-args-rel object/cfg_args \
  --bounce-bg-ply-rel bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply \
  --object-scale 0.7057883553969182 --object-max-scale 0.03 \
  --object-crop-radius 0.24 \
  --object-opacity-boost 4.0 --composite-mode alpha
```

Recipe knobs (clean, generalizable — no per-splat surgery):
- `--object-max-scale 0.03` drop huge "spill" splats (optional).
- `--object-crop-radius` crop the diffuse halo in 4DGS units. Choose so
  `crop × object_scale ≈ 0.17 m` (matches the GT ball's visible radius). For 50k
  (scale 0.706) that is **0.24**; for 75k (scale 0.436) it is **~0.39**.
- `--composite-mode alpha` layered 2D alpha-matte (default). Use `--composite-mode
  merge3d` or `--no-composite-2d` only for the old single-pass 3D merge.
- `--object-opacity-boost 4.0` makes semi-transparent object splats read solid.

### Step 6 — evaluate (local CPU)
```bash
python scripts/bounce/step6_eval.py --predicted <OUT>/step4c/pred.csv \
  --gt-poses <scene>/object_poses.csv --rendered <OUT>/step5_rerun_renders \
  --gt-rgb-root <scene>/rgb --out <OUT>/step6 --fps 120
```

---

## 3. Results — 50k rerun

Smoothing sweep (median extraction; refit on frames 0–240, predict 241–360, position
RMSE vs GT):

| Trajectory variant | pos RMSE | restitution e | scale | milestone <0.05 m |
|---|---|---|---|---|
| median, raw | 0.1214 m | 0.915 | 0.705 | MISS |
| **median, savgol-w9** | **0.0406 m** | **0.901** | **0.706** | **PASS** ✓ |

We also explored a compact-core centroid estimator that *improved* the 75k checkpoint
(0.0556 → 0.0343 m) but **hurt** the 50k checkpoint (0.041 → ~0.10 m): it assumes a
dense opaque core + diffuse halo, which holds for one checkpoint's geometry and not the
other. Because it did not generalize it was removed from the pipeline; the full-set
median + light Savitzky–Golay smoothing is the single recipe. The old 75k core outputs
are retained only as data records (`step4a_core/`, `rigid_tracks.npz`), not as a code path.

**Adopted 50k result (median + savgol-w9):**

| Metric | Value | Target | Pass |
|---|---|---|---|
| position RMSE | **0.0406 m** | < 0.05 m | ✓ |
| velocity R² | 0.270 | > 0.9 | ✗ (bounce-phase drift) |
| fitted restitution e | 0.901 | (GT 0.93) | — |
| object_scale (Umeyama) | 0.7058 | — | — |
| render PSNR | 20.7 dB | > 25 dB | ✗ |
| render SSIM | 0.802 | > 0.85 | ✗ |
| render MAE | 0.095 | — | — |

Render: 480 held-out views (cams 0,2,4,6 × frames 241–360), 2D-matte composite,
102,515 object Gaussians after filter+crop. PSNR/SSIM are whole-frame (dominated by
background); the object itself is solid and well-placed.

### Output artifacts (`outputs/bounce_pipeline/wu_ball12_2s_blue_e93_iter50k/`)
- `step4a_fixedset/` — median trajectory (old checkpoint, untouched)
- `step4a_core/` — new core trajectory + `rigid_tracks.npz`
- `step4b_metric_rerun/refit_metric.json` — adopted physics fit (median-w9)
- `step4c_metric_rerun/pred_w9.csv` — predicted metric trajectory 241–360
- `step5_rerun_renders/` — 480 composite PNGs
- `step6_rerun_full/metrics.json` — full eval (trajectory + render)
- `analysis_xyz_50k_rerun.png` — GT vs extracted vs prediction (x/y/z)
- `ring4_50k_rerun_vs_gt.png` — 4-camera × 4-frame composite vs GT grid

---

## 4. Limitations (what caps accuracy)
- **Centroid measurement:** tracking the centroid of a diffuse/deforming Gaussian
  blob is noisy (±mm in x/y); it is not a true rigid pose, and spin is not recovered.
- **Restitution weakly identified:** only ~2 bounces in the 2 s train window; fitted
  e (0.90) underestimates GT (0.93) → bounce-phase drift in extrapolation (low
  vel_R²). Position envelope stays good (RMSE < 5 cm).
- **Contact model:** instantaneous, single restitution, flat fitted ground, zero
  horizontal velocity, spin ignored (GT carries angular velocity).
- **Umeyama coupling:** metric scale/rotation fit from the same noisy trajectory;
  scale differs markedly between checkpoints (0.706 @50k vs 0.436 @75k).
