# Ball Bounce — Pipeline Spec (Milestone 2-aligned)

**Goal:** Given multi-view PyBullet ball-drop video, train on the first half, predict the second half. Object segmentation + physics priors turn 4D reconstruction into prediction.

**Scope:** Ball drop scenario. The pipeline structure generalizes to collision / stacking / deformable (see [Generalization](#generalization-to-other-scenarios)); details below are for the rigid sphere case.

**4DGS backend:** [hustvl/4DGaussians (Wu et al., CVPR 2024)](https://github.com/hustvl/4DGaussians).
Canonical 3D Gaussians + deformation field — chosen because the "freeze canonical, rigidly warp by physics" pattern in Step 5 is essentially free with this architecture.

---

## Pipeline overview (per Milestone 2 slide 3)

```
1. Video Input  ──►  2. Object Segmentation (SAM2)  ──►  3. 4D Reconstruction (Wu 4DGS, object-only)
                              [OURS]                            [adapted from baseline]
                                                                          │
                                                                          ▼
                                                         4. Trajectory Estimation & Extrapolation
                                                                       [OURS]
                                                                          │
                                                                          ▼
                                                            5. Rebuild final scene  ──►  6. Evaluate
                                                                       [OURS]
```

**Train / test split:** first 80% of each PyBullet video for training; held-out final 20% for prediction and scoring. The exact split is recorded in each scene's `config.json` (`train_frames`, `test_frames`).

**Headline contribution (Milestone 2 slide 6):** none of the baselines (4DGS, PhysGaussian, GASP) extrapolate object trajectory beyond the observed video. This pipeline does, by **fitting transformation matrices (not neural networks)** in Step 4 and applying them to the canonical Gaussians in Step 5.

---

## Status — where we are now

| Step | Status | Notes |
|---|---|---|
| 1. PyBullet data gen | **done** | 4 scenarios × 10 variants × 12 views, 60 fps |
| 2. SAM2 segmentation | **done** | Object-only mask validated on synthetic; IoU/F against PyBullet oracle |
| 3. Wu 4DGS training | **done** (per reference scene) | Verify each new run is **train-only** (no `--all-train`) |
| **4a. Trajectory extraction** | **next** | One Wu-touching module (~30 LOC), produces a CSV |
| **4b. Physics fit** | **next** | scipy/NumPy on CSV — no 4DGS code |
| **4c. Extrapolation** | **next** | NumPy on params — no 4DGS code |
| 5. Rebuild final scene | **remaining** | Background 3DGS + rigid warp + render |
| 6. Evaluation | **remaining** | Trajectory metrics + PSNR/SSIM |

Step 4a is the only place where the implementation touches the Wu codebase. Once `trajectory_smoothed.csv` exists, Steps 4b/4c are backend-agnostic.

---

# Step 1 — Video Input *(done)*

PyBullet synthetic data: 4 scenarios × 10 variants × 12 views, 60 fps (FPS in `export_meta.json`).

**Per scene directory:**
```
scene_<id>/
  config.json          # scenario, variant params, train_frames, test_frames
  rgb/<cam>/<frame>.png
  masks/<cam>/<frame>.png
  object_poses.csv     # PyBullet GT (x_m, y_m, z_m, qx, qy, qz, qw per frame)
  cameras.json         # intrinsics + extrinsics per camera
```

---

# Step 2 — Object Segmentation (SAM2) *(done)*

SAM2 video segmentation, object-only mask (better than object+table per M2 slide 5).
Temporal sparsity filter for collision scenarios (drop frames with < `c` object pixels).
Validated with IoU (`J`) and F-Measure (`F`) against PyBullet oracle masks.

Output: `masks_sam2/<cam>/<frame>.png`.

---

# Step 3 — 4D Reconstruction (Wu et al. 4DGS, object-only) *(done per reference scene)*

Wu et al. 4DGaussians trained on the segmented (object-only) frames, **train indices only** (no `--all-train`).

Loss: standard Wu et al. `L = |Î - I| + L_tv`. The optional `L_physics` regularizer from M2 slide 7 is deferred — post-hoc fit in Step 4 is cleaner; revisit as an ablation if time permits.

## Artifacts produced (consumed by Step 4a)

```
output/<scene>/
  point_cloud/iteration_30000/
    point_cloud.ply            # canonical 3D Gaussians (xyz, opacity logit, scale, rot, SH)
    deformation.pth            # deformation network state dict
  cameras.json
  cfg_args                     # training config — includes time range / time_duration
```

Plus the DyNeRF-style export used at train time:
```
4dgs/experiments/object_only/runs/<run_name>/
  transforms_train.json        # per-frame time values, camera poses
  frame_map.json               # kept_index ↔ original_frame
  export_meta.json             # fps, time_duration
```

## Critical training invariant

Train on train frames only. If `--all-train` slips through, the 4DGS has seen the test frames and held-out evaluation is invalid. **Step 4a should fail loudly** if `cfg_args` / training log indicates `--all-train`.

---

# Step 4 — Trajectory Estimation & Extrapolation (current focus)

**The core contribution.** Three sub-steps: extract → fit → roll forward.

> "Fit physics params (e.g., velocity, restitution) on observed part → roll forward → transform object position in 2nd part (beyond 4DGS). Fit transformation matrices instead of neural networks." — Milestone 2 slide 3

## Step 4a — Extract trajectory from Wu 4DGS

**Goal:** Per-frame 3D ball centroid `p̂(t)` over the train window, written as a CSV.

This is the only step that touches Wu code. Everything downstream operates on the CSV.

### Wu repo as dependency

Vendor the repo as a git submodule once:
```bash
git submodule add https://github.com/hustvl/4DGaussians third_party/4DGaussians
```
Import their classes rather than re-implementing the deformation architecture:
```python
import sys
sys.path.insert(0, "third_party/4DGaussians")
from scene.gaussian_model import GaussianModel
from scene.deformation import deform_network
```

### Extraction

```python
import torch, json, numpy as np

# 1. Load canonical Gaussians
gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply("output/<scene>/point_cloud/iteration_30000/point_cloud.ply")

# 2. Load deformation network
deform = deform_network(args).cuda()       # rebuild `args` from cfg_args
deform.load_state_dict(torch.load(
    "output/<scene>/point_cloud/iteration_30000/deformation.pth"))
deform.eval()

# 3. Canonical positions + opacity weights for centroid
mu0 = gaussians.get_xyz.detach()           # (N, 3)
w   = gaussians.get_opacity.detach().squeeze()   # sigmoid already applied by the getter

# 4. Time normalization — match the training-time mapping (see Gotcha #2 below)
with open("<dynerf_export>/transforms_train.json") as f:
    tt = json.load(f)
train_times = sorted(frame["time"] for frame in tt["frames"])
t_min, t_max = train_times[0], train_times[-1]
unique_times = sorted(set(train_times))
T_obs   = len(unique_times)
t_sec   = np.array(unique_times)
t_norm  = (t_sec - t_min) / (t_max - t_min)

# 5. Opacity-weighted centroid per frame
centroids = np.zeros((T_obs, 3))
with torch.no_grad():
    for i, tn in enumerate(t_norm):
        time_input  = torch.full((mu0.shape[0], 1), float(tn), device="cuda")
        d_xyz, _, _ = deform.step(mu0, time_input)
        mu_t        = (mu0 + d_xyz).cpu().numpy()
        centroids[i] = (w.cpu().numpy()[:, None] * mu_t).sum(0) / w.sum().item()

# 6. Save — from here on, no more Wu code touches the data
np.savetxt("trajectory_raw.csv",
           np.c_[np.arange(T_obs), t_sec, centroids],
           header="frame,t_sec,x,y,z", delimiter=",", comments="")
```

Apply Savitzky-Golay (`window_length=5`, `polyorder=2`) per axis and write `trajectory_smoothed.csv`. Reduce polyorder to 1 if bounce inflections get rounded off.

### Outputs (`step4a/`)
- `trajectory_raw.csv`, `trajectory_smoothed.csv` — schema `frame, t_sec, x, y, z`
- `trajectory_plot.png` — raw / smoothed / GT overlay

### Acceptance criteria
- CSVs: `T_obs` rows, no NaNs, monotonic `t_sec`
- ≥ 1 visible bounce on the z-axis in the plot
- Smoothed curve preserves bounce cusps
- MSE vs `object_poses.csv` on train frames < 0.05 m (apply Procrustes alignment first if there's only a constant world-frame offset)

### Gotchas — the two that will actually bite

**1. Opacity getter vs raw field.** `gaussians.get_opacity` applies sigmoid; `gaussians._opacity` is the raw logit. Use the getter. If you instead load the PLY directly with `plyfile`, the `opacity` field is logit and you must apply `sigmoid` yourself — otherwise the weights are silently wrong (no crash, just biased centroid).

**2. Time normalization must match training exactly.** Wu's dataloader maps frame times to `[0, 1]` over the training time range. If you use a different normalization (e.g., over the full sim duration instead of the train window), the deformation will be queried at the wrong points and the trajectory will be stretched/compressed and bounces will land at the wrong frames. Read the times directly from `transforms_train.json` and normalize over that range, as shown above. **If the trajectory looks visibly off and the code structure seems correct, this is the first place to look.**

**3. World-frame offset.** 3DGS world frame ≠ PyBullet world frame. A constant offset is fine — Step 4b fits in shape + gravity direction, and Step 5 uses pose deltas, so the offset cancels. If MSE in validation looks bad but the *shape* of the trajectory matches GT, apply Procrustes before declaring failure.

## Step 4b — Fit physics parameters

**Goal:** Fit `(p₀, v₀, g, e, z_g)` such that a discrete bounce simulator reproduces `trajectory_smoothed.csv`.

**Backend-independent — pure scipy/NumPy on the CSV. No Wu code touches it.**

These are the **transformation matrices** in the M2-slide-3 sense (closed-form, not learned):

| Symbol | Meaning |
|---|---|
| `p₀ ∈ ℝ³` | initial position |
| `v₀ ∈ ℝ³` | initial velocity |
| `g`       | gravity (fix to −9.81 m/s² for first run, free as ablation) |
| `e ∈ [0,1]` | coefficient of restitution |
| `z_g`     | ground plane height |

### Method
1. **Detect bounces** — finite-difference `v_z`; sign flip (− → +) with sub-frame zero crossing; threshold on `|v_z|` to ignore apex noise.
2. **Initial guesses** — `p₀` from row 0; `v₀` from first ~5 frames; `g ≈ −9.81`; `z_g ≈ min(z)`; `e ≈ mean(-v_z_after / v_z_before)`.
3. **Joint nonlinear least squares** — `scipy.optimize.least_squares` with LM:
```python
def simulate(params, T, dt):
    p0, v0, g, e, z_g = unpack(params)
    p, v = p0.copy(), v0.copy()
    traj = np.zeros((T, 3))
    for i in range(T):
        traj[i] = p
        v[2] += g * dt
        p += v * dt
        if p[2] < z_g and v[2] < 0:
            v[2] = -e * v[2]
            p[2] = z_g
    return traj
```

### Outputs (`step4b/`)
- `physics_params.json`
- `physics_fit_plot.png`

### Acceptance criteria
- `e ∈ [0.3, 0.95]`
- `g` within ±15% of −9.81 (if free)
- `fit_mse` < 0.01 m² on train
- Detected bounce count matches the plot

## Step 4c — Roll forward (extrapolate)

**Goal:** Predict ball positions for the held-out test frames.

**Backend-independent — NumPy only.**

Advance the fitted simulator from its state **at the end of the train window** (not from `p₀`). Write positions for test frame indices only.

### Outputs (`step4c/`)
- `trajectory_predicted.csv` — test frames only
- `trajectory_full_plot.png` — train (observed) + predicted + GT

### Acceptance criteria
- Row count matches the number of held-out frames
- Boundary continuity: gap between last observed and first predicted < 1 cm
- No floor penetration: `z ≥ z_g − ε`

---

# Step 5 — Rebuild final scene (remaining)

**Goal:** For each held-out frame, render `(translated canonical object Gaussians) ∪ (static background)` from the test cameras.

## Step 5a — Background 3DGS (one-time, parallelizable with Step 4)

Train a static 3DGS of the room/table once per scene. Two options:
- (a) Standard 3DGS on a single frame where the ball is out of view, or
- (b) 3DGS across multi-view frames with the **inverted** object mask.

Output: `background.ply`. **Step 5 blocker** — should start in parallel with Step 4, not after.

## Step 5b — Per-frame rigid warp

For each predicted test frame `t`:
```python
delta = p_pred[t] - p_ref           # p_ref = canonical centroid (or trajectory frame 0)
mu_t  = canonical.xyz + delta       # pure translation; sphere → no rotation
```
All other Gaussian attributes (scale, rotation, opacity, SH) unchanged.

**Do not also apply the deformation field.** The deformation encodes observed-window motion; applying both it and the physics translation double-moves the ball. This is the entire reason Wu et al. is the right backend — the canonical set is already a valid static 3DGS, so "freeze and translate" is free.

## Step 5c — Composite & rasterize

```python
combined = concat_gaussians(translated_object, background)
image    = rasterize(combined, camera_pose=poses[t])
save(image, f"renders/{t:04d}_cam{c}.png")
```
Use the standard 3DGS rasterizer shipped with the Wu codebase.

### Outputs (`step5/`)
- `renders/<frame>_cam<idx>.png`
- `composite_video.mp4` (optional)

### Acceptance criteria
- One PNG per (test frame, camera)
- Ball visible at a plausible position
- No double image of the ball (confirms canonical only, no deformation)

### Fallback if renders are visually rough
The headline contribution is the **trajectory**, not photorealism. If the composite disappoints, present a **trajectory overlay** instead: predicted ball position drawn over a static background render, with GT alongside.

---

# Step 6 — Evaluation (remaining)

## Trajectory metrics (held-out frames)
- Position RMSE: `sqrt(mean(‖p_pred(t) − p_gt(t)‖²))`
- Velocity / acceleration R² (Pearson, via finite differences)

## Rendering metrics (held-out frames, all cameras)
- PSNR, SSIM against PyBullet GT renders

## Milestone targets (tune per scene)
- Position RMSE < 0.05 m
- Velocity R² > 0.9
- PSNR > 25 dB, SSIM > 0.85

### Outputs (`step6/`)
- `metrics.json`, `metrics_plot.png`

---

# End-to-end CLI

```bash
SCENE_DIR="dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_<id>"
RUN_NAME="ball_drop_<variant>_object"
WU_OUT="${SCENE_DIR}/4dgs_wu/${RUN_NAME}"
EXPORT="4dgs/experiments/object_only/runs/${RUN_NAME}"
OUT="outputs/bounce_pipeline/${RUN_NAME}"

# Step 4a — extract trajectory (only Wu-touching step)
python scripts/bounce/step4a_extract.py \
  --canonical "${WU_OUT}/point_cloud/iteration_30000/point_cloud.ply" \
  --deform    "${WU_OUT}/point_cloud/iteration_30000/deformation.pth" \
  --cfg-args  "${WU_OUT}/cfg_args" \
  --dynerf-export "$EXPORT" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --out "${OUT}/step4a"

# Step 4b — fit physics (CSV in, JSON out)
python scripts/bounce/step4b_fit.py \
  --traj "${OUT}/step4a/trajectory_smoothed.csv" --out "${OUT}/step4b"

# Step 4c — extrapolate (JSON in, CSV out)
python scripts/bounce/step4c_predict.py \
  --params "${OUT}/step4b/physics_params.json" \
  --scene-config "${SCENE_DIR}/config.json" \
  --frame-map "$EXPORT/frame_map.json" \
  --out "${OUT}/step4c"

# Step 5 — background + render
python scripts/bounce/step5_render.py \
  --canonical "${WU_OUT}/point_cloud/iteration_30000/point_cloud.ply" \
  --bg-ply "${SCENE_DIR}/background_3dgs/background.ply" \
  --predicted "${OUT}/step4c/trajectory_predicted.csv" \
  --ref-traj "${OUT}/step4a/trajectory_smoothed.csv" \
  --dynerf-export "$EXPORT" \
  --out "${OUT}/step5"

# Step 6 — full eval (trajectory + rendering metrics)
python scripts/bounce/step6_eval.py \
  --predicted "${OUT}/step4c/trajectory_predicted.csv" \
  --gt-poses "${SCENE_DIR}/object_poses.csv" \
  --rendered "${OUT}/step5/renders" \
  --gt-rgb-root "${SCENE_DIR}/rgb" \
  --out "${OUT}/step6"
```

**Prerequisites (from earlier steps):**
1. Wu et al. 4DGS trained on **train-only** timestamps (Step 3).
2. `transforms_train.json` and `frame_map.json` available in the DyNeRF export.
3. `object_poses.csv` with PyBullet GT.

---

# Generalization to other scenarios

The same five-step structure carries over; only Steps 4 and 5 change.

| Scenario | Step 4 changes | Step 5 changes |
|---|---|---|
| **Collision** (two rigid bodies) | per-object trajectory + Procrustes rotation; PyBullet as forward model (gradient-free fit) | SE(3) warp per object |
| **Stacking** | adds friction as a fit parameter; contact-rich regime is harder | per-object SE(3) |
| **Deformable** | rigid-body assumption breaks — would need MPM or extrapolation of the 4DGS deformation field itself | per-Gaussian warp, no rigid factorization |

Ball drop is the cleanest case and the natural one to validate first.

---

# Repo layout

```
third_party/4DGaussians/         # git submodule — hustvl/4DGaussians
src/phys4d/bounce/
  load_4dgs.py        # Step 4a — wraps GaussianModel + deform_network
  extract.py          # centroid, smooth, validate
  physics.py          # Steps 4b–4c — detect, fit, simulate (backend-independent)
  render_compose.py   # Step 5 — translate canonical + merge
  metrics.py          # Step 6 — trajectory + rendering metrics
scripts/bounce/
  step4a_extract.py
  step4b_fit.py
  step4c_predict.py
  step5_render.py
  step6_eval.py
outputs/bounce_pipeline/<RUN_NAME>/step{4a,4b,4c,5,6}/
```

---

# Immediate next steps

1. Submodule `hustvl/4DGaussians` under `third_party/`.
2. Verify the existing Wu checkpoint was trained **train-only** (read `cfg_args`, fail loudly otherwise). If `--all-train` was used, retrain — non-negotiable for valid held-out eval.
3. Step 4a on the reference scene; validate trajectory against `object_poses.csv` (four acceptance criteria). The two gotchas (opacity getter, time normalization) are the most likely failure points.
4. Steps 4b–4c on CPU once Step 4a CSVs exist.
5. Background 3DGS in parallel with Steps 4b/4c (Step 5 blocker).
6. Step 5 render + Step 6 metrics once 4c and background are both done.