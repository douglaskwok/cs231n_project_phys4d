# Wu 4DGS Experiment Log

This log tracks the small-object Wu/HUSTVL 4DGaussians experiments so we do not lose the working settings.

## Next Debugging Checklist

Use this when the next run is blurry, overgrown, partially invisible, or temporally unstable. Do not change the data recipe casually; first diagnose the failure mode against the checks below.

### Non-Negotiable Data Recipe

- Use all 12 synchronized cameras.
- Preserve the native render size; do not resize to square or stretch aspect ratio.
- Use object-only spatial masks.
- Use temporal segmentation with `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.
- Preserve original timestamps and synchronized multiview frames.
- For the current Wu ball-bounce fitting target, use the first 2 seconds at 120 FPS with no frame stride.
- Do not hide a fitting failure by dropping cameras, skipping most frames, or making the ball absurdly large.

### Current Likely Failure Modes

- **Blur / halo / overgrown ball:** too many semi-transparent Gaussians survive around the object. Check render/GT foreground-area ratio and inspect contact sheets.
- **Ball disappears:** foreground loss/mask support is too weak, opacity reset or pruning may be killing object Gaussians, or initialization is outside the visible object support.
- **Temporal smearing:** Gaussians may be trying to explain multiple timestamps with the wrong temporal support instead of sharply representing the object at each time.
- **Aspect distortion:** first check image resize and camera metadata. We already saw elongation caused by resizing artifacts, not necessarily Wu itself.

### Literature-Guided Directions To Search/Implement

- Wu 4D-GS itself uses a deformation field over canonical 3D Gaussians and 4D neural voxel features, so our current pipeline is in the intended family of dynamic-scene representations rather than per-frame independent 3DGS.
- Several newer 4DGS papers point toward temporal-support fixes for dynamic artifacts:
  - temporal slicing / native XYZT primitives for abrupt motion,
  - temporal relevance or lifespan-aware pruning,
  - temporal opacity or lifespan modulation,
  - motion deblurring / sharpness losses for fast motion.
- If quality remains soft after reasonable fitting, prioritize opacity/temporal-support changes over naive extra iterations:
  - prune low-support halo Gaussians after fitting,
  - add an explicit silhouette or alpha/mask loss if the current RGB-only foreground loss cannot keep the ball compact,
  - tune opacity reset and densification schedules,
  - compare final checkpoints at multiple iterations instead of assuming the last checkpoint is best.

Useful references searched:

- Wu et al., `4D Gaussian Splatting for Real-Time Dynamic Scene Rendering`, arXiv:2310.08528.
- Duan et al., `4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes`, arXiv:2402.03307.
- Javed et al., `Temporally Compressed 3D Gaussian Splatting for Dynamic Scenes`, arXiv:2412.05700.
- `MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video`, arXiv:2504.15122.

### Immediate Next Actions

1. Finish the active `den4000_iter15k` run and build QA artifacts.
2. Compare it directly against `den4000_iter12k`.
3. If 15k is sharper without increasing area ratio or zero frames, mark it as best.
4. If 15k is worse, keep 12k as the current best and stop assuming more iterations solves blur.
5. Next serious experiment should be mask/opacity-aware compactness or post-fit pruning, not camera/frame reduction.

## Current Best Direction

Use object-only spatial masks plus temporal filtering, but do not train vanilla Wu loss on black-background object videos. Vanilla loss rewards black pixels too strongly and the object disappears.

Required settings so far:

- Spatial segmentation: object-only mask frames.
- Temporal segmentation: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.
- All 12 views in train: default `--all-train` in `train_one_scene_wu4dgs.sh`.
- Smaller Wu bounds for the object-only scene: `--bounds 0.8`.
- Foreground-weighted loss enabled from `modal_app.py`.
- PyBullet-guided `fused.ply` initialization enabled via `--init-points`.
- Limit densification aggressively and effectively disable opacity reset for the current best small-ball run.
- Current best smaller-ball Wu fit: radius `0.20m`, foreground weight `20`, densify until `2000`, no opacity reset.

## Replication Handoff For Fresh Context

This section is the minimum complete recipe for a new Codex instance to reproduce the current working Wu/HUSTVL 4DGaussians small-ball result. It intentionally focuses on the fitting-side settings that worked, not the full exploration history.

Treat this run as a possible baseline for comparison and ablation studies. It is not the final target quality; it is the first smaller-ball Wu setup that consistently renders the object across cameras and time. Future work should improve on this base while keeping the same evaluation artifacts and metrics so changes are comparable.

### What Worked

The current best smaller-ball Wu result is:

```text
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty
```

It reconstructs the object-only ball from:

```text
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0
```

That source scene is a short synthetic PyBullet room clip with:

- 12 synchronized camera views.
- 120 FPS.
- 0.5 seconds, 61 timestamps.
- Ball radius `0.20m`.
- Restitution `0.90`.
- Drop angle `0.0`.
- Object-only spatial masks in `masks/`.
- Original RGB views in `rgb/`.
- Camera/config metadata in `config.json` and related metadata files.

The successful Wu fit uses:

- Spatial segmentation: object-only mask frames.
- Temporal segmentation: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.
- All 12 cameras as train views.
- A single canonical object initialization from PyBullet object poses, not a trajectory tube:
  - `--init-points 10000`
  - `--init-center-mode first`
  - `--init-surface-ratio 0.85`
- Compact Wu bounds: `--bounds 0.8`.
- Foreground-weighted RGB loss: `--foreground-loss-weight 20`.
- Shorter densification: `--wu-densify-until-iter 2000`.
- Disabled opacity reset in practice: `--wu-opacity-reset-interval 300000`.
- Fine iterations: `8000`.
- Coarse iterations: `1000`.

The resulting QA numbers were:

- Point count: `30086`.
- Render foreground / GT foreground area ratio: mean `1.386`, median `1.264`.
- 732 rendered train views: `12 cameras x 61 timestamps`.
- Verdict: current smaller-ball baseline / ablation candidate. It is not perfect, but it is visible across cameras and time, and much less overgrown than the earlier `den4k` attempt.

### Exact Training Command

Run from the repository root:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

This script:

1. Exports the object-only scene into a Wu-compatible DyNeRF-style dataset under `4dgs/experiments/wu4dgs/runs/<run_name>`.
2. Uploads it to the Modal volume `phys4d-gs-data:/4d_scene`.
3. Trains Wu 4DGS on Modal.
4. Renders the trained Wu model on Modal.
5. Downloads the model and render artifacts into the scene-local `4dgs_wu/<run_name>` folder.

Expected downloaded artifacts:

```text
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/<run_name>/wu4dgs_<run_name>/point_cloud/iteration_8000/point_cloud.ply
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/<run_name>/wu4dgs_<run_name>/point_cloud/iteration_8000/deformation.pth
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/<run_name>/wu4dgs_<run_name>/chkpnt_fine_8000.pth
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/<run_name>/wu4dgs_<run_name>/train/ours_8000/renders/*.png
dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/<run_name>/wu4dgs_<run_name>/train/ours_8000/gt/*.png
```

### Build And Serve The Viewer

Build the local time viewer after the render exists:

```bash
phys_sim/bin/python 4dgs/scripts/view_4dgs_time.py build \
  --render-dir dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/wu4dgs_wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/train/ours_8000/renders \
  --dataset 4dgs/experiments/wu4dgs/runs/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --out dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer
```

Serve it:

```bash
phys_sim/bin/python 4dgs/scripts/view_4dgs_time.py serve \
  --dir dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer \
  --port 8765
```

Open:

```bash
open http://127.0.0.1:8765
```

The viewer supports time and camera selection, plus GT comparison if the build step was given `--dataset`.

### Validation Commands

Use a selected camera/time contact sheet to inspect shape and temporal consistency:

```bash
phys_sim/bin/python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

run = 'wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty'
base = Path('dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu') / run
root = base / ('wu4dgs_' + run) / 'train/ours_8000'
render = root / 'renders'
gt = root / 'gt'
out = base / 'qa_render_gt_selected.png'

times = [0, 15, 30, 45, 60]
cams = [0, 3, 6, 9]
thumb_w, thumb_h = 192, 109
label_h = 18
sheet = Image.new('RGB', (len(times) * 2 * thumb_w, len(cams) * (thumb_h + label_h)), (24, 24, 24))
d = ImageDraw.Draw(sheet)

for r, cam in enumerate(cams):
    for ti, t in enumerate(times):
        idx = cam * 61 + t
        for kind, folder in [('R', render), ('GT', gt)]:
            c = ti * 2 + (kind == 'GT')
            img = Image.open(folder / f'{idx:05d}.png').convert('RGB').resize((thumb_w, thumb_h))
            x = c * thumb_w
            y = r * (thumb_h + label_h) + label_h
            sheet.paste(img, (x, y))
            d.text((x + 4, y - label_h + 3), f'cam{cam:02d} t{t:02d} {kind}', fill=(230, 230, 230))

out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(out)
PY
```

Compute foreground-area ratios:

```bash
phys_sim/bin/python - <<'PY'
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

label = 'r0p20_fg20_den2k'
root = Path(
    'dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/'
    'wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/'
    'wu4dgs_wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/'
    'train/ours_8000'
)

rfiles = sorted((root / 'renders').glob('*.png'))
gfiles = sorted((root / 'gt').glob('*.png'))
rs, gs, ratios = [], [], []

for rp, gp in zip(rfiles, gfiles):
    r = imageio.imread(rp)[..., :3]
    g = imageio.imread(gp)[..., :3]
    ra = int((r.max(axis=2) > 8).sum())
    ga = int((g.max(axis=2) > 8).sum())
    rs.append(ra)
    gs.append(ga)
    ratios.append(ra / max(ga, 1))

print(label)
print('  renders/gt:', len(rfiles), len(gfiles))
print('  render_fg mean/min/max:', round(float(np.mean(rs)), 1), int(np.min(rs)), int(np.max(rs)))
print('  gt_fg mean/min/max:', round(float(np.mean(gs)), 1), int(np.min(gs)), int(np.max(gs)))
print('  ratio mean/median:', round(float(np.mean(ratios)), 3), round(float(np.median(ratios)), 3))
PY
```

Expected ratio for the current best: mean about `1.386`, median about `1.264`.

### Important Implementation Notes

Relevant local files:

- `4dgs/scripts/train_one_scene_wu4dgs.sh`
- `4dgs/experiments/object_only/export_object_only_dynerf.py`
- `modal_app.py`
- `4dgs/scripts/view_4dgs_time.py`
- `4dgs/scripts/render_index.py`

Important code behavior:

- `train_one_scene_wu4dgs.sh` defaults to all cameras as train views.
- `export_object_only_dynerf.py` must preserve synchronized multiview frames and original timestamps. Do not drop individual camera views unless explicitly testing that; for 4DGS the cameras need to remain aligned.
- `--drop-invisible-frames` drops an entire timestamp only if it fails the multi-camera visibility threshold. It should not desynchronize cameras.
- `--init-center-mode first` is important. The older all-pose initialization creates a trajectory-shaped canonical cloud, which is bad for deformation-based Wu 4DGS.
- `modal_app.py` patches Wu training in the remote container to use foreground-weighted L1. Without this, black-background object-only video often collapses to a black render.
- `--mask-loss-weight` exists from an attempted alpha/silhouette loss experiment, but the current implementation failed by collapsing to black. Do not use it for the current best recipe.

### What Not To Repeat

- Do not use vanilla Wu loss on black-background object-only videos; the ball disappears.
- Do not use `--foreground-loss-weight 5` for the `0.20m` ball; it collapsed to black.
- Do not use `--foreground-loss-weight 10` for the `0.20m` ball; it recovered only some views and had median foreground ratio `0.0`.
- Do not use the `fg20 + den4k` version as the best result; it is visible but overgrown/fuzzy, with mean/median foreground ratio `1.880/1.808`.
- Do not use the current `--mask-loss-weight 5` alpha-mask branch; it collapsed to black.

### Next Reasonable Fitting Experiments

Start from the current best recipe and change only one thing at a time:

- `--compactness-loss-weight 0.01` to discourage the canonical Gaussian cloud from stretching into a tube.
- `--wu-densify-until-iter 1500` to reduce remaining fuzz.
- `--foreground-loss-weight 18` to slightly reduce over-coverage while keeping visibility.
- Add a non-cheating weak geometry regularizer:
  - Gaussian scale anisotropy penalty, so individual splats do not become long ellipsoids.
  - Canonical compactness penalty, so the object does not smear into a trajectory-like cloud.
  - Optional radial consistency penalty if we decide a ball prior is acceptable.

First compactness ablation command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --compactness-loss-weight 0.01 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Compactness ablation result:

- Run: `wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty`
- Local output: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty`
- Point count: `37489`.
- Rendered train views: `732`.
- Foreground area ratio mean/median: `1.300/1.218`.
- Baseline ratio mean/median for comparison: `1.386/1.264`.
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer/index.html`
- Verdict: `usable` and likely better than the prior baseline by foreground-size metric. The ball is still not a perfect sphere, but compactness reduced over-coverage without making the object disappear.

## Runs

### `wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg20_bounds0p8_dropempty`

Command shape:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg20_bounds0p8_dropempty \
  --iterations 5000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --render
```

Result:

- First non-black Wu result across all 12 cameras and all 61 timestamps.
- Still very fuzzy and inflated.
- Render foreground area was about `1.91x` GT foreground area on average.
- Local output: `dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg20_bounds0p8_dropempty`

### `wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_bounds0p8_dropempty`

Command shape:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_bounds0p8_dropempty \
  --iterations 5000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Better than the path-shaped init. The initialization is a single canonical ball instead of a trajectory tube.
- Still too furry/inflated, but less bad.
- Render foreground area was about `1.73x` GT foreground area on average.
- Wu artifacts:
  - `point_cloud/iteration_5000/point_cloud.ply`
  - `point_cloud/iteration_5000/deformation.pth`
  - `chkpnt_fine_5000.pth`
- Viewer: `.../viewer/index.html`
- QA sheet: `.../qa_render_gt_selected.png`

### `wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg10_init10k_first_surface85_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg10_init10k_first_surface85_bounds0p8_dropempty \
  --iterations 10000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 10 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Cleanly trained and rendered all 732 views: 12 cameras x 61 synchronized timestamps.
- Less overgrown than fg20, but still too fuzzy/furry for the target.
- Render foreground area was about `1.59x` GT foreground area on average.
- Final point count: `66860`.
- Viewer: `.../viewer/index.html`
- QA sheet: `.../qa_render_gt_selected.png`

### `wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 5 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Former best Wu run before limiting densification.
- Still visibly fuzzy, but much less overgrown than fg20/fg10.
- Render foreground area was about `1.38x` GT foreground area on average.
- Final point count: `51050`.
- Viewer: `.../viewer/index.html`
- QA sheet: `.../qa_render_gt_selected.png`

### `wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 5 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Current best Wu run as of this log.
- Tracks the ball across all 12 cameras and all 61 synchronized timestamps.
- Still has a soft/fuzzy edge, but the silhouette is much less overgrown.
- Render foreground area was about `1.16x` GT foreground area on average, down from `1.38x` for the same run without the densification/opacity-reset changes.
- Final point count: `45339`, down from `51050`.
- Local output: `dataset/outputs/wu_debug_ball_area2x_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_area2x_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`
- PLY: `.../point_cloud/iteration_8000/point_cloud.ply`
- Deformation: `.../point_cloud/iteration_8000/deformation.pth`
- Checkpoint: `.../chkpnt_fine_8000.pth`
- Viewer: `.../viewer/index.html`
- QA sheet: `.../qa_render_gt_selected.png`

### Smaller-ball recovery sweep, 2026-05-29

Goal: keep the current best Wu recipe but reduce the debug ball size below the working `0.25m` radius.

Common generation shape:

```bash
phys_sim/bin/python dataset/export_ping_pong_12view.py \
  --variation-set single \
  --output-dir <output_dir> \
  --restitution 0.90 \
  --ball-angle-deg 0.0 \
  --ball-radius-m <radius> \
  --environment room \
  --video-fps 120 \
  --sim-hz 480 \
  --duration-sec 0.5 \
  --render-camera-names all \
  --video-camera-names all
```

Common Wu training shape:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  <scene_dir> \
  <run_name> \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 5 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

#### `wu_debug_ball_r0p12_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Result:

- Radius: `0.12m`.
- Point count: `10000`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `0.000`, median `0.000`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p12_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p12_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p12_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p12_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed`. The render collapsed to black.

#### `wu_debug_ball_r0p16_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Result:

- Radius: `0.16m`.
- Point count: `10000`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `0.000`, median `0.000`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p16_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p16_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p16_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p16_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed`. The render collapsed to black.

#### `wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Result:

- Radius: `0.20m`.
- Point count: `10000`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `0.000`, median `0.000`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed`. The render collapsed to black.

#### `wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Radius: `0.20m`.
- Point count: `198475`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `1.880`, median `1.808`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed`. It recovered visibility, but the result is overgrown/fuzzy and misses the target area ratio of about `1.0-1.25`.

#### `wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_alphamask5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_alphamask5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 5 \
  --mask-loss-weight 5 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Radius: `0.20m`.
- Point count: `10195`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `0.000`, median `0.000`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_alphamask5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg5_alphamask5_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed`. The attempted alpha/silhouette loss collapsed to a black render.

#### `wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg10_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg10_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 10 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Radius: `0.20m`.
- Point count: `25431`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `0.607`, median `0.000`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg10_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg10_init10k_first_surface85_den4k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed`. It recovered one selected camera row but still disappeared in most views.

#### `wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Radius: `0.20m`.
- Point count: `30086`.
- Temporal filtering kept all 61 timestamps; all 12 cameras passed visibility.
- Render foreground / GT foreground area ratio: mean `1.386`, median `1.264`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `best`. This is the best smaller-ball Wu fit so far: visible across selected cameras/timestamps, much less overgrown than the `den4k` run, and close enough to the target foreground area range to use as the current recovery baseline.

### Compactness Regularizer Ablation

We added an optional canonical compactness regularizer to the Wu training patch:

```text
L = L_fg_weighted_rgb + compactness_weight * mean(||xyz - mean(xyz)||^2)
```

This is not a sphere initializer. It is a weak regularizer on the canonical Gaussian cloud that discourages the object from spreading into a long fuzzy blob. These runs use the same successful `0.20m` short-debug scene and the same spatial+temporal segmentation setup as the current recovery baseline:

- Scene: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0`
- Spatial masks: object-only masks under `masks/`
- Temporal filtering: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`
- Views/timestamps: all 12 cameras, 61 synchronized timestamps
- Base recipe: `fg20`, `init10k`, `first` center, `surface_ratio=0.85`, `bounds=0.8`, `densify_until=2000`, no opacity reset, `1000/8000` coarse/fine iterations

#### Compactness `0.01`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --compactness-loss-weight 0.01 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `37489`.
- Render foreground / GT foreground area ratio: mean `1.303`, median `1.218`.
- BBox area ratio: mean `2.054`, median `1.874`.
- Abs aspect error: mean `0.132`, median `0.116`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `usable`. Improves area and bbox spread over the no-compactness baseline, but the silhouette is still slightly elongated.

#### Compactness `0.02`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact2e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --compactness-loss-weight 0.02 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `29805`.
- Render foreground / GT foreground area ratio: mean `1.420`, median `1.336`.
- BBox area ratio: mean `2.666`, median `2.366`.
- Abs aspect error: mean `0.094`, median `0.074`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact2e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact2e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `failed/over-regularized`. Aspect is rounder numerically, but the silhouette gets too large again.

#### Compactness `0.015`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --compactness-loss-weight 0.015 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `36958`.
- Render foreground / GT foreground area ratio: mean `1.270`, median `1.235`.
- BBox area ratio: mean `2.102`, median `1.970`.
- Abs aspect error: mean `0.141`, median `0.133`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `best compactness balance so far`. It has the best foreground area match among the compactness runs and is visually presentable, though not perfectly spherical. Use this as the current compactness baseline/ablation comparison point.

### Scale-Isotropy Regularizer Ablation

We added a second optional regularizer that discourages individual Gaussians from becoming highly stretched:

```text
L = L_fg_weighted_rgb
  + compactness_weight * mean(||xyz - mean(xyz)||^2)
  + scale_isotropy_weight * mean((log(scale) - mean(log(scale)))^2)
```

This is also not a sphere initializer and does not use the known ball radius. It is a generic Gaussian-shape prior: each Gaussian is discouraged from becoming a needle or pancake, but the reconstruction still has to discover the object silhouette from the multi-view masked images.

#### Compactness `0.015` + Scale Isotropy `0.01`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_scaleiso1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --compactness-loss-weight 0.015 \
  --scale-isotropy-loss-weight 0.01 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `31558`.
- Render foreground / GT foreground area ratio: mean `1.290`, median `1.238`.
- BBox area ratio: mean `2.078`, median `1.968`.
- Abs aspect error: mean `0.159`, median `0.161`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_scaleiso1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_scaleiso1e-2_init10k_first_surface85_den2k_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `usable but not an improvement`. The area and bbox spread are close to the compactness-only run, but aspect error is worse. Do not replace the compactness-only baseline with this setting.

### Densification Cutoff Ablation

After scale isotropy did not improve the visible silhouette, we kept the same non-cheating recipe and changed only the optimization schedule: stop Wu densification earlier. This tests whether the elongated/halo shape is caused by late-added Gaussians spreading around the object.

#### Compactness `0.015` + Densify Until `1500`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --compactness-loss-weight 0.015 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `18024`.
- Render foreground / GT foreground area ratio: mean `1.363`, median `1.236`.
- BBox area ratio: mean `1.946`, median `1.712`.
- Abs aspect error: mean `0.080`, median `0.062`.
- Zero-render frames: `4/732`, all from camera `5`, timestamps `31-34`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_fg20_compact1p5e-2_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty/qa_render_gt_selected.png`
- Verdict: `current best shape-fit baseline`. It is much rounder by bbox/aspect metrics and visually cleaner than the den2k compactness baseline. Caveat: it briefly disappears in one training camera, so the next ablation should try to preserve this roundness while recovering those four frames.

## Native Aspect + Silhouette Roundness Recovery

We found a major source of the apparent elongation: Wu's Blender/D-NeRF loader was hardcoded to resize all input images to `800x800`:

```python
image = PILtoTorch(image, (800, 800))
```

Our Phys4D frames are `960x544`, so Wu was training and rendering against a vertically warped target. The exported DyNeRF frames and masks were already native-resolution and mostly round; the distortion happened inside Wu's loader. `modal_app.py` now patches Wu's loader to preserve the native image aspect/resolution:

```python
image = PILtoTorch(image, None)
```

This is not a sphere prior. It only prevents accidental image warping.

### Stable Native-Aspect Baseline: Roundness `0.02`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --silhouette-roundness-loss-weight 0.02 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `16736`.
- Render/GT resolution: `960x544` for both, confirming the aspect patch worked.
- Moment aspect ratio: mean `1.098`, median `1.060`, p90 `1.228`.
- Largest-component bbox aspect at threshold `60`: mean `1.086`, median `1.040`, p90 `1.215`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty/qa_nativeaspect_roundness_selected.png`
- Verdict: `current stable best`. This is the first Wu run that hits the approximate `1.1` aspect-ratio target without relying on a sphere initializer or known radius.

### Longer Fit: Native Aspect Roundness `0.02`, 15k Iterations

This run keeps the same passable data/regularization recipe and only extends fine optimization from `8000` to `15000` iterations. This was tested because the 8k result was structurally acceptable but slightly blurry/fuzzy.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --silhouette-roundness-loss-weight 0.02 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `17003`.
- Final logged fine-training PSNR: about `34.33`.
- Render FPS on Modal: about `14.65`.
- Moment aspect ratio: mean `1.064`, median `1.029`, p90 `1.175`.
- Largest-component bbox aspect: mean `1.229`, median `1.169`, p90 `1.470`.
- Render/GT foreground area ratio: mean `1.377`, median `1.282`, p90 `1.707`.
- Missing/zero-render frames: `0/732`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/viewer/index.html`
- QA comparison sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/compare_round0p02_8k_15k.png`
- PLY: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/wu4dgs_wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Checkpoint: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/wu4dgs_wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/chkpnt_fine_15000.pth`
- Verdict: `best current passable baseline`. Compared with the 8k stable run, 15k reduces visible halo/fuzz, improves moment aspect from `1.097/1.061/1.225` to `1.064/1.029/1.175`, and reduces area overgrowth from `1.513/1.442/1.950` to `1.377/1.282/1.707` (mean/median/p90). It still has late-frame tail artifacts, but it is the better version for baseline comparison and ablation unless a later run specifically fixes those tails.

### Higher Table / Higher Drop Ablation: Table `z=0.40`, Drop `0.80m`, 15k

This tested the hypothesis that a higher visible support surface and a higher drop could improve the Wu ball-bounce fit by giving the object better framing/contact context. It uses the same native-aspect, spatial+temporal segmentation, 12-view, 15k fitting recipe as the current baseline, but changes the PyBullet scene setup.

Data generation command:

```bash
phys_sim/bin/python dataset/export_ping_pong_12view.py \
  --variation-set single \
  --output-dir dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s \
  --restitution 0.90 \
  --ball-angle-deg 0.0 \
  --ball-radius-m 0.20 \
  --table-top-z 0.40 \
  --drop-height-above-surface-m 0.80 \
  --video-fps 120 \
  --sim-hz 480 \
  --duration-sec 0.5 \
  --environment room
```

Training command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 75 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --silhouette-roundness-loss-weight 0.02 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Source data: 61 timestamps x 12 cameras = `732` train views. No frames were dropped by the temporal visibility filter.
- Mask visibility sanity check: every sampled timestamp had `12/12` cameras above `50` mask pixels.
- Point count after fitting: `130720`.
- Render/GT foreground area ratio: mean `6.729`, median `4.954`, p90 `10.558`.
- Missing/zero-render frames: `0/732`.
- Viewer: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/qa_table0p40_h0p80_selected.png`
- PLY: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/wu4dgs_wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Verdict: `failed ablation`. The object remains visible, but the fit massively over-splats into blue sheets/haze. This is much worse than the original short-run 15k baseline (`1.377/1.282/1.707` mean/median/p90 foreground ratio), so the regression is not caused by 15k optimization alone. Do not use the table `z=0.40`, drop `0.80m` setup as the next baseline without additional spill/opacity controls.

#### No-Densify Counter-Test On Higher Table / Higher Drop

This tested whether the failure above was mainly caused by densification exploding from 10k initial points to 130k+ Gaussians. It keeps the same higher-table/higher-drop data but disables meaningful densification and adds explicit anti-spill/area/scale controls.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0 \
  wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 75 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 2 \
  --area-loss-weight 1 \
  --silhouette-roundness-loss-weight 0.02 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --max-scale-loss-weight 100 \
  --max-gaussian-scale 0.008 \
  --wu-densify-until-iter 1 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Point count: `10000` throughout fitting.
- Final logged fine-training PSNR sample: about `29.76`.
- Render FPS on Modal: about `16.64`.
- Render/GT foreground area ratio: mean `1.614`, median `1.672`, p90 `2.377`.
- Missing/zero-render frames: `92/732`.
- Viewer: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/qa_nodensify_spill_area_selected.png`
- PLY: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/wu4dgs_wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Verdict: `usable diagnostic, not best`. Disabling densification fixed the huge blue-sheet failure and reduced overgrowth from `6.73x` to `1.61x`, so densification was a major failure source for this altered scene. However, the model now has 92 missing frames and is still blurrier/hazier than the original `r0p20` 15k baseline. Keep this as evidence that table/drop changes need a different densification schedule, but do not replace the current baseline.

#### Extra Table0p40 Fitting Ablations

After the no-densify counter-test looked passable but still hazy, two smaller fitting-only variants were tested on the same `table0p40_h0p80_r0p20` data. The data stayed unchanged: native `960x544`, all 12 views, object-only masks, `--drop-invisible-frames`, `--min-visible-cameras 6`, and `--min-mask-pixels 50`.

1. `init20k`, no densify, same cap `0.008`.

   - Run: `wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init20k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k`
   - Exact change from the usable diagnostic: `--init-points 20000`.
   - Point count: `20000`.
   - Render/GT foreground area ratio: mean `0.324`, median `0.000`, p90 `0.238`.
   - Missing/zero-render frames: `623/732`.
   - QA sheet: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale100_cap0p008_scaleiso5e-2_cloudiso1e-1_init20k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/qa_init20k_nodensify_selected.png`
   - Verdict: `failed`. More initial points did not improve coverage; it caused most frames to disappear.

2. Tighter max-scale cap `0.006`, no densify, 10k init.

   - Run: `wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale150_cap0p006_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k`
   - Exact changes from the usable diagnostic: `--max-gaussian-scale 0.006`, `--max-scale-loss-weight 150`.
   - Point count: `10000`.
   - Final logged fine-training PSNR sample: about `24.85`.
   - Render/GT foreground area ratio: mean `1.409`, median `0.000`, p90 `3.332`.
   - Missing/zero-render frames: `614/732`.
   - QA sheet: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale150_cap0p006_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/qa_cap0p006_nodensify_selected.png`
   - PLY: `dataset/outputs/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale150_cap0p006_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/wu4dgs_wu_debug_ball_table0p40_h0p80_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_spill2_area1_maxscale150_cap0p006_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_nodensify_noreset_bounds0p8_dropempty_iter15k/point_cloud/iteration_15000/point_cloud.ply`
   - Verdict: `failed`. The tighter scale cap suppresses the visible cloud too aggressively. The mean area ratio is misleading because most frames are black and a few frames over-render.

Takeaway: for the higher-table/higher-drop data, the best fit remains the 10k no-densify cap-`0.008` diagnostic run. Making the cloud denser, adding more initial points, or tightening max scale all made visibility worse. If this data variant is revisited, the next sensible direction is not "more points"; it is a gentler opacity/visibility schedule that preserves the 10k no-densify coverage while reducing haze.

### Stronger Roundness: `0.08`

Command was identical to the stable native-aspect baseline except:

```bash
--silhouette-roundness-loss-weight 0.08
```

Result:

- Point count: `18372`.
- Moment aspect ratio: mean `1.104`, median `1.032`, p75 `1.099`, p90 `1.365`.
- Viewer: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p08_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty/viewer/index.html`
- QA comparison sheet: `dataset/outputs/wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0/4dgs_wu/compare_round0p02_round0p08.png`
- Verdict: `not the default`. It makes the central/core silhouette rounder in many frames, but it creates worse tail artifacts near the end of the sequence. Keep it as an ablation, but use `round0p02` as the stable baseline.

Important nuance: some GT/object-only frames also have high bbox aspect, especially low/oblique camera views near table contact. Those are data/projection/visibility cases, so use aggregate metrics and QA sheets rather than judging only the worst single frame.

## Notes

- Foreground loss solves black collapse but can over-inflate the object if weighted too high.
- The old `--init-points` behavior sampled along every object pose, which creates a trajectory-shaped canonical cloud. This is probably wrong for deformation-based Wu 4DGS.
- Added init controls:
  - `--init-center-mode all|first|middle`
  - `--init-surface-ratio X`
- For the `0.20m` ball, foreground weight below `20` is too weak: `fg10` disappears in many views and `fg5` collapses to black.
- Added Wu config controls:
  - `--wu-densify-until-iter N`
  - `--wu-opacity-reset-interval N`
- For the older `0.25m` area2x scene, stopping densification at iteration 4000 and setting opacity reset to a very large interval reduced overgrowth to about `1.16x`.
- The `0.12m` and `0.16m` smaller-ball retries failed with the old `fg5` recipe by collapsing to black. A `0.20m` retry also collapsed to black at `fg5`.
- Raising foreground loss to `20` recovered the `0.20m` ball. The full `den4k` version overgrew it to about `1.88x` GT area, while `den2k` reduced that to mean/median `1.386/1.264`.
- The old alpha/silhouette branch failed by collapsing to black before the native-aspect fix. After the native-aspect patch, `--mask-loss-weight 1.0` plus weak silhouette roundness is usable.
- Weak compactness helps. `0.015` is the current best balance by area match (`1.270/1.235`). `0.02` made the aspect numerically rounder but overgrew the silhouette, so do not simply keep increasing this weight.
- Scale isotropy at `0.01` did not help on top of compactness `0.015`; it reduced point count but did not make the silhouette more spherical.
- Earlier densification cutoff helped. `den1500` with compactness `0.015` was the best pre-native-aspect shape-fit run: bbox area ratio `1.946/1.712`, aspect error `0.080/0.062`, and only four missing render frames from camera 5.
- Current best passable baseline is the native-aspect `round0p02` 15k run: moment aspect mean/median/p90 `1.064/1.029/1.175`, foreground area ratio mean/median/p90 `1.377/1.282/1.707`, and no missing render frames.
- Next experiment should start from the native-aspect `round0p02` 15k recipe and tune only one stabilizer at a time: slightly lower roundness, stricter tail/artifact suppression, or foreground/mask weights. Do not go back to square-resized Wu inputs.

Recommended next smaller-ball command shape:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  <smaller_ball_scene_dir> \
  <smaller_ball_run_name> \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 45 \
  --bounds 0.8 \
  --wu-densify-until-iter 2000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --silhouette-roundness-loss-weight 0.02 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

## Stacking Wu 4DGS: Separate Object Fit

Goal: fit the existing stacking video without changing the stacking PyBullet data. For this scenario we use object-specific masks, preserve the original frame timeline, and apply temporal filtering only at export time so weak/blank observations do not dominate training. The current target is `block_00` from:

`dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room`

The stacking data has 12 cameras, `960x544` native aspect, 60 FPS, 301 frames, and per-object masks:

- `masks_block_00`
- `masks_block_01`
- `masks_block_02`
- `masks_table`

For stacking, do not use ball-specific sphere/roundness/cloud-isotropy regularizers. The fit should stay box/object general.

### Export/Fit Setup

Working export settings:

```bash
--mask-subdir masks_block_00
--drop-invisible-frames
--min-visible-cameras 6
--min-mask-pixels 50
--all-train
```

Implementation note: `export_object_only_dynerf.py` now initializes stacking blocks from `object_poses.csv` and the box size in `config.json`, rather than using only mask backprojection. This gives a much better box-shaped `fused.ply` for the selected object while still training from the masked RGB views.

### Failed/Diagnostic Runs

Initial runs with raw foreground weighting either collapsed to black or produced massive red overgrowth:

- `stacking_big_block_00_wu_boxinit_fg10_mask1_iter8k`: mostly black.
- `stacking_big_block_00_wu_boxinit_fg80_mask10_iter5k`: mostly black with full-frame artifacts.
- `stacking_big_block_00_wu_boxinit_fg1000_iter3k`: visible object, but huge overgrowth.

Normalized foreground/background loss improved visibility but produced red smear/fan artifacts. The key diagnosis was that giant Gaussian scale outliers were part of the problem:

- `stacking_big_block_00_wu_boxinit_normfg2_mask5_scaleiso5e-2_den1000_iter3k`
  - Point count: `13389`.
  - Render/GT foreground area ratio: mean `60.50`, median `32.50`, p90 `70.35`.
  - Max Gaussian scale p99/max: about `1.87m / 52.54m`.
  - Verdict: failed. Shows giant scale outlier artifact.

Adding max-scale regularization and disabling densification controlled those huge splats:

- `stacking_big_block_00_wu_boxinit_normfg2_mask20_maxscale50_cap0p01_nodensify_iter3k`
  - Point count: `12000`.
  - Max Gaussian scale median/p90/p95/p99/max: `0.0021 / 0.0068 / 0.0087 / 0.0100 / 0.0125m`.
  - Render/GT foreground area ratio: mean `85.81`, median `42.03`, p90 `81.68`.
  - Verdict: failed, but useful. Giant scale was fixed; remaining problem is red opacity/RGB spill.

The RGB-spill experiment over-regularized the render and erased many views:

- `stacking_big_block_00_wu_boxinit_normfg5_mask0p1_rgbspill_maxscale50_cap0p01_nodensify_iter3k`
  - Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room \
  stacking_big_block_00_wu_boxinit_normfg5_mask0p1_rgbspill_maxscale50_cap0p01_nodensify_iter3k \
  --mask-subdir masks_block_00 \
  --iterations 3000 \
  --coarse-iterations 1000 \
  --time-resolution 75 \
  --bounds 1.6 \
  --wu-densify-until-iter 1 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 5 \
  --mask-loss-weight 0.1 \
  --scale-isotropy-loss-weight 0.05 \
  --max-scale-loss-weight 50 \
  --max-gaussian-scale 0.01 \
  --init-points 12000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

  - Point count: `12000`.
  - Max Gaussian scale median/p90/p95/p99/max: `0.0023 / 0.0084 / 0.0100 / 0.0114 / 0.0895m`.
  - Opacity median/p90/p99: `0.633 / 0.993 / 1.000`.
  - Render/GT foreground area ratio: mean `1232.60`, median `0.00`, p90 `1108.79`.
  - Zero render frames: `2754/3612`.
  - Viewer: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_normfg5_mask0p1_rgbspill_maxscale50_cap0p01_nodensify_iter3k/viewer/index.html`
  - QA sheet: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_normfg5_mask0p1_rgbspill_maxscale50_cap0p01_nodensify_iter3k/qa_render_gt_selected.png`
  - Verdict: failed. Scale is mostly controlled, but many frames collapse and faint full-frame spill remains. Next runs should use gentler spill/mask terms and try `--init-center-mode middle` to reduce deformation burden.

### Latest Stacking Attempts: Block 00 Surface-Only Initialization

These runs keep the stacking source data unchanged and only adjust the fit. They use the same per-object spatial mask and temporal filtering:

- Source scene: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room`
- Object mask: `masks_block_00`
- Temporal/spatial filter: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`
- Export result: 301 synchronized timestamps, 3612 train views, all 12 cameras. Original timestamps are preserved.

#### `stacking_big_block_00_wu_boxinit_surface3k_fg80_bg5_area5_tr16_maxscale1000_cap0p006_nodensify_iter4k`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room \
  stacking_big_block_00_wu_boxinit_surface3k_fg80_bg5_area5_tr16_maxscale1000_cap0p006_nodensify_iter4k \
  --mask-subdir masks_block_00 \
  --iterations 4000 \
  --coarse-iterations 1000 \
  --time-resolution 16 \
  --bounds 1.6 \
  --wu-densify-until-iter 1 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 80 \
  --bg-spill-loss-weight 5 \
  --area-loss-weight 5 \
  --scale-isotropy-loss-weight 0.05 \
  --max-scale-loss-weight 1000 \
  --max-gaussian-scale 0.006 \
  --init-points 3000 \
  --init-center-mode middle \
  --init-surface-ratio 1.0 \
  --render
```

Result:

- Point count: `3000`.
- Max Gaussian scale median/p90/p95/p99/max: `0.0057 / 0.0060 / 0.0060 / 0.0075 / 0.0245m`.
- Opacity median/p90/p99: `0.985 / 1.000 / 1.000`.
- Render/GT foreground area ratio: mean `53.00`, median `0.00`, p90 `203.21`.
- Zero render frames: `2776/3612`.
- Viewer: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_surface3k_fg80_bg5_area5_tr16_maxscale1000_cap0p006_nodensify_iter4k/viewer/index.html`
- QA sheet: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_surface3k_fg80_bg5_area5_tr16_maxscale1000_cap0p006_nodensify_iter4k/qa_render_gt_selected.png`
- Verdict: failed, but current best diagnostic for stacking block 00. It is visibly closer than the 12k dense attempts, but still has dim red sheets/haze and too many missing frames.

#### `stacking_big_block_00_wu_boxinit_surface1k_fg80_bg10_area10_tr16_maxscale2000_cap0p004_nodensify_iter4k`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room \
  stacking_big_block_00_wu_boxinit_surface1k_fg80_bg10_area10_tr16_maxscale2000_cap0p004_nodensify_iter4k \
  --mask-subdir masks_block_00 \
  --iterations 4000 \
  --coarse-iterations 1000 \
  --time-resolution 16 \
  --bounds 1.6 \
  --wu-densify-until-iter 1 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 80 \
  --bg-spill-loss-weight 10 \
  --area-loss-weight 10 \
  --scale-isotropy-loss-weight 0.05 \
  --max-scale-loss-weight 2000 \
  --max-gaussian-scale 0.004 \
  --init-points 1000 \
  --init-center-mode middle \
  --init-surface-ratio 1.0 \
  --render
```

Result:

- Point count: `1000`.
- Max Gaussian scale median/p90/p95/p99/max: `0.0040 / 0.0040 / 0.0041 / 0.0044 / 0.0049m`.
- Opacity median/p90/p99: `0.999 / 1.000 / 1.000`.
- Render/GT foreground area ratio: mean `1356.60`, median `203.21`, p90 `1108.79`.
- Zero render frames: `1368/3612`.
- Viewer: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_surface1k_fg80_bg10_area10_tr16_maxscale2000_cap0p004_nodensify_iter4k/viewer/index.html`
- QA sheet: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_surface1k_fg80_bg10_area10_tr16_maxscale2000_cap0p004_nodensify_iter4k/qa_render_gt_selected.png`
- Verdict: failed. The scale clamp works, but the model compensates with nearly opaque Gaussians and produces huge foreground spill. Fewer points did not solve the stacking artifact.

Next direction: keep the stacking data unchanged, but do not simply reduce point count further. The likely issue is that the Wu deformation/opacity fit is explaining a moving box with persistent high-opacity splats. Try either `--init-center-mode all` with moderate surface points to initialize along the full motion path, or lower temporal resolution / stronger background spill with less opaque initialization. Avoid sphere-specific regularizers for stacking.

#### `stacking_big_block_00_wu_boxinit_all3k_fg80_bg5_area5_tr16_maxscale1500_cap0p004_nodensify_iter4k`

This tested trajectory-aware initialization: 3000 box surface points distributed across all 301 object poses, while preserving the exact same 12-view masked dataset and temporal filter.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room \
  stacking_big_block_00_wu_boxinit_all3k_fg80_bg5_area5_tr16_maxscale1500_cap0p004_nodensify_iter4k \
  --mask-subdir masks_block_00 \
  --iterations 4000 \
  --coarse-iterations 1000 \
  --time-resolution 16 \
  --bounds 1.6 \
  --wu-densify-until-iter 1 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --foreground-loss-weight 80 \
  --bg-spill-loss-weight 5 \
  --area-loss-weight 5 \
  --scale-isotropy-loss-weight 0.05 \
  --max-scale-loss-weight 1500 \
  --max-gaussian-scale 0.004 \
  --init-points 3000 \
  --init-center-mode all \
  --init-surface-ratio 1.0 \
  --render
```

Result:

- Point count: `3000`.
- Max Gaussian scale median/p90/p95/p99/max: `0.0040 / 0.0040 / 0.0040 / 0.0044 / 0.0998m`.
- Opacity median/p90/p99: `0.9999 / 1.0000 / 1.0000`.
- Render/GT foreground area ratio: mean `1278.81`, median `0.00`, p90 `1108.79`.
- Zero render frames: `2350/3612`.
- Viewer: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_all3k_fg80_bg5_area5_tr16_maxscale1500_cap0p004_nodensify_iter4k/viewer/index.html`
- QA sheet: `dataset/outputs/room_physics_12view_stacking_big/scene_0000_stacking_room/4dgs_wu/stacking_big_block_00_wu_boxinit_all3k_fg80_bg5_area5_tr16_maxscale1500_cap0p004_nodensify_iter4k/qa_render_gt_selected.png`
- Verdict: failed. All-pose initialization reduced zero-render frames compared with the surface3k middle run, but it made foreground spill dramatically worse and produced a max-scale outlier. Do not use trajectory-shaped canonical initialization as the next default for stacking.

Updated next direction: the sticking point is not simply point count or center choice. The stacked block is being fit with high-opacity persistent splats, and Wu's deformation field is not cleanly reconciling the moving masked object with black background supervision. Next stacking attempt should avoid all-pose canonical init and instead test a more direct visibility/opacity constraint, such as pruning or freezing low-support Gaussians after initialization, or adding a mask-space loss that uses the actual binary masks rather than inferring foreground from black-background RGB.

## Q&A Notes: Gaussian Fitting Approach

These are the notes to use if asked how the Gaussian fitting pipeline works, especially for the Wu 4DGS ball-bounce experiments.

### What We Fit

We fit an object-only 4D Gaussian representation from synthetic multi-view PyBullet videos. The input is a synchronized 12-camera video sequence with known cameras, RGB frames, and object masks. The model learns a canonical Gaussian cloud plus a time-dependent deformation field, so the same learned object can be rendered at different timestamps and viewpoints.

For the current Wu ball-bounce baseline, the source data is a short 120 FPS object-only clip with:

- 12 camera views.
- Spatial segmentation masks for the ball.
- Temporal visibility filtering.
- Black background outside the object mask.
- A low floor-like platform rather than a normal table. In the Wu ball-bounce setup, the platform top is around `z = 0.06m`, with thickness `0.06m` and size around `5.2m x 5.2m`.

### Why Spatial And Temporal Segmentation Matter

Raw 4DGS struggled on the bouncing ball because the object is small, fast-moving, and occupies only a small fraction of the pixels. If we train on full frames, the reconstruction loss is dominated by background. If we train on object-only masks but keep frames where the object is barely visible or absent, the model can learn that rendering nothing is acceptable.

Our fix is:

- Spatial segmentation: use the object mask to isolate the foreground object and remove the background from supervision.
- Temporal segmentation: drop timestamps where the object is not visible enough across the camera rig.
- Multi-view consistency: keep all 12 views for each retained timestamp, so the fit still gets multi-view geometry rather than a single-view silhouette.

The main visibility rule used in the stable Wu experiments is:

- `--drop-invisible-frames`
- `--min-visible-cameras 6`
- `--min-mask-pixels 50`

This means a timestamp is kept only when at least 6 cameras see the object with at least 50 mask pixels. This is not changing the physics; it removes frames that provide weak or misleading object supervision.

### Optimization Objective

At a high level, the fitting loss combines:

- Photometric reconstruction: render the Gaussians and match the masked RGB frames.
- Foreground/object emphasis: weight object pixels more strongly so the small moving object is not overwhelmed by black/background pixels.
- Background spill control: discourage Gaussians from explaining pixels outside the mask.
- Shape/compactness regularization: discourage degenerate elongated Gaussian clouds.

The compactness/roundness regularizer is a weak geometric prior. It is not a sphere initializer and does not use the known true ball radius. The goal is to prevent the object-only Gaussian cloud from becoming a long streak or halo while still letting image evidence drive the fit.

### Why The Ball Became Elongated

A small moving sphere can be under-constrained by sparse multi-view silhouettes, especially over a short clip. Many elongated or anisotropic Gaussian arrangements can project to roughly the right 2D masks from the training cameras. Without a compactness prior, the optimizer can reduce image loss using stretched Gaussian blobs, opacity halos, or motion streaks instead of a clean sphere-like cloud.

This is why more iterations alone did not necessarily fix elongation. If the loss landscape allows a stretched solution, longer fitting can make the artifact sharper rather than more spherical. The useful improvement came from changing the fitting constraints, not simply training longer.

### Current Baseline Status

The current best Wu ball-bounce run is a passable baseline for comparison and ablation, not a final perfect reconstruction. It is useful because:

- The ball is visible and tracks the motion.
- The object size is presentable rather than huge.
- Spatial and temporal segmentation are both applied.
- The fitting uses all 12 views.
- The compactness/roundness regularization improves the aspect ratio relative to earlier elongated fits.

Known limitations:

- Some blur remains.
- Late-frame tail artifacts can still appear.
- The result is sensitive to mask quality, visibility thresholds, and object scale.
- It should be reported as a baseline/ablation version that we aim to improve, not as a solved final result.

### Short Explanation For Slides Or Q&A

> We found that fitting 4D Gaussians directly to the full videos was dominated by background and blank frames, so we converted the problem into object-only 4DGS fitting. We use segmentation masks spatially to isolate the object, and temporal visibility filtering to remove frames where the object is not sufficiently observed by the camera rig. For the Wu implementation, the main artifact was that the ball became elongated because a sparse set of silhouettes can be explained by stretched Gaussian clouds. We added weak compactness/roundness regularization, without initializing as a sphere or using the true radius, to make the learned cloud less degenerate while preserving image-driven fitting. This gives us a reasonable baseline for ablation and comparison.

## 2026-05-30: 3s Bouncy Ball Recovery Attempt

Goal: create a 3-second PyBullet bounce where the ball is visibly still bouncing, then fit Wu 4DGS with spatial and temporal segmentation. This section is meant to preserve the exact state of the current working direction for a clean-memory handoff.

Important correction: compactness regularization was not the main fix here. The dominant issue was that earlier taller/longer bounces used canonical initialization centered near the first frame, so the learned deformation AABB did not cover the full vertical path. The fix that mattered was `--init-center-mode all`, which builds initialization support from all retained object poses and gives the deformation field bounds over the full bounce.

### Source PyBullet Data

Generated dataset:

`dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`

Generation command:

```bash
phys_sim/bin/python dataset/export_ping_pong_12view.py \
  --variation-set single \
  --output-dir dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy \
  --environment room \
  --video-fps 120 \
  --sim-hz 480 \
  --duration-sec 3.0 \
  --restitution 0.93 \
  --ball-angle-deg 0.0 \
  --ball-radius-m 0.20 \
  --table-top-z 0.40 \
  --drop-height-above-surface-m 0.80 \
  --camera-target-z 0.75 \
  --ball-visual-style marker \
  --render-camera-names front_right,right,back_right,back_left,front_left,top_oblique,low_front_left,low_back_right \
  --video-camera-names all
```

Data notes:

- 3 seconds at 120 FPS.
- Ball radius `0.20m`.
- Table/platform top at `z = 0.40m`.
- Drop height above platform `0.80m`.
- Restitution `0.93`.
- Marker texture enabled to add visible surface cues.
- Training/render camera subset: 8 oblique cameras.
- Full video cameras are still exported.
- Object trajectory spans approximately `z = 0.60m..1.20m`, and the ball remains visibly bouncing over the 3 seconds.

### Segmentation / Temporal Filtering

Spatial segmentation uses the object-only ball masks from the PyBullet export. Temporal segmentation preserves synchronized camera/timestamp structure:

- `--drop-invisible-frames`
- `--min-visible-cameras 6`
- `--min-mask-pixels 50`

The 3-second source scene has no empty masks under this rule. For stride-4 experiments, the retained frame list is:

`dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_stride4.txt`

This contains 91 timestamps: `0, 4, 8, ..., 360`.

### Stable Baseline Family

Best currently usable family:

- Use stride-4 timestamps.
- Use full-trajectory initialization: `--init-center-mode all`.
- Use moderate early stopping.
- Avoid the all-frame runs for now.

Canonical stable command pattern:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  <RUN_NAME> \
  --iterations <3000-or-5000> \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 25 \
  --bg-spill-loss-weight 2 \
  --max-scale-loss-weight 50 \
  --max-gaussian-scale 0.01 \
  --scale-isotropy-loss-weight 5e-2 \
  --wu-densify-until-iter 700 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_stride4.txt \
  --init-points 20000 \
  --init-center-mode all \
  --init-surface-ratio 0.85 \
  --render
```

The learned AABB for the successful stride-4 runs spans the whole bounce:

```text
max approx [0.1999951, 0.3999495, 1.3974172]
min approx [-0.1999396, 0.0000113, 0.4037370]
```

This is the key difference from the broken first-frame-centered fits.

### Run Results

#### Stride-4 5k: Current Usable Baseline Candidate

Run:

`wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter5k`

Artifacts:

- Viewer: `dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter5k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter5k/qa_fg25_bgspill2_iter5k_selected.png`

Metrics:

```json
{
  "num_frames": 728,
  "ratio_mean": 0.9433379894317957,
  "ratio_median": 0.0,
  "ratio_p90": 2.3772220283602654,
  "zero_render_frames": 376,
  "nonzero_gt_frames": 728
}
```

Verdict: usable baseline / ablation candidate. It is not solved, but it is better than the failed longer/all-frame variants. The ball is visible in many frames and tracks motion, with remaining dropout and ghosting.

#### Stride-4 3k: Similar But Not Clearly Better

Run:

`wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter3k`

Artifacts:

- Viewer: `dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter3k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter3k/qa_fg25_bgspill2_iter3k_selected.png`

Metrics:

```json
{
  "num_frames": 728,
  "ratio_mean": 1.1503263718872763,
  "ratio_median": 0.0,
  "ratio_p90": 3.4571603738919907,
  "zero_render_frames": 443,
  "nonzero_gt_frames": 728
}
```

Verdict: usable for comparison, but not better than 5k. Mean foreground area is closer to target, but zero-render frames are worse and the contact sheet shows more ghosted blobs.

#### Stride-4 8k: Worse

Run:

`wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_stride4_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter8k`

Metrics:

```json
{
  "num_frames": 728,
  "ratio_mean": 0.49079148643945,
  "ratio_median": 0.0,
  "ratio_p90": 3.232517892897926,
  "zero_render_frames": 628,
  "nonzero_gt_frames": 728
}
```

Verdict: failed/worse. More training did not help; it over-suppressed the object and produced many blank renders.

#### All-Frame 2200: Worse

Run:

`wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_allframes_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter2200`

Artifacts:

- Viewer: `dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_allframes_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter2200/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_debug_ball_marker_oblique8_table0p40_h0p80_r0p20_120fps_3p0s_e0p93_a0p0_wu_object_allframes_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_noreset_bounds0p8_dropempty_iter2200/qa_allframes_iter2200_selected.png`

Metrics:

```json
{
  "num_frames": 2888,
  "ratio_mean": 0.7329463241081405,
  "ratio_median": 0.0,
  "ratio_p90": 2.3234723213535378,
  "zero_render_frames": 1885,
  "nonzero_gt_frames": 2888
}
```

Verdict: failed/worse. All frames increased temporal coverage but made the fit less stable and more smeary. The 3000 all-frame version hit NaN around fine iteration 2560 and auto-restarted, so it was killed.

### Current Takeaways

- The 3-second PyBullet video is good enough for continued fitting work: it visibly bounces over the full clip.
- The critical Wu fix is full-trajectory initialization: `--init-center-mode all`.
- Stride-4 is currently better than all-frame training for this setup.
- More iterations are not automatically better. 8k was worse than 5k, and all-frame 3k became unstable.
- The current best baseline candidate is stride-4 5k.
- Remaining failure mode is not just object size; it is view/time dropout and ghosted foreground blobs.
- Next best direction is likely better fit stability or loss scheduling, not a generic compactness-first approach.

## 2026-05-30: All-12 Camera 15k Requirement Check

User constraint: stop using reduced camera subsets. The fit must use all 12 cameras, spatial object masks, temporal filtering, and more iterations. For this check, we used the same 3-second bouncy source scene:

`dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`

Source generation settings:

- 12 rendered cameras and 12 video cameras.
- Room setting.
- 120 FPS.
- 3.0 seconds.
- Restitution `0.93`.
- Drop angle `0.0`.
- Ball radius `0.20m`.
- Table top `z=0.40m`.
- Drop height above table `0.80m`.
- Marker-textured ball.

Shared Wu export/training settings:

- `--drop-invisible-frames`
- `--min-visible-cameras 6`
- `--min-mask-pixels 50`
- `--frame-list .../frame_list_stride4.txt`
- `--init-points 10000`
- `--init-center-mode all`
- `--init-surface-ratio 0.85`
- `--bounds 0.8`
- `--time-resolution 120`
- `--coarse-iterations 1000`
- `--iterations 15000`
- `--wu-densify-until-iter 1`
- `--wu-opacity-reset-interval 300000`
- `--render`

Export verification:

```json
{
  "train_cameras": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "num_train_views": 1092,
  "num_train_timestamps": 91,
  "empty_train_masks": 0,
  "below_threshold_train_masks": 0,
  "dropped_train_views": 0,
  "preserves_original_timestamps": true,
  "preserves_synchronized_multiview_frames": true
}
```

### `wu_ball12_3s_e93_fg20_mask1_minimal15k`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_3s_e93_fg20_mask1_minimal15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --wu-densify-until-iter 1 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_stride4.txt \
  --init-points 10000 \
  --init-center-mode all \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Completed full 15k fine iterations.
- Point count stayed at `10000`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_fg20_mask1_minimal15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_fg20_mask1_minimal15k/qa_selected.png`

Metrics:

```json
{
  "num_frames": 1092,
  "ratio_mean": 29.88214600505598,
  "ratio_median": 31.830316328396417,
  "ratio_p90": 47.87591507592608,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 1092
}
```

Verdict: failed. This satisfies all-12 camera and 15k training requirements mechanically, but the render has a large blue background smear. The ball is visible, but foreground area is about 30x the GT mask.

### `wu_ball12_3s_e93_fg20_mask1_bgspill1_minimal15k`

Change from the previous run: added only `--bg-spill-loss-weight 1.0` to suppress RGB leakage outside the object mask.

Result:

- Failed during training.
- Hit `loss is nan,end training, reexecv program now` around fine iteration `2590`.
- Killed after auto-restart.

Verdict: failed/unstable. A strong direct RGB spill penalty is not stable in this setup.

### `wu_ball12_3s_e93_fg20_mask5_minimal15k`

Change from the first run: increased silhouette mask loss from `1.0` to `5.0`, with no RGB spill penalty and no geometry regularizers.

Result:

- Completed full 15k fine iterations.
- Point count stayed at `10000`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_fg20_mask5_minimal15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_fg20_mask5_minimal15k/qa_selected.png`

Metrics:

```json
{
  "num_frames": 1092,
  "ratio_mean": 26.794780462132096,
  "ratio_median": 28.176079980310607,
  "ratio_p90": 47.518698618434385,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 1092
}
```

Verdict: failed, but informative. Increasing mask loss alone is stable and slightly reduces foreground spill, but not enough to make the fit acceptable. The dominant failure mode remains blue background smear, not missing frames or missing cameras.

### Takeaway From All-12 15k Check

- The 12-camera export path is correct.
- Temporal filtering is present and preserves synchronized multiview timestamps.
- More iterations alone do not fix Wu on this scene.
- Mask loss alone does not sufficiently suppress background-colored smear.
- Direct RGB spill loss can suppress the right failure mode in principle, but weight `1.0` is unstable here.
- Next focused retry should use a much smaller spill weight or a scheduled/clamped spill term, while preserving all-12 cameras and 15k iterations.

### `wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k`

Purpose: best current 3-second Wu attempt for the bouncing-ball setup. This run preserves the working spatial+temporal segmentation recipe, uses all 12 cameras, keeps the original synchronized timestamps, and extends the fit to 15k fine iterations.

Dataset:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- Duration: `3.0s`
- FPS: `120`
- Exported train timestamps: `0, 4, 8, ..., 360` (`91` timestamps)
- Cameras: all 12 train cameras `[0..11]`
- Training views: `1092`
- Table top: `0.40m`
- Ball radius: `0.20m`
- Drop height above table: `0.80m`
- Restitution: `0.93`

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --silhouette-roundness-loss-weight 0.02 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_stride4.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Completed full `15000` fine iterations.
- Final point count: `226924`.
- Local folder: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/qa_selected.png`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/wu4dgs_wu_ball12_3s_e93_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/point_cloud/iteration_15000/point_cloud.ply`

Metrics:

```json
{
  "num_frames": 1092,
  "ratio_mean": 3.6394067352355064,
  "ratio_median": 3.3558050548820697,
  "ratio_p90": 4.594095337246503,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 1092
}
```

Verdict: usable 3-second reference, but not final. It is much better than the failed 15k all-init/minimal runs and improves over the 8k dense run (`ratio_mean` dropped from about `4.86` to `3.64`). The ball is visible through the sequence and follows the GT motion, but it is still too fuzzy/large, so the next improvement should target sharper foreground support rather than more basic visibility.

### Anti-Fuzz Follow-Ups After 3s Baseline

These runs were tested after the 3-second 15k baseline because the baseline is visible and temporally coherent but visibly fuzzy. The goal was to reduce blue haze/spill without losing the ball.

#### `wu_ball12_3s_e93_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_iter5k`

Purpose: aggressively suppress blue background spill and large Gaussians.

Result:

- Final point count: `32364`.
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_iter5k/qa_selected.png`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_fg25_bgspill2_maxscale50_cap0p01_scaleiso5e-2_init20k_all_surface85_den700_iter5k/viewer/index.html`

Metrics:

```json
{
  "ratio_mean": 1.8576,
  "ratio_median": 0.0,
  "ratio_p90": 3.2348,
  "zero_render_frames": 622,
  "nonzero_gt_frames": 1092
}
```

Verdict: failed. The foreground ratio looks lower only because the render disappears in most frames. This setting is too restrictive.

#### `wu_ball12_3s_e93_first_fg20_mask1_maxscale25_cap0p012_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1200_iter8k`

Purpose: preserve the working first-frame initialization and roundness/scale isotropy settings, but add a softer max-scale cap to reduce haze.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_3s_e93_first_fg20_mask1_maxscale25_cap0p012_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1200_iter8k \
  --iterations 8000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --max-scale-loss-weight 25 \
  --max-gaussian-scale 0.012 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --silhouette-roundness-loss-weight 0.02 \
  --wu-densify-until-iter 1200 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_stride4.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Completed `8000` fine iterations.
- Final point count: `45888`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_maxscale25_cap0p012_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1200_iter8k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_maxscale25_cap0p012_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1200_iter8k/qa_selected.png`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_first_fg20_mask1_maxscale25_cap0p012_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1200_iter8k/wu4dgs_wu_ball12_3s_e93_first_fg20_mask1_maxscale25_cap0p012_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1200_iter8k/point_cloud/iteration_8000/point_cloud.ply`

Metrics:

```json
{
  "num_frames": 1092,
  "ratio_mean": 14.581422176370559,
  "ratio_median": 9.777360510986753,
  "ratio_p90": 29.333586289715544,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 1092
}
```

Verdict: failed. The ball stays visible, but the blue haze grows dramatically. This is worse than the current 3-second baseline (`ratio_mean` about `3.64`), so max-scale capping alone is not the fix.

#### `wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k`

Purpose: test whether the fuzzy/halo artifact was caused by temporal subsampling. This run uses the same visible 3-second ball-bounce scene and the same stable fitting recipe as the prior baseline, but removes the stride-4 frame list so Wu trains/renders on the full 120 FPS sequence.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --silhouette-roundness-loss-weight 0.02 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Data used:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `361`
- Cameras: all `12`
- Training views: `4332`
- Spatial segmentation: object-only masks.
- Temporal segmentation: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`. For this scene no frames/views were dropped because the object is visible in all masks.

Result:

- Completed `15000` fine iterations.
- Final point count: `165780`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/qa_selected.png`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/wu4dgs_wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/wu4dgs_wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/point_cloud/iteration_15000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/wu4dgs_wu_ball12_3s_e93_full120_first_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_surface85_den1500_iter15k/chkpnt_fine_15000.pth`

Metrics:

```json
{
  "num_frames": 4332,
  "ratio_mean": 3.2148341584642974,
  "ratio_median": 2.9274762911056014,
  "ratio_p90": 4.24131508959242,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 4332
}
```

Verdict: usable baseline / ablation candidate, but not a final quality fix. Full 120 FPS modestly improves foreground area ratio compared with the stride-4 3-second baseline (`3.21` vs about `3.64`) and preserves all frames/cameras, but the render still has a soft blue halo. This suggests the main issue is not only temporal sampling; the next fitting improvement should target excessive background/halo opacity while preserving visibility.

#### `wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k`

Purpose: test whether the blue halo was caused by blue appearance leakage. This run uses a derived 2-second scene with the same physics, cameras, object poses, and masks, but rewrites RGB frames to a pure white ball on a black background.

Derived scene creation:

```bash
phys_sim/bin/python scripts/filter_object_appearance_scene.py \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0 \
  --duration-s 2.0 \
  --object-color 255,255,255 \
  --background-color 0,0,0
```

Training command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0 \
  wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --scale-isotropy-loss-weight 0.05 \
  --cloud-isotropy-loss-weight 0.1 \
  --silhouette-roundness-loss-weight 0.02 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Data used:

- Source scene: `dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241`
- Cameras: all `12`
- Training views: `2892`
- Spatial segmentation: object-only masks.
- Temporal segmentation: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`. No frames/views were dropped.

Result:

- Completed `15000` fine iterations.
- Final point count: `10337`.
- Viewer: `dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/qa_selected.png`
- PLY: `dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/wu4dgs_wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/wu4dgs_wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/point_cloud/iteration_15000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_white_on_black_all12_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/wu4dgs_wu_ball12_2s_white_black_e93_fg20_mask1_init10k_den1500_iter15k/chkpnt_fine_15000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "ratio_mean": 17.929996561771752,
  "ratio_median": 17.739941432075398,
  "ratio_p90": 24.91264162675944,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: failed. The white-on-black appearance made leakage much worse, producing large bright streak/plane artifacts. This suggests the halo is not simply blue color leakage; Wu is still using low-opacity/background structure to explain the object-only sequence. For future attempts, prefer preserving natural object appearance or using a light gray background plus stronger mask/opacity constraints rather than pure white-on-black binary frames.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k`

Purpose: test whether the current best 2-second natural-blue setup improves simply by fitting longer. This keeps the same data, spatial masks, temporal filtering, 120 FPS, all 12 cameras, and frame-step-1 sampling. The only intended change from the 8k candidate is longer fine optimization.

Training command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 1500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Data used:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241` original timestamps, frames `0..240`
- Cameras: all `12`
- Training views: `2892`
- Spatial segmentation: object-only masks.
- Temporal segmentation: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`. No frames/views were dropped.
- Native image size preserved: `960x544`; do not resize to square, because that was the earlier elongation artifact.

Result:

- Completed `15000` fine iterations.
- Final point count: `119342`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k/point_cloud/iteration_15000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k/chkpnt_fine_15000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 25913.988589211618,
  "render_area_max": 179034,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.8630700531634974,
  "ratio_median": 1.771447722358813,
  "ratio_p90": 2.134673650408658,
  "render_max_mean": 189.8634163208852,
  "render_max_max": 235,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: usable, but not best. It removes blank frames and preserves motion, but the mean foreground area ratio is slightly worse than the 8k area0.10 candidate (`1.86` vs `1.82`) and the ball remains soft/haloed. Longer fitting alone does not solve the blur. Next sensible fit-only experiment: preserve the same data and segmentation, but allow densification longer so the model has enough Gaussian support for a sharper sphere instead of smearing a smaller point set.
#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k`

Purpose: fit-only retry on the current 2-second blue-ball setup, preserving the no-nonsense data recipe: all 12 cameras, native `960x544`, 120 FPS, first 2 seconds, no frame stride, spatial object-only masks, and temporal filtering with `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`. The only intended fitting change from the den1500 baselines is longer densification (`4000`) so the model has more Gaussian support before freezing topology.

Training command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k \
  --iterations 12000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Data/export checks:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241` original timestamps, frames `0..240`
- Cameras: all `12`
- Training views: `2892`
- Temporal segmentation: no frames/views were dropped; `empty_train_masks=0`, `below_threshold_train_masks=0`, `dropped_train_views=0`
- Native size preserved: `960x544`

Result:

- Completed `12000` fine iterations.
- Final point count: `368656`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k/point_cloud/iteration_12000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k/point_cloud/iteration_12000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k/chkpnt_fine_12000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 23508.549100968186,
  "render_area_max": 148779,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.6931832895710857,
  "ratio_median": 1.602387250827276,
  "ratio_p90": 1.9142599892208556,
  "render_max_mean": 197.44640387275243,
  "render_max_max": 255,
  "zero_render_frames": 3,
  "nonzero_gt_frames": 2892
}
```

Verdict: current best candidate on the strict 2-second all-12/all-frame setup. The area ratio improves over both den1500 baselines (`1.69` vs `1.82` for 8k and `1.86` for 15k), and the QA sheet shows less overgrown halo. It still has a few weak frames (`zero_render_frames=3`) and remains somewhat soft, so the next clean fit-only test is to keep densification at `4000` and run longer fine optimization.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k`

Purpose: direct follow-up to the `den4000_iter12k` candidate. Keep the exact strict data recipe and fitting settings, but extend fine optimization from `12000` to `15000` iterations to test whether longer fitting sharpens the object.

Training command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Data/export checks:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241` original timestamps, frames `0..240`
- Cameras: all `12`
- Training views: `2892`
- Temporal segmentation: no frames/views were dropped; `empty_train_masks=0`, `below_threshold_train_masks=0`, `dropped_train_views=0`
- Native size preserved: `960x544`

Result:

- Completed `15000` fine iterations.
- Final point count: `406249`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k/point_cloud/iteration_15000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k/chkpnt_fine_15000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 24647.31500691563,
  "render_area_max": 161261,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.7863516168234694,
  "ratio_median": 1.6670137557810811,
  "ratio_p90": 2.112032262890103,
  "render_max_mean": 193.93222683264176,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: usable, but not best. It removes the `zero_render_frames=3` issue seen in the 12k run, but the foreground area ratio regresses (`1.79` vs `1.69`) and the QA sheet shows more halo/soft spread. Keep `den4000_iter12k` as the current best baseline. Longer fine optimization alone is not the fix; the next improvement should target halo control, opacity/temporal support, or post-fit pruning while preserving the strict all-12/all-frame segmentation recipe.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`

Purpose: test whether substantially longer fine optimization helps the strict 2-second, all-12-camera, all-frame Wu setup after the 15k run looked visually more stable but overgrew in foreground area. This keeps the exact same data, segmentation, initialization, densification, and loss weights as the 12k/15k candidates, changing only fine iterations to `30000`.

Training command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k \
  --iterations 30000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Data/export checks:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241` original timestamps, frames `0..240`
- Cameras: all `12`
- Training views: `2892`
- Temporal segmentation: no frames/views were dropped; source frame list preserves synchronized first-2-second timestamps
- Native size preserved: `960x544`

Result:

- Completed `30000` fine iterations.
- Final point count: `376792`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/chkpnt_fine_30000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 20419.784232365146,
  "render_area_max": 151895,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.4732886206598925,
  "ratio_median": 1.4242730199112208,
  "ratio_p90": 1.6240718126722407,
  "render_max_mean": 195.87793914246197,
  "render_max_max": 250,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: new best working Wu baseline for the strict ball-bounce setup. Compared with 12k, the mean foreground area ratio improves from `1.69` to `1.47` and the missing-frame issue goes from `3` zero-render frames to `0`. Compared with 15k, it also reduces overgrowth (`1.47` vs `1.79`). The QA sheet still shows softness, but this is the cleanest balance so far between visibility, size control, all-12-camera consistency, and temporal tracking. Use this as the baseline comparison / ablation-study checkpoint unless a later fit beats both the metrics and the viewer.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k`

Purpose: continue the 30k strict Wu baseline to 50k fine iterations to test whether more optimization reduces blur/blinking without changing the data, masks, camera set, temporal filtering, initialization, densification window, or loss weights.

Resume command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k \
  --iterations 50000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --wu-start-checkpoint wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/chkpnt_fine_30000.pth \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --skip-export \
  --skip-upload \
  --render
```

Data/export checks:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241` original timestamps, frames `0..240`
- Cameras: all `12`
- Rendered views: `2892`
- Temporal segmentation: no frames/views were dropped; source frame list preserves synchronized first-2-second timestamps
- Native size preserved: `960x544`
- Resume behavior: Wu reported `start from fine stage, skip coarse stage`, then ran `20000` additional fine steps from the 30k checkpoint.

Result:

- Completed `50000` fine iterations by resuming from `chkpnt_fine_30000.pth`.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k/point_cloud/iteration_50000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k/point_cloud/iteration_50000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k/chkpnt_fine_50000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 20191.445712309822,
  "render_area_max": 126943,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.4479558424139887,
  "ratio_median": 1.3840009714780126,
  "ratio_p90": 1.7422207659761844,
  "render_max_mean": 196.41459197786997,
  "render_max_max": 243,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: usable, slight quantitative improvement over 30k but not a decisive visual fix. The mean/median foreground area ratio improves (`1.45`/`1.38` vs `1.47`/`1.42`), and the worst foreground-area outlier is lower (`126943` vs `151895`), with zero missing frames. The QA sheet still shows softness and occasional overgrown smear, so 50k is useful evidence that longer optimization helps a little but does not fully solve blinking/blur by itself. Keep the 30k run as the cleaner baseline checkpoint, and keep this 50k run as the longer-fit ablation.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch`

Purpose: train the same strict 2-second setup to 50k fine iterations from scratch, instead of resuming from the 30k checkpoint. This isolates whether the longer-fit improvement is due to additional optimization itself or the resume dynamics.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch \
  --iterations 50000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --skip-export \
  --skip-upload \
  --render
```

Data/export checks:

- Source scene: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0`
- FPS: `120`
- Timesteps: `241` original timestamps, frames `0..240`
- Cameras: all `12`
- Rendered views: `2892`
- Temporal segmentation: source frame list keeps synchronized first-2-second timestamps; no views were dropped after the 6-camera / 50-pixel visibility check
- Native size preserved: `960x544`
- Point count at final checkpoint: `436585`

Result:

- Completed `50000` fine iterations from scratch.
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/point_cloud/iteration_50000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/point_cloud/iteration_50000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/chkpnt_fine_50000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 18170.849239280775,
  "render_area_max": 142540,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.3142967685673541,
  "ratio_median": 1.2710755088248598,
  "ratio_p90": 1.482912522226427,
  "render_max_mean": 195.57157676348547,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: new best quantitative Stage 1 run. It keeps zero missing render frames and improves foreground-area ratios substantially compared with both the 30k baseline (`1.31` mean vs `1.47`) and the 50k resumed run (`1.31` mean vs `1.45`). The QA sheet still shows visible softness, so this is not a solved reconstruction, but it is the best current checkpoint for sharing and for the next ablation/variant comparison.

### Strict 2-Second Blue-Ball Wu Comparison

This section ranks the comparable strict 2-second Wu ball-bounce runs on the current source scene:

```text
dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0
```

All ranked runs below use the same no-nonsense data recipe unless explicitly noted: 12 cameras, native `960x544`, 120 FPS source, first 2 seconds, frame-step-1 sampling, object-only spatial masks, and temporal visibility filtering with `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.

| Rank | Run | Zero frames | Area ratio mean / median / p90 | Notes |
|---:|---|---:|---|---|
| 1 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch` | 0 | `1.314 / 1.271 / 1.483` | Best quantitative fit so far. Still soft, but foreground size is closest to GT without missing frames. |
| 2 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k` | 0 | `1.473 / 1.424 / 1.624` | Strong baseline / ablation checkpoint. Visible, temporally coherent, less smeared than shorter den1500 runs. |
| 3 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k` | 0 | `1.448 / 1.384 / 1.742` | Slightly better mean/median than 30k but not visually decisive; one QA sample has stronger smear. Keep as longer-training ablation. |
| 4 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k` | 241 | `1.353 / 1.404 / 1.628` | Densifying until 6k did not help: area ratio is superficially close, but one full camera/timestep lane drops to black. Failed as a candidate. |
| 5 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter15k` | 0 | `1.786 / 1.667 / 2.112` | Stable visibility but more halo/soft spread than 30k. |
| 6 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter8k` | 0 | `1.823 / 1.718 / 2.103` | Earlier visible baseline, but under-supported and blurrier. |
| 7 | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1500_iter15k` | 0 | `1.863 / 1.771 / 2.135` | Longer fine fitting at low densification did not improve shape. |
| Diagnostic | `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter12k` | 3 | `1.693 / 1.602 / 1.914` | Good area ratio but still has missing render frames, so not the baseline. |

Current decision: use the 50k scratch run as the best current Wu ball result and the 30k run as the cleaner baseline comparison / ablation checkpoint. The 50k resumed run is useful because it shows that resume dynamics are not necessary for the metric improvement; from-scratch 50k is better. Next controlled ablations should isolate densification/opacity scheduling, temporal window length, and small restitution/angle changes.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k`

Purpose: test whether extending Wu densification to 6k iterations improves softness/blinking while keeping the same strict 2-second source data and loss settings.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k \
  --iterations 30000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 6000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --skip-upload \
  --render
```

Result:

- Completed `30000` fine iterations.
- Point count at final checkpoint: `386278`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den6000_iter30k/chkpnt_fine_30000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 19346.57745504841,
  "render_area_max": 113170,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.3525423728752268,
  "ratio_median": 1.4041388822947556,
  "ratio_p90": 1.627609687823232,
  "render_max_mean": 178.17185338865838,
  "render_max_max": 252,
  "zero_render_frames": 241,
  "nonzero_gt_frames": 2892
}
```

Verdict: failed as an improvement despite a plausible mean area ratio. The run has `241` zero-render frames, so one full 241-frame lane is effectively blank. The QA sheet also shows slightly dimmer/softer renders than the 50k scratch run. Do not use this as a baseline; keep it only as evidence that simply extending densification longer can destabilize visibility.

#### `wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`

Purpose: test whether reducing temporal difficulty from the strict 2-second window to the first 1.5 seconds improves fit quality without changing the scene, masks, camera count, or frame stride.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k \
  --iterations 30000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/frame_list_first1p5s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Completed `30000` fine iterations.
- Point count at final checkpoint: `465855`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_ball12_1p5s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/chkpnt_fine_30000.pth`

Metrics:

```json
{
  "num_frames": 2172,
  "threshold": 8,
  "render_area_mean": 20164.49217311234,
  "render_area_max": 127508,
  "gt_area_mean": 13833.113259668507,
  "gt_area_max": 28540,
  "ratio_mean": 1.419553407683168,
  "ratio_median": 1.3238879583014844,
  "ratio_p90": 1.658692412244047,
  "render_max_mean": 202.58287292817678,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2172
}
```

Verdict: usable diagnostic, not the new best. The shorter window keeps all 12 cameras and has no blank render frames, but its average foreground footprint is larger than the 50k strict 2-second scratch run (`1.42` mean vs `1.31`). The QA sheet shows visible ball tracking, but still with soft spread. Keep this as evidence that simply shortening to 1.5 seconds does not solve the blur/area issue.

### Stage 2 Variant Sweep

Goal: test whether the current best Wu recipe transfers to nearby physics settings before scaling to the full 3x3 grid. These runs keep the strict segmentation/fitting assumptions:

- 12 training cameras
- 120 FPS
- 2.0 second window
- object-only ball masks
- spatial segmentation from masks
- temporal filtering with `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`
- no frame stride

#### `wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`

Purpose: nearby restitution variant, centered below the promising `e0p93_a0p0` baseline, to test transfer of the recipe to `e=0.89`, `angle=0`.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0 \
  wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k \
  --iterations 30000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Completed `30000` fine iterations.
- Point count at final checkpoint: `387553`
- Modal train run: `ap-yofqTfpE35u5jZchOLGV2t`
- Modal render run: `ap-HzrMTxnEDusijITln5E5e8`
- Local model folder: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`
- Viewer: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.png`
- QA JSON: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.json`
- PLY: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_variant_ball_marker_e0p89_a0p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p89_a0p0/4dgs_wu/wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_variant_ball_e0p89_a0p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/chkpnt_fine_30000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 13694.899031811894,
  "render_area_max": 87862,
  "gt_area_mean": 13408.985822959889,
  "gt_area_max": 28540,
  "ratio_mean": 0.9729121408223769,
  "ratio_median": 1.3373205693743189,
  "ratio_p90": 1.5928754056951144,
  "render_max_mean": 129.22821576763485,
  "render_max_max": 255,
  "zero_render_frames": 964,
  "nonzero_gt_frames": 2892
}
```

Verdict: failed/diagnostic. The mean area ratio looks close to GT, but it is misleading because `964 / 2892` render frames are blank. The QA sheet shows multiple `R/GT=0.00` panels. Do not use this as a good variant result; the recipe did not transfer cleanly to `e=0.89` at 30k.
