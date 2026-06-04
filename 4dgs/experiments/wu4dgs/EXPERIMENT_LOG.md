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

#### `wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`

Purpose: nearby angle variant, keeping the promising `e=0.93` restitution but changing the drop angle to `+5` degrees.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0 \
  wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k \
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
- Point count at final checkpoint: `387835`
- Modal train run: `ap-6i9ZNGPOYJOqYTVIpMKr0O`
- Modal render run: `ap-Kd5VSxKAqvX55lcyq3Av2g`
- Local model folder: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k`
- Viewer: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.png`
- QA JSON: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k_qa.json`
- PLY: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter30k/chkpnt_fine_30000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 21707.03284923928,
  "render_area_max": 129056,
  "gt_area_mean": 13739.724066390041,
  "gt_area_max": 29428,
  "ratio_mean": 1.5900275903535666,
  "ratio_median": 1.521176739311732,
  "ratio_p90": 1.9091125587494557,
  "render_max_mean": 193.58367911479945,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: usable diagnostic, not the new best. Unlike the `e0p89_a0p0` variant, this does not blank out (`zero_render_frames = 0`) and the ball is visible across the sampled QA frames. However, the rendered foreground is inflated compared with GT (`ratio_mean = 1.59`, `ratio_median = 1.52`, `ratio_p90 = 1.91`), so the baseline `e0p93_a0p0` 50k scratch run remains the better result for presentation and ablation.

#### `wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch`

Purpose: controlled rerun of the `+5` degree angle variant from scratch at `50000` fine iterations. This isolates whether the angled case simply needed longer optimization to become densely packed.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0 \
  wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch \
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
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Result:

- Completed `50000` fine iterations.
- Point count at final checkpoint: `372439`
- Modal train run: `ap-k0Gy5GmQlL5VBY0QIl946S`
- Modal render run: `ap-TCN3gLjsaW9IA0PzIPLRQ9`
- Local model folder: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch`
- Viewer: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_viewer/index.html`
- QA sheet: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_qa.png`
- QA JSON: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch_qa.json`
- PLY: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/point_cloud/iteration_50000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/point_cloud/iteration_50000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_variant_ball_marker_e0p93_a5p0_table0p40_h0p80_r0p20_120fps_2p0s/scene_0000_e0p93_a5p0/4dgs_wu/wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/wu4dgs_wu_variant_ball_e0p93_a5p0_2s_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/chkpnt_fine_50000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 23140.277316735825,
  "render_area_max": 178671,
  "gt_area_mean": 13739.724066390041,
  "gt_area_max": 29428,
  "ratio_mean": 1.6900005217592022,
  "ratio_median": 1.638663800834216,
  "ratio_p90": 1.9730529985816814,
  "render_max_mean": 196.87551867219918,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: usable diagnostic, but worse than the `+5` degree 30k run and worse than the straight-down 50k baseline. It preserves visibility across all frames (`zero_render_frames = 0`) and uses all 12 cameras, but longer optimization alone did not make the angled case densely packed. The rendered foreground grew more inflated (`ratio_mean = 1.69`, `ratio_median = 1.64`, `ratio_p90 = 1.97`) than the 30k angle run (`ratio_mean = 1.59`). This suggests the next useful knob is not simply more iterations; try tightening opacity/densification or stronger anti-spill/area pressure while keeping the same spatial+temporal segmentation.

### Tightened Support Trials

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill2_area0p2_scaleiso5e-2_den2500_iter50k_tight_first2s`

Purpose: test whether stronger outside-mask pressure and earlier densification cutoff make the best straight-down baseline less fuzzy/blinky.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill2_area0p2_scaleiso5e-2_den2500_iter50k_tight_first2s \
  --iterations 50000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 2.0 \
  --area-loss-weight 0.20 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 2500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Export check:

- Correct strict 2-second setup: `241` timestamps, `2892` train views, all 12 cameras.
- Spatial+temporal segmentation preserved: `drop_invisible_frames=true`, `min_visible_cameras=6`, `min_mask_pixels=50`, no empty or below-threshold masks.

Result: aborted as unstable. The run hit NaN/restart around fine iteration `~2970`, then hit NaN/restart again around `~5250`. The Modal app was stopped manually (`ap-xQUxBAcTDdv5f9omYNTJ1o`) before wasting more time.

Verdict: failed/unstable. Do not use the combined `bg_spill=2.0 + area=0.20 + densify_until=2500` recipe as the next baseline. The loss pressure is likely too sharp numerically for this Wu setup.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den2500_iter50k_densifyonly_first2s`

Purpose: isolate whether stopping densification at `2500` instead of `4000` reduces fuzzy support while keeping the otherwise stable best-baseline losses.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den2500_iter50k_densifyonly_first2s \
  --iterations 50000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 2500 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Export check:

- Correct strict 2-second setup: `241` timestamps, `2892` train views, all 12 cameras.
- Spatial+temporal segmentation preserved.

Result: aborted early because the premise failed. By fine iteration `~2100`, the point count had already jumped to about `397624`, i.e. the run had already entered the same large-cloud regime as the previous dense baselines before the `2500` cutoff mattered. The Modal app was stopped manually (`ap-lUttFUMWvHZeRxQNHHJD5f`).

Verdict: failed diagnostic, not a quality failure. `densify_until=2500` is not an effective "fewer points / tighter support" knob for this scene because most of the point explosion happens before then. If testing density control again, use a much earlier cutoff or a different densification threshold, not merely `2500`.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s`

Purpose: test whether directly penalizing large Gaussian scale can reduce fuzz/blinking in the current straight-down 2-second baseline without changing the data.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s \
  --iterations 50000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 0.8 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --max-scale-loss-weight 50 \
  --max-gaussian-scale 0.01 \
  --wu-densify-until-iter 4000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --frame-list dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/frame_list_first2s.txt \
  --init-points 10000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

Export check:

- Correct strict 2-second setup: `241` timestamps, `2892` train views, all 12 cameras.
- Spatial+temporal segmentation preserved: `drop_invisible_frames=true`, `min_visible_cameras=6`, `min_mask_pixels=50`, and `frame_list_first2s.txt`.

Result:

- Completed `50000` fine iterations after one early NaN/restart; second attempt was stable.
- Point count at final checkpoint: `370426`.
- Modal train run: `ap-JZjTQBi2OLwGNPmOGTL8T5`
- Modal render run: `ap-cOpDTgzZncjhSIGymVt9rK`
- Local model folder: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s/point_cloud/iteration_50000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s/point_cloud/iteration_50000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale50_cap0p01_den4000_iter50k_first2s/chkpnt_fine_50000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 20545.198824343017,
  "render_area_max": 522240,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.709439267080635,
  "ratio_median": 1.1713841713136406,
  "ratio_p90": 1.4445719947367552,
  "render_max_mean": 98.77213001383126,
  "render_max_max": 242,
  "zero_render_frames": 1335,
  "nonzero_gt_frames": 2892
}
```

Verdict: failed diagnostic / not the new best. The max-scale penalty tightens the median foreground area (`ratio_median = 1.17`) and the QA sheet shows less obvious smear in several sampled views, but it makes support too weak or too faint in many camera-time pairs (`zero_render_frames = 1335`). This is worse than the current baseline for presentation and sharing. Keep `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch` as the best straight-down baseline unless a later run improves visibility without reintroducing fuzz.

#### `wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k`

Purpose: test whether continuing the current best 50k baseline to 75k fine iterations improves density/fuzz/blinking without changing the data or adding extra regularization. This is the same straight-down 2-second blue-ball setup, resumed from the 50k scratch checkpoint.

Command:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k \
  --iterations 75000 \
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
  --wu-start-checkpoint wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_scratch/chkpnt_fine_50000.pth \
  --render
```

Export check:

- Correct strict 2-second setup: `241` timestamps, `2892` train views, all 12 cameras.
- Spatial+temporal segmentation preserved: `drop_invisible_frames=true`, `min_visible_cameras=6`, `min_mask_pixels=50`, and `frame_list_first2s.txt`.
- Resume confirmed from `chkpnt_fine_50000.pth`; Wu skipped coarse stage and ran `25000` additional fine steps.

Result:

- Completed fine continuation to iteration `75000`.
- Final point count: `436585`.
- Modal train run: `ap-0TjCQUmjvaU6PcnyuJP8z9`
- Modal render run: `ap-lcOM8qqP50m8mYV3Cpvc4D`
- Local model folder: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/point_cloud/iteration_75000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/point_cloud/iteration_75000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/chkpnt_fine_75000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 17898.456085753805,
  "render_area_max": 129237,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.288030112043353,
  "ratio_median": 1.2698273948273948,
  "ratio_p90": 1.50525296152535,
  "render_max_mean": 192.3454356846473,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Comparison against prior kept baselines:

- `30k`: `ratio_mean=1.4733`, `ratio_median=1.4243`, `ratio_p90=1.6241`, `zero_render_frames=0`.
- `50k scratch`: `ratio_mean=1.3143`, `ratio_median=1.2711`, `ratio_p90=1.4829`, `zero_render_frames=0`.
- `75k from 50k`: `ratio_mean=1.2880`, `ratio_median=1.2698`, `ratio_p90=1.5053`, `zero_render_frames=0`.

Verdict: usable and likely the best quantitative baseline so far among the kept non-failed runs. It slightly improves mean foreground area ratio over 50k while preserving visibility in every rendered frame. The median is nearly tied with 50k, and `ratio_p90` is slightly worse, so visual QA still decides whether the extra 25k steps are worth the added size/time. Keep both `50k_scratch` and `75k_from50k` until human visual comparison confirms which has less blinking/fuzz.

#### Overnight 3x3 PyBullet generation and 100k continuation

Purpose: prepare the next parameter sweep while continuing the strongest straight-down Wu baseline from `75k` to `100k`.

Generated PyBullet data:

- Output folder: `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s`
- Duration: `2.0s`
- FPS: `120`
- Cameras: all 12 views
- Frames per scene: `241`
- Images/masks per scene: `2892`
- Main 3x3 grid: restitution `0.89, 0.93, 0.97` crossed with angle `-5, 0, +5`
- Extra row: restitution `0.85` with angle `-5, 0, +5`
- Fixed setup: marker ball, radius `0.20m`, table top z `0.40m`, drop height above surface `0.80m`, room cameras consistent with the successful Wu baseline.
- Manifest: `dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s/scenario_manifest.json`

Scenes:

- `scene_0000_e0p89_am5p0`
- `scene_0001_e0p89_a0p0`
- `scene_0002_e0p89_a5p0`
- `scene_0003_e0p93_am5p0`
- `scene_0004_e0p93_a0p0`
- `scene_0005_e0p93_a5p0`
- `scene_0006_e0p97_am5p0`
- `scene_0007_e0p97_a0p0`
- `scene_0008_e0p97_a5p0`
- `scene_0009_e0p85_am5p0`
- `scene_0010_e0p85_a0p0`
- `scene_0011_e0p85_a5p0`

100k continuation:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0 \
  wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k \
  --iterations 100000 \
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
  --wu-start-checkpoint wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k/chkpnt_fine_75000.pth \
  --skip-export \
  --skip-upload \
  --render
```

Result:

- Completed fine continuation from `75000` to `100000`.
- Remote model: `phys4d-gs-output:/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k`
- Modal train run: `ap-tvrbDnJ3VRJoSL8q9TMEvP`
- Modal render run: `ap-XYBIFTAW2qxjPexaIJZXu7`
- Local model folder: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k`
- Viewer: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k_viewer/index.html`
- QA sheet: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k_qa.png`
- QA JSON: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k_qa.json`
- PLY: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k/point_cloud/iteration_100000/point_cloud.ply`
- Deformation: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k/point_cloud/iteration_100000/deformation.pth`
- Checkpoint: `dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter100k_from75k/chkpnt_fine_100000.pth`

Metrics:

```json
{
  "num_frames": 2892,
  "threshold": 8,
  "render_area_mean": 17801.59163208852,
  "render_area_max": 149265,
  "gt_area_mean": 13620.92254495159,
  "gt_area_max": 28540,
  "ratio_mean": 1.2897762311488485,
  "ratio_median": 1.2722187185471754,
  "ratio_p90": 1.5500204877381265,
  "render_max_mean": 191.41528354080222,
  "render_max_max": 255,
  "zero_render_frames": 0,
  "nonzero_gt_frames": 2892
}
```

Verdict: completed and viewable. Quantitatively it is very close to `75k_from50k`: it keeps `zero_render_frames = 0`, but its mean/median area ratio does not materially improve over 75k and `ratio_p90` is a little worse. Human visual QA should decide whether 100k reduces blinking enough to justify keeping it as the new visual baseline. Keep `75k_from50k` and `100k_from75k` side by side for now.

## 2026-06-04: Collision Object-A Wu Canary Runs

Goal: test whether the collision failure was caused by the extra Wu mask-loss term or by the broader collision fitting setup. Both runs used the same collision source scene and preserved the required segmentation discipline:

- Source scene: `dataset/outputs/phys4d_final/collision_elastic_transparent_rails_centered_slow_2p6s_60fps/scene_0000_collision_room`
- Object mask: `masks_object_a`
- Spatial segmentation: object-only masked RGB on black background.
- Temporal filtering: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`
- Export check: 157 synchronized timestamps, 1884 train views, all 12 cameras, no empty or below-threshold masks.
- Init: `--init-points 12000 --init-center-mode first --init-surface-ratio 0.85`
- Fit controls: `--bounds 0.45 --foreground-loss-weight 20 --bg-spill-loss-weight 1.0 --area-loss-weight 0.10 --scale-isotropy-loss-weight 0.05 --max-scale-loss-weight 1.0 --max-gaussian-scale 0.04 --compactness-loss-weight 0.001 --wu-densify-until-iter 2500 --wu-opacity-reset-interval 300000`

### `wu_collision_recovery_a_firstinit_bounds0p45_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale1_cap0p04_compact1e-3_iter8000`

Difference tested: includes `--mask-loss-weight 1.0`.

Result:

- Trained and rendered after one NaN/restart.
- Local model folder: `dataset/outputs/phys4d_final/collision_elastic_transparent_rails_centered_slow_2p6s_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_recovery_a_firstinit_bounds0p45_fg20_mask1_spill1_area0p1_scaleiso5e-2_maxscale1_cap0p04_compact1e-3_iter8000`
- Render frames: 1884.
- Zero-render frames by threshold `>8`: `1738 / 1884`.
- Nonzero frames: `146 / 1884`.
- Foreground area mean/median/p90/max: `40470.83 / 0.0 / 0.0 / 522240`.

Verdict: failed. It is not the old all-black failure, but most frames are black and some nonzero frames are full-frame spill. The mask-loss term did not rescue collision.

### `wu_collision_recovery_a_nomaskloss_firstinit_bounds0p45_fg20_spill1_area0p1_scaleiso5e-2_maxscale1_cap0p04_compact1e-3_iter8000`

Difference tested: same recipe, but remove the extra `--mask-loss-weight` term while keeping masked RGB supervision and temporal filtering.

Result:

- Trained and rendered after one NaN/restart.
- Local model folder: `dataset/outputs/phys4d_final/collision_elastic_transparent_rails_centered_slow_2p6s_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_recovery_a_nomaskloss_firstinit_bounds0p45_fg20_spill1_area0p1_scaleiso5e-2_maxscale1_cap0p04_compact1e-3_iter8000`
- Render frames: 1884.
- Zero-render frames by threshold `>8`: `1869 / 1884`.
- Nonzero frames: `15 / 1884`.
- Foreground area mean/median/p90/max: `4157.96 / 0.0 / 0.0 / 522240`.

Verdict: failed, worse than the mask-loss canary. Removing the mask-loss term alone makes the render even more blank. The collision issue is therefore not explained by mask loss alone; the next useful direction is likely data/scale/visibility or a gentler collision-specific fitting recipe rather than simply dropping mask loss.

## 2026-06-04: Scaled Collision Direct Per-Object Wu 4DGS Canary

Goal: try Wu 4DGS directly on the newer scaled collision scene, fitting each object separately with spatial masks and temporal filtering. This is not yet the final collision recipe; it is a canary to test whether the larger objects, adjusted side cameras, and capped densification avoid the earlier black / full-frame spill failures.

Source scene:

- `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room`
- 12 cameras, 60 FPS, 2.6 s, 157 synchronized timestamps.
- Collision setup: object/wall footprint scaled by `1.75`, velocity scaled by `1.4`, side cameras moved outward/up.
- Per-object masks used: `masks_object_a` and `masks_object_b`.
- Temporal filtering: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.
- Export result for each object: 1884 train views, all 12 cameras, no dropped empty/below-threshold masks.

Shared Wu recipe:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  <run_name> \
  --mask-subdir <masks_object_a_or_b> \
  --mode object \
  --background black \
  --iterations 15000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --render
```

### Object A: `wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k`

Result:

- Completed 1000 coarse + 15000 fine iterations.
- Final point count: `50794`.
- Local model folder: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/point_cloud/iteration_15000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/chkpnt_fine_15000.pth`
- Render frames: `train/ours_15000/renders/*.png`
- QA sheet: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/train/ours_15000/collision_object_a_qa_sheet.jpg`

Verdict: usable canary, not final. Object A is visible and tracks motion across sampled timestamps/cameras, but render has blur/halo near the box edges.

### Object B: `wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k`

Result:

- Completed 1000 coarse + 15000 fine iterations.
- Final point count: `42740`.
- Local model folder: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/point_cloud/iteration_15000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/point_cloud/iteration_15000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/chkpnt_fine_15000.pth`
- Render frames: `train/ours_15000/renders/*.png`
- QA sheet: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/train/ours_15000/collision_object_b_qa_sheet.jpg`

Verdict: usable canary, not final. Object B is visible and tracks the blue box, but is blurrier/foggier than object A in the sampled render-vs-GT sheet.

Overall verdict: this is the first scaled collision Wu attempt where both per-object models are visible instead of black. It supports the direction of larger object scale + all-12-view spatial/temporal segmentation + capped densification. Still not presentation quality; next collision work should improve sharpness and then compose the two per-object 4DGS models in a common scene/time frame.

## 2026-06-04: Scaled Collision Per-Object Wu 30k Continuation

Goal: continue the usable scaled collision per-object canaries from 15k to 30k fine iterations, keeping the exact same data, masks, temporal filtering, and fitting recipe. This tests whether additional fitting time improves sharpness/stability without changing the scene or accidentally introducing a new variable.

Source scene:

- `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room`
- 12 cameras, 60 FPS, 2.6 s, 157 synchronized timestamps.
- Per-object masks:
  - Object A: `masks_object_a`
  - Object B: `masks_object_b`
- Temporal filtering: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.
- Export result for both objects: 1884 train views, all 12 cameras, no dropped frames/masks.
- Same coordinate/time frame for A and B; `transforms_train.json` is identical except expected mask-pixel metadata.

Shared continuation recipe:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  <run_name> \
  --mask-subdir <masks_object_a_or_b> \
  --mode object \
  --background black \
  --iterations 30000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --wu-start-checkpoint <matching_15k_remote_checkpoint> \
  --render
```

### Object A: `wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k`

Started from:

- Remote checkpoint: `wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/chkpnt_fine_15000.pth`

Result:

- Continued fine stage from 15k to 30k; coarse stage was skipped on resume.
- Final point count: `50794`.
- Local model folder: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/chkpnt_fine_30000.pth`
- Render frames: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/train/ours_30000/renders`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_a_30k/index.html`

### Object B: `wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k`

Started from:

- Remote checkpoint: `wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter15k/chkpnt_fine_15000.pth`

Result:

- Continued fine stage from 15k to 30k; coarse stage was skipped on resume.
- Final point count: `42740`.
- Local model folder: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/point_cloud/iteration_30000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/point_cloud/iteration_30000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/chkpnt_fine_30000.pth`
- Render frames: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/train/ours_30000/renders`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_30k/index.html`

Viewer notes:

- Both 30k viewers were built with all 12 cameras and 157 timesteps.
- Ground-truth comparison is enabled in the viewer.

Verdict: pending visual QA by viewer, but the train/resume/export path is now clean for both objects. These are the correct 30k continuation artifacts to inspect before deciding whether to continue to 50k or compose A+B in one render.

## 2026-06-04: Scaled Collision Per-Object Wu 50k Continuation

Goal: continue the scaled collision per-object Wu 4DGS runs from 30k to 50k fine iterations, using the same data and hyperparameters as the 30k run. This is the next quality checkpoint for the larger collision setup before attempting A+B composition in one shared 4DGS space.

Source scene:

- `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room`
- 12 cameras, 60 FPS, 2.6 s, 157 synchronized timestamps.
- Per-object masks:
  - Object A: `masks_object_a`
  - Object B: `masks_object_b`
- Spatial/temporal filtering: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`.
- Export result for both objects: 1884 train views, all 12 cameras, no dropped frames/masks.
- Same coordinate/time frame for A and B; this is important for later composition.

Shared continuation recipe:

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  <run_name> \
  --mask-subdir <masks_object_a_or_b> \
  --mode object \
  --background black \
  --iterations 50000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1.0 \
  --bg-spill-loss-weight 1.0 \
  --area-loss-weight 0.10 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.85 \
  --wu-start-checkpoint <matching_30k_remote_checkpoint> \
  --render
```

### Object A: `wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k`

Started from:

- Remote checkpoint: `wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/chkpnt_fine_30000.pth`

Result:

- Continued fine stage from 30k to 50k; coarse stage was skipped on resume.
- Final point count: `50794`.
- Local model folder: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/point_cloud/iteration_50000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/point_cloud/iteration_50000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth`
- Render frames: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/train/ours_50000/renders`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_a_50k/index.html`

### Object B: `wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k`

Started from:

- Remote checkpoint: `wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter30k_from15k/chkpnt_fine_30000.pth`

Result:

- Continued fine stage from 30k to 50k; coarse stage was skipped on resume.
- Final point count: `42740`.
- Local model folder: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k`
- Remote model: `phys4d-gs-output:/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/point_cloud/iteration_50000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/point_cloud/iteration_50000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth`
- Render frames: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/train/ours_50000/renders`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_50k/index.html`

Viewer notes:

- Both 50k viewers were built with all 12 cameras and 157 timesteps.
- Ground-truth comparison is enabled in both viewers.

Verdict: ready for visual QA. This is the current strongest per-object collision pair because it keeps the successful scaled collision data recipe and adds another 20k fine optimization steps on top of the 30k continuation. Next step is to inspect A/B 30k vs 50k visually, then decide whether to keep 50k as the collision baseline or attempt composition directly in shared Gaussian space.

## 2026-06-04: Collision A+B Gaussian-Space Composition Render

Goal: combine the two separately fit collision object models in Gaussian splat space, rather than compositing videos. For Wu 4DGS, a single native model folder has one deformation network, so directly concatenating two independently trained checkpoints would discard one deformation field. The correct practical composition is to load both object models, deform each model at the same camera timestamp, concatenate the deformed Gaussian tensors, and rasterize once.

Implementation:

- Added `render_wu_4dgs_composed` in `modal_app.py`.
- It loads multiple Wu model folders from `phys4d-gs-output`, keeps each model's own deformation network, and concatenates:
  - deformed means,
  - scales,
  - rotations,
  - opacities,
  - SH/color features,
  before a single Gaussian rasterization pass.
- This is not a 2D pixel overlay. It is a shared rasterization of both Gaussian sets in the same camera/time coordinate frame.

Command used:

```bash
arch -arm64 modal run modal_app.py \
  --render-wu-4d-compose \
  --render-wu-4d-compose-models wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k,wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k \
  --render-wu-4d-compose-output-model wu_collision_scale1p75_v1p4_ab_composed_50k \
  --render-wu-4d-iteration 50000 \
  --wu-time-resolution 120 \
  --wu-bounds 1.2
```

Result:

- Remote composed render: `phys4d-gs-output:/wu_collision_scale1p75_v1p4_ab_composed_50k/train/ours_50000`
- Local render frames: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/composed_ab_50k_render/train/ours_50000/renders`
- Local viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_ab_composed_50k/index.html`
- Viewer server used during QA: `http://127.0.0.1:8877/`
- Viewer contains all 12 cameras and 157 synchronized timestamps.

Storage note:

- Disk was full during viewer generation. Cleared the obsolete 15k/30k scaled collision Wu artifacts and their old viewers, preserving the 50k A/B models and the composed A+B render.

Verdict: implementation path works. This is the first collision output where A and B are rendered together through a single Gaussian rasterizer pass while preserving each object's learned deformation field.

## 2026-06-04: Collision A+B Composition With Render-Time Opacity Pruning

Why:

- The unpruned A+B Gaussian-space composition renders both independently fit objects through one rasterizer pass, but low-opacity splat dust from each object can accumulate when the objects overlap.
- Individually, those weak Gaussians are barely visible. Together, they can create haze, strange overlap artifacts, or extra mass between/around the two boxes.
- This run tests the conservative cleanup option before attempting any joint fine-tuning.

Implementation:

- Added a render-time opacity threshold to `render_wu_4dgs_composed` in `modal_app.py`.
- The checkpoint is not changed and no retraining happens.
- For each object at each timestamp:
  - deform the object with its own Wu deformation field,
  - activate opacity,
  - drop Gaussians with activated opacity below the threshold,
  - concatenate the remaining Gaussian tensors from A and B,
  - rasterize once.

Command:

```bash
arch -arm64 modal run modal_app.py \
  --render-wu-4d-compose \
  --render-wu-4d-compose-models wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k,wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k \
  --render-wu-4d-compose-output-model wu_collision_scale1p75_v1p4_ab_composed_50k_opacity0p05 \
  --render-wu-4d-iteration 50000 \
  --wu-time-resolution 120 \
  --wu-bounds 1.2 \
  --render-wu-4d-compose-opacity-threshold 0.05
```

Result:

- Remote render: `phys4d-gs-output:/wu_collision_scale1p75_v1p4_ab_composed_50k_opacity0p05/train/ours_50000`
- Local render frames: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/composed_ab_50k_opacity0p05_render/train/ours_50000/renders`
- Local viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_ab_composed_50k_opacity0p05/index.html`
- Viewer server used during QA: `http://127.0.0.1:8878/`
- Viewer contains all 12 cameras and 157 synchronized timestamps.
- Render/GT PNG count verified locally: `1884` renders and `1884` GT images.

Download note:

- `modal volume get` without a trailing slash on the remote `renders` path saved only one PNG-like file locally.
- The reliable folder command was:

```bash
arch -arm64 modal volume get phys4d-gs-output \
  wu_collision_scale1p75_v1p4_ab_composed_50k_opacity0p05/train/ours_50000/renders/ \
  tmp_renders_op05_dir \
  --force
```

Verdict:

- Ready for visual comparison against the unpruned composition.
- If `0.05` causes holes or removes too much mass, try `0.02`.
- If `0.05` barely changes the haze/overlap artifact, try `0.10`.

## 2026-06-04: Collision Trajectory Anchor Loss Ablation

Why:

- A and B are trained as separate object-only Wu 4DGS models, then composed later in Gaussian space.
- Even though both exports use the same source cameras, source frames, and scene coordinate system, separate masked/black-background optimization can still learn slightly different effective support/depth/opacity distributions.
- This ablation tested a light trajectory-centroid anchor from PyBullet `object_poses.csv`, without changing the source collision data.

Implementation:

- `4dgs/experiments/object_only/export_object_only_dynerf.py` now writes `trajectory_anchors.json` for object-mode exports when `object_poses.csv` is available.
- The anchor file stores the object name, original frame id, seconds, Wu-normalized time, and PyBullet world-space object center.
- `modal_app.py` can optionally patch Wu training with `--wu-trajectory-anchor-loss-weight`.
- The loss is only active when the weight is positive. Existing runs with the default `0` are unchanged.
- The loss computes the opacity-weighted centroid of the deformed Gaussian means at the current camera time and penalizes its squared distance from the nearest PyBullet anchor center.

Base run:

- Scene: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room`
- Object tested: `object_b`
- Start checkpoint: `wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth`
- Spatial segmentation: `masks_object_b`
- Temporal segmentation: `--drop-invisible-frames --min-visible-cameras 6 --min-mask-pixels 50`
- Cameras/timestamps after export: all 12 cameras, 157 timestamps.

Run 1: strong anchor

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  wu_collision_scale1p75_v1p4_object_b_anchor0p5_from50k_to60k \
  --mask-subdir masks_object_b \
  --mode object \
  --background black \
  --iterations 60000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 1 \
  --area-loss-weight 0.1 \
  --scale-isotropy-loss-weight 0.05 \
  --trajectory-anchor-loss-weight 0.5 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.8 \
  --wu-start-checkpoint wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth \
  --render
```

Result:

- Local model: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_anchor0p5_from50k_to60k`
- Local viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_anchor0p5_60k/index.html`
- Metrics against GT object-B masks:
  - old 50k mean centroid error: `20.21 px`, median `14.72 px`, mean area ratio `2.42`
  - anchor `0.5` mean centroid error: `29.40 px`, median `21.78 px`, mean area ratio `2.93`

Verdict: failed. The strong anchor made the object more smeared and less aligned.

Run 2: weak anchor

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k \
  --mask-subdir masks_object_b \
  --mode object \
  --background black \
  --iterations 55000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 1 \
  --area-loss-weight 0.1 \
  --scale-isotropy-loss-weight 0.05 \
  --trajectory-anchor-loss-weight 0.05 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.8 \
  --wu-start-checkpoint wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth \
  --render
```

Result:

- Local model: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k/point_cloud/iteration_55000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k/point_cloud/iteration_55000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_anchor0p05_from50k_to55k/chkpnt_fine_55000.pth`
- Local viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_anchor0p05_55k/index.html`
- Contact sheet: `/tmp/collision_object_b_anchor0p05_compare.jpg`
- Metrics against GT object-B masks:
  - old 50k mean centroid error: `20.21 px`, median `14.72 px`, mean area ratio `2.42`
  - anchor `0.05` mean centroid error: `30.92 px`, median `22.01 px`, mean area ratio `3.02`

Verdict: failed. The weak anchor still worsened centroid alignment and foreground area. Centroid anchoring alone is not enough for collision-object composition because it does not constrain shape/support/opacity; it can pull the mean while allowing a diffuse splat cloud.

Next recommendation:

- Keep the anchor code as an optional ablation, but do not use it for the current best collision models.
- For Gaussian-space composition, focus next on support cleanup or object-local normalization rather than centroid-only supervision.

## 2026-06-04: Collision Object-B Compactness Warm Starts

Goal:

- Improve collision object-B support before composing object A and object B in Gaussian space.
- Start from the existing object-B 50k checkpoint instead of retraining from scratch.
- Keep the same spatial and temporal segmentation rules used for the collision fits:
  - `masks_object_b`
  - `--drop-invisible-frames`
  - `--min-visible-cameras 6`
  - `--min-mask-pixels 50`
- Do not use trajectory anchor loss for these runs.

Common setup:

- Scene: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room`
- Start checkpoint: `wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth`
- Export status: all 12 cameras, 157 timestamps, no dropped views.
- Object-B physical size from metadata: approximately `0.42 x 0.42 x 0.2625 m`.

Baseline object-B 50k metrics:

- Local viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_50k/index.html`
- Mean centroid error: `20.21 px`
- Median centroid error: `14.72 px`
- Mean foreground area ratio: `2.42`
- Median foreground area ratio: `2.32`

Run A: stronger spill/area cleanup, same scale-isotropy weight

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k \
  --mask-subdir masks_object_b \
  --mode object \
  --background black \
  --iterations 60000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 2 \
  --area-loss-weight 0.2 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.8 \
  --wu-start-checkpoint wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth \
  --render
```

Result:

- Local model: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/point_cloud/iteration_60000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/point_cloud/iteration_60000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/chkpnt_fine_60000.pth`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_compactA_60k/index.html`
- Mean centroid error: `17.00 px`
- Median centroid error: `11.34 px`
- Mean foreground area ratio: `2.09`
- Median foreground area ratio: `2.04`

Verdict: current best object-B compactness recipe. It reduces centroid error and excess foreground support relative to the old 50k run.

Run B: stronger scale-isotropy weight

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k \
  --mask-subdir masks_object_b \
  --mode object \
  --background black \
  --iterations 60000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 2 \
  --area-loss-weight 0.2 \
  --scale-isotropy-loss-weight 0.1 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.8 \
  --wu-start-checkpoint wu4dgs_wu_collision_scale1p75_v1p4_object_b_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth \
  --render
```

Result:

- Local model: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k/point_cloud/iteration_60000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k/point_cloud/iteration_60000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactB_bg2_area0p2_iso0p1_from50k_to60k/chkpnt_fine_60000.pth`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_b_compactB_60k/index.html`
- Mean centroid error: `19.58 px`
- Median centroid error: `13.16 px`
- Mean foreground area ratio: `2.14`
- Median foreground area ratio: `2.07`

Verdict: usable but not better than run A. The stronger isotropy regularization cleaned foreground area relative to the old 50k baseline, but alignment was worse than compact-A.

Comparison artifact:

- Contact sheet: `/tmp/collision_object_b_compactAB_compare.jpg`

Recommendation:

- Use compact-A for object B unless later visual inspection finds a severe artifact not captured by the metrics.
- Next, apply the same compact-A recipe to object A from its 50k checkpoint, then attempt A+B Gaussian-space composition again.

Follow-up: compact-A applied to object A

```bash
bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
  dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room \
  wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k \
  --mask-subdir masks_object_a \
  --mode object \
  --background black \
  --iterations 60000 \
  --coarse-iterations 1000 \
  --time-resolution 120 \
  --bounds 1.2 \
  --foreground-loss-weight 20 \
  --mask-loss-weight 1 \
  --bg-spill-loss-weight 2 \
  --area-loss-weight 0.2 \
  --scale-isotropy-loss-weight 0.05 \
  --wu-densify-until-iter 1000 \
  --wu-opacity-reset-interval 300000 \
  --drop-invisible-frames \
  --min-visible-cameras 6 \
  --min-mask-pixels 50 \
  --init-points 20000 \
  --init-center-mode first \
  --init-surface-ratio 0.8 \
  --wu-start-checkpoint wu4dgs_wu_collision_scale1p75_v1p4_object_a_fg20_mask1_spill1_area0p1_scaleiso5e-2_den1000_iter50k_from30k/chkpnt_fine_50000.pth \
  --render
```

Result:

- Local model: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k`
- PLY: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/point_cloud/iteration_60000/point_cloud.ply`
- Deformation: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/point_cloud/iteration_60000/deformation.pth`
- Checkpoint: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/chkpnt_fine_60000.pth`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_object_a_compactA_60k/index.html`
- Old A 50k metrics:
  - mean centroid error: `9.34 px`
  - median centroid error: `9.26 px`
  - mean foreground area ratio: `1.73`
  - median foreground area ratio: `1.56`
- Compact-A 60k metrics:
  - mean centroid error: `8.61 px`
  - median centroid error: `7.15 px`
  - mean foreground area ratio: `1.58`
  - median foreground area ratio: `1.46`

Verdict: modest improvement. Compact-A is also better for object A, so it is a consistent recipe for both collision objects.

Gaussian-space A+B composition with compact-A

```bash
arch -arm64 modal run modal_app.py \
  --render-wu-4d-compose \
  --render-wu-4d-compose-models wu4dgs_wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k,wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k \
  --render-wu-4d-compose-output-model wu4dgs_collision_scale1p75_v1p4_ab_compactA_60k_composed \
  --render-wu-4d-iteration 60000 \
  --wu-time-resolution 120 \
  --wu-bounds 1.2 \
  --render-wu-4d-compose-opacity-threshold 0.0
```

Result:

- Remote render: `phys4d-gs-output:/wu4dgs_collision_scale1p75_v1p4_ab_compactA_60k_composed/train/ours_60000`
- Local render: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/wu4dgs_collision_scale1p75_v1p4_ab_compactA_60k_composed`
- Viewer: `dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room/4dgs_wu/viewer_ab_compactA_60k_composed/index.html`
- Contact sheet: `/tmp/collision_ab_compactA_60k_composed_sheet.jpg`

Verdict: improved but still not final. The render is a true Gaussian-space composition: deformed Gaussian tensors are concatenated before rasterization. It is less chaotic than the earlier composed 50k/opacity-threshold attempts, but some cameras still show dark/hazy residual support. If continuing, the next useful test is a slightly more aggressive compact-A variant on both objects, not trajectory anchoring.
