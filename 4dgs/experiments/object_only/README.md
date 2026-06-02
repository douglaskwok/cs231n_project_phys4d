# Experiment: Object-Only 4DGS

Goal: train [fudan 4DGS](https://github.com/fudan-zvg/4d-gaussian-splatting) on only the segmented ball pixels (or other masked modes).

This is the more aggressive experiment. It directly tests whether a small moving object can be reconstructed as a 4D Gaussian scene without the table/room background. Expect it to be more brittle than mask-guided binding because the ball is small and nearly textureless.

Parent docs: [`4dgs/README.md`](../../README.md).

## Build object-only DyNeRF dataset

From repo root:

```bash
python 4dgs/experiments/object_only/export_object_only_dynerf.py \
  --rgb-root outputs/sphere_bounce_m2/rgb \
  --masks-root outputs/sphere_bounce_m2/masks \
  --output 4dgs/experiments/object_only/runs/dynerf_ball_only
```

Output layout (fudan-compatible):

- `transforms_train.json`, `transforms_test.json`
- `images/` (symlinks or copies depending on mode)

Smoke test:

```bash
python 4dgs/experiments/object_only/export_object_only_dynerf.py --smoke-test
```

## Train on Modal

```bash
modal run modal_app.py --upload-4d \
  --upload-4d-path 4dgs/experiments/object_only/runs/dynerf_ball_only
modal run modal_app.py --train-4d
```

Or upload only (no auto-export):

```bash
python 4dgs/scripts/upload_4d_scene_to_modal.py \
  4dgs/experiments/object_only/runs/dynerf_ball_only
modal run modal_app.py --train-4d
```

## Modes

`--mode ball_only` — keep only masked object pixels (default for small ball tests).

Other modes (`background`, `full`, etc.) are documented in `export_object_only_dynerf.py --help`.

Black background usually matches `white_background: false` in `4dgs/configs/sphere_bounce_4dgs.yaml`.
