# Experiment 2: Object-Only 4DGS

Goal: train fudan 4DGS on only the segmented ball pixels.

This is the more aggressive experiment. It directly tests whether a small moving object can be reconstructed as a 4D Gaussian scene without the table/room background. Expect it to be more brittle than Experiment 1 because the ball is small and nearly textureless.

## Inputs

- Full PyBullet RGB frames:
  - `outputs/sphere_bounce_m2/rgb/camXX/frameYYYYY.png`
- Ball masks:
  - `outputs/sphere_bounce_m2/masks/camXX/frameYYYYY.png`
- Camera metadata:
  - `outputs/sphere_bounce_m2/cameras.json`
- Existing config split:
  - `configs/sphere_bounce_m2.json`

## Build Object-Only DyNeRF Dataset

```bash
python segmentation/02_object_only_4dgs/export_object_only_dynerf.py \
  --config configs/sphere_bounce_m2.json \
  --masks-root outputs/sphere_bounce_m2/masks \
  --output segmentation/02_object_only_4dgs/dynerf_ball_only
```

The output has the normal 4DGS input layout:

- `images/`
- `transforms_train.json`
- `transforms_test.json`
- `export_meta.json`

## Smoke Test

```bash
python segmentation/02_object_only_4dgs/export_object_only_dynerf.py --smoke-test
```

## Run On Modal

After building the object-only dataset locally:

```bash
modal run modal_app.py \
  --upload-4d \
  --upload-4d-path segmentation/02_object_only_4dgs/dynerf_ball_only

modal run modal_app.py --train-4d
modal run modal_app.py --render-4d
modal run modal_app.py --eval-4d
```

This uses the existing Modal 4DGS path. The custom upload path lets this experiment replace `/data/4d_scene` on the Modal volume with the object-only dataset.

## Variants

Try both background modes:

```bash
# Black background outside mask.
python segmentation/02_object_only_4dgs/export_object_only_dynerf.py \
  --masks-root outputs/sphere_bounce_m2/masks \
  --background black

# White background outside mask.
python segmentation/02_object_only_4dgs/export_object_only_dynerf.py \
  --masks-root outputs/sphere_bounce_m2/masks \
  --background white
```

Black usually matches `white_background: false` in `configs/sphere_bounce_4dgs.yaml`.

