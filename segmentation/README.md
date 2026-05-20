# Segmentation Experiments

Tests for using video/object segmentation with Gaussian splatting.

## 01: Mask-Guided Gaussian Binding

Folder: `01_mask_guided_gaussian_binding/`

Use masks to select which already-trained full-scene Gaussians belong to the ball. This is the more realistic path and should be the first one to run.

## 02: Object-Only 4DGS

Folder: `02_object_only_4dgs/`

Use masks to create a ball-only DyNeRF/4DGS dataset, then train upstream fudan 4DGS on Modal. This is more experimental because the ball is small and smooth.

## SA4D-Inspired: Gaussian Identity Table

Folder: `sa4d/`

Build a multi-frame Gaussian identity table from a full-scene 4DGS PLY and object masks. This is a lightweight local approximation of SA4D's identity-table idea, without training their temporal identity feature field.

## Modal Note

`modal_app.py --upload-4d` now accepts:

```bash
--upload-4d-path segmentation/02_object_only_4dgs/dynerf_ball_only
```

If omitted, it keeps the original behavior and exports/uploads `outputs/sphere_bounce_m2/dynerf_sphere_bounce`.
