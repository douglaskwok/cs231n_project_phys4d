# Experiment 1: Mask-Guided Gaussian Binding

Goal: train or download a normal full-scene Gaussian model, then use ball masks to select the Gaussians that likely belong to the ball.

This is the safer first experiment because full-scene 3DGS/4DGS has stable background context, while masks are used only for object binding.

## Inputs

- Gaussian PLY from 3DGS or exported 4DGS:
  - example: `gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply`
  - example: `4dgs_sphere_bounce/point_cloud/exported/point_cloud.ply`
- Mask frames:
  - expected layout: `outputs/sphere_bounce_m2/masks/cam00/frame00000.png`
- Camera metadata:
  - PyBullet cameras: `outputs/sphere_bounce_m2/cameras.json`
- Optional object poses:
  - `outputs/sphere_bounce_m2/object_poses.csv`

The masks can come from PyBullet first, then SAM-2 later. The selection logic is identical.

## Run

```bash
python segmentation/01_mask_guided_gaussian_binding/select_ball_gaussians.py \
  --ply gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply \
  --masks-root outputs/sphere_bounce_m2/masks \
  --cameras-json outputs/sphere_bounce_m2/cameras.json \
  --frame 0 \
  --min-camera-hits 2 \
  --out-dir segmentation/01_mask_guided_gaussian_binding/runs/frame00000
```

The script writes:

- `ball_gaussians.ply`
- `background_gaussians.ply`
- `selection_report.json`

## Smoke Test

This does not need a real trained PLY:

```bash
python segmentation/01_mask_guided_gaussian_binding/select_ball_gaussians.py --smoke-test
```

## Why This Tests The Idea

Each Gaussian center is projected into calibrated camera views. If it lands inside the ball mask in enough views, it is treated as ball-linked. Multi-view agreement is what makes this realistic.

Recommended starting thresholds:

- `--min-camera-hits 1`: high recall, noisy
- `--min-camera-hits 2`: good first real run
- `--min-camera-hits 3+`: cleaner, may miss small ball Gaussians

