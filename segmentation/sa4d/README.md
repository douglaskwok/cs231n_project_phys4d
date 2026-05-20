# SA4D-Inspired Gaussian Identity Table

This folder is a local experiment inspired by **Segment Any 4D Gaussians (SA4D)**.

The public `jsxzs/SA4D` repository currently exposes the paper/project page but not a runnable implementation. This folder therefore implements the first practical piece we can test in this repo:

```text
full-scene 4DGS PLY + multiview masks over time
-> per-frame Gaussian identity table
-> stable/object Gaussian PLYs
```

This is not the full SA4D temporal identity feature field. It is a mask-projection baseline that helps us test whether our segmented videos can identify ball-linked Gaussians across time.

## Inputs

- A full-scene Gaussian PLY exported from trained 4DGS.
- `masks/camXX/frameYYYYY.png`
- `cameras.json`

## Run

After training/exporting a full-scene 4DGS PLY:

```bash
phys_sim/bin/python segmentation/sa4d/build_identity_table.py \
  --ply 4dgs_sphere_bounce/point_cloud/exported/point_cloud.ply \
  --masks-root dataset/outputs/ping_pong_12view_restitution_ball_angle_poc/scene_0003_e0p90_a15p0/masks \
  --cameras-json dataset/outputs/ping_pong_12view_restitution_ball_angle_poc/scene_0003_e0p90_a15p0/cameras.json \
  --frame-start 0 \
  --frame-end 90 \
  --frame-step 10 \
  --min-camera-hits 2 \
  --out-dir segmentation/sa4d/runs/ping_pong_scene_0003
```

Outputs:

- `identity_table.npz`: boolean table with shape `(num_frames, num_gaussians)`.
- `identity_table.json`: readable summary.
- `stable_ball_gaussians.ply`: Gaussians selected at least `--min-temporal-hits` times.
- `stable_background_gaussians.ply`

For per-frame PLYs, add:

```bash
--write-frame-plys
```

## Smoke Test

```bash
phys_sim/bin/python segmentation/sa4d/build_identity_table.py --smoke-test
```

## How This Relates To SA4D

SA4D trains a time-dependent identity feature field:

```text
identity = phi(position, time)
```

and refines/stores identities in a Gaussian identity table. This local experiment skips the learned feature field and directly builds an identity table from mask projection. It is the right lightweight baseline before implementing a learned temporal identity model.
