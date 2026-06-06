# Collision 4DGS Object Composition

## Summary

For the collision scenes, we do not train object A and object B as one fused Wu 4DGS model. Instead, we train one object-wise dynamic Gaussian field per object, using the same PyBullet scene, cameras, timestamps, and coordinate system. The two learned object fields therefore live in the same scene/world coordinate frame. For visualization, we then combine them either by same-coordinate Gaussian-space rendering or, for the best viewers, by composing same-coordinate object renders as layers.

This keeps object identity explicit, which is useful for downstream trajectory prediction.

## Data Setup

Each collision scene contains:

```text
scene/
  rgb/
  masks_object_a/
  masks_object_b/
  cameras.json
  object_poses.csv
  config.json
  metadata.json
```

`object_poses.csv` stores both objects with stable names:

```text
object_a
object_b
```

The important contract is that both objects share the same frame indices, timestamps, camera calibration, and PyBullet world coordinates. This is what lets object A and object B be recombined without estimating an extra alignment transform between them.

## Object-Wise 4DGS Fitting

We fit the two objects separately:

- object A uses `masks_object_a`
- object B uses `masks_object_b`

Both fits use the same RGB frames and the same cameras. The masks define which object each run should reconstruct.

The fitting uses both spatial and temporal segmentation:

- spatial segmentation: object-specific masks
- temporal segmentation: keep timestamps where the object is visible in at least 6 cameras
- visibility threshold: more than 50 mask pixels in a camera counts as visible

The exports preserve original timestamps/frame IDs, so object A and object B stay aligned after separate fitting.

In the successful collision runs, the main fitting pattern was:

- start from an object-specific Wu 4DGS fit, usually around `50k` iterations
- continue or clean up the object fit to a selected checkpoint, commonly `60k`
- keep both objects in the original scene coordinate frame rather than recentering them independently
- build the combined viewer from the selected object A and object B render folders

The exact selected checkpoint varies by scene. For example, scene `0000` uses object A `compactA_60k` and object B `compactA_60k`; scene `0004` uses object A `70k` cleanup with object B `60k`; scene `0006` uses object A `60k` with the object B all-pose `30k`-init retry continued to `60k`.

## Same-Coordinate Combination

There are two combination paths, both based on the same coordinate system:

1. Gaussian-space composition: deform object A and object B at the same timestamp, concatenate their Gaussian tensors, and rasterize them together with the same camera.
2. Layered render composition: render object A and object B separately with the same camera/timestamp, then composite the resulting foreground layers.

The first path is implemented by `render_wu_4dgs_composed` in `modal_app.py`. It keeps each object's learned deformation network, evaluates both fields at the current camera timestamp, concatenates the deformed means/scales/rotations/opacities/SH features, and rasterizes them in one pass. This is the direct Gaussian-space version of "combine them in the same coordinate system."

The second path is implemented by `4dgs/scripts/composite_4dgs_renders.py`. It still depends on the same shared camera/time/world coordinates, but the final composition happens after rendering.

## Why Not Use Direct Gaussian Concatenation As Best?

We tested direct Gaussian-space concatenation, where the deformed Gaussians from object A and object B are merged into one rasterization pass.

That was not the best result. The separately trained objects looked reasonable individually, but the naive fused render became too dark or hazy. The likely reason is alpha-compositing interference: broad low-opacity support from one object can suppress or obscure the other object in a single shared raster pass.

So the problem was not coordinate mismatch. The objects were already in a shared coordinate system. The issue was the rendering/compositing behavior of independently trained Gaussian fields in a single rasterization pass.

## Current Best Composition

The current best approach is layered object-wise rendering:

1. Render object A at a given camera/time.
2. Render object B at the same camera/time.
3. Composite the two rendered foreground layers.

The best collision visualization used a thresholded layered composition:

```text
viewer_ab_compactA_60k_layered_thr48
```

Here `thr48` is the important render-composition parameter. In
`4dgs/scripts/composite_4dgs_renders.py`, the foreground mask is:

```python
max(rgb_channel) > threshold
```

so `thr48` means pixels with maximum RGB value `<= 48` are treated as background before layer compositing. In the viewer tracker this is also described as a `0.48` opacity/foreground threshold; operationally, the local layered render metadata records the integer threshold as `48`.

The selected mode for this family of viewers is thresholded max compositing:

- render object A and object B as black-background object-only images
- zero sub-threshold pixels in each object layer
- take the per-pixel maximum across the surviving object layers
- preserve the original 12-camera, synchronized timestamp layout

For scene `0000`, the local QA comparison was:

- naive single-raster A+B composition: mean foreground area ratio `0.37`, foreground mean intensity `17.2`
- layered max without threshold: mean foreground area ratio `1.62`, foreground mean intensity `58.5`
- layered `thr48`: mean foreground area ratio `1.26`, foreground mean intensity `71.7`, zero blank samples

This best viewer is not a single fused Gaussian tensor. It is a layered, object-wise 4DGS representation in one shared scene/time coordinate system. The underlying object fields are still same-coordinate fields; only the final visual compositing is layered.

## Current Scene Picks

The current combined collision viewers are:

| Scene | Combined viewer | Object checkpoints used | Status |
| --- | --- | --- | --- |
| `0000` | `viewer_ab_compactA_60k_layered_thr48` | A `compactA_60k`, B `compactA_60k` | best |
| `0001` | `viewer_ab_staged_50k10k_layered_thr48` | staged A/B `50000_to60000` | baseline |
| `0002` | `viewer_ab_compactA_60k_layered_thr48` | A/B `compactA_60k` combined renders | best |
| `0003` | `viewer_ab_a60k_b80k_desaturn_layered_thr48` | A `60k`, B `80k` desaturn | best |
| `0004` | `viewer_ab_a70k_b60k_layered_thr48` | A `70k` cleanup, B `60k` redo | best |
| `0005` | `viewer_ab_staged_50k10k_layered_thr48` | staged A/B `50000_to60000` | best |
| `0006` | `viewer_ab_a60k_bretry_all30k_60k_layered_thr48` | A `60k`, B all-pose `30k`-init retry to `60k` | best |
| `0007` | `viewer_ab_compactA_60k_layered_thr48` | A/B `compactA_60k` | best |

The diagnostic `viewer_ab_75k_joint_thr048` for scene `0004` should not be treated as the best composition; it was a joint diagnostic viewer and looked worse than the layered A/B composition.

## Why This Is Useful For Prediction

For trajectory prediction, preserving object identity is a feature, not a bug. The model can reason about:

- object A trajectory
- object B trajectory
- interaction/collision timing
- object-wise Gaussian appearance

Then the predicted object states can be rendered or evaluated per object before composing the final view.

## One-Sentence Explanation

We fit object A and object B as separate dynamic Gaussian fields in the same scene/world coordinate system, then either concatenate their deformed Gaussian tensors for one-pass rendering or, for the best viewers, render them at the same camera/time and use thresholded layered compositing to avoid opacity artifacts.
