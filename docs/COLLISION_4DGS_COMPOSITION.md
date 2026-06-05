# Collision 4DGS Object Composition

## Summary

For the collision scenes, we do not train object A and object B as one fused Wu 4DGS model. Instead, we train one object-wise dynamic Gaussian field per object, using the same PyBullet scene, cameras, timestamps, and coordinate system. We then compose the object-wise renders as aligned layers.

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

The important contract is that both objects share the same frame indices, timestamps, camera calibration, and PyBullet world coordinates.

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

## Why Not Direct Gaussian Concatenation?

We tested direct Gaussian-space concatenation, where the deformed Gaussians from object A and object B are merged into one rasterization pass.

That was not the best result. The separately trained objects looked reasonable individually, but the naive fused render became too dark or hazy. The likely reason is alpha-compositing interference: broad low-opacity support from one object can suppress or obscure the other object in a single shared raster pass.

So the problem was not that the objects used different coordinates. They were aligned. The issue was the rendering/compositing behavior of independently trained Gaussian fields.

## Current Best Composition

The current best approach is layered object-wise rendering:

1. Render object A at a given camera/time.
2. Render object B at the same camera/time.
3. Composite the two rendered foreground layers.

The best collision visualization used a thresholded layered composition:

```text
viewer_ab_compactA_60k_layered_thr48
```

This is not a single fused Gaussian tensor. It is a layered, object-wise 4DGS representation in one shared scene/time coordinate system.

## Why This Is Useful For Prediction

For trajectory prediction, preserving object identity is a feature, not a bug. The model can reason about:

- object A trajectory
- object B trajectory
- interaction/collision timing
- object-wise Gaussian appearance

Then the predicted object states can be rendered or evaluated per object before composing the final view.

## One-Sentence Explanation

We fit object A and object B as separate dynamic Gaussian fields using shared cameras, timestamps, and world coordinates, then render them as aligned layers because naive one-pass Gaussian concatenation caused opacity artifacts while layered composition preserved both object identity and visual alignment.
