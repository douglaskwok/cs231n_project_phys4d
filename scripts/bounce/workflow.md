# Bounce pipeline workflow

Single-object trajectory prediction + composite render (steps 4a→6). Collision
(`scripts/collision/workflow.md`) is the multi-object analogue.

Always run modal commands from **your terminal** so progress bars are visible.

In `scripts/bounce`:

1. `step4a_extract.py` — Wu 4DGS trajectory extraction (default savgol window 9,
   polyorder 2). Writes `trajectory_smoothed.csv`.
2. `refit_metric_gt.py` — Umeyama align + least-squares fit `(p0, v0, e, g, y_ground)`.
   Writes `refit_metric.json` with `similarity.scale` for Step 5.
3. `predict_metric_from_refit.py` — integrate through the held-out window.
   Writes `trajectory_predicted.csv`.
4. `step5_render.py` on Modal with `--object-scale` (from refit) and
   `--object-crop-radius` (choose so `crop × scale ≈ 0.17 m`). **Use layered alpha
   compositing** (`--composite-mode alpha`, default): render background and object in
   separate 3D passes, then alpha-matte the object over the background. Avoids
   background floaters swallowing the ball in a single 3D merge. Optional:
   `--object-opacity-boost 4.0`.
5. `step6_eval.py` — trajectory + render metrics.

Then create an mp4 for the fused final scene.

## Step 5 notes

- Modal `--step5` always writes to volume root `phys4d-gs-output:step5/` (not
  `--bounce-out-rel`; that flag only applies to step4a). Wipe stale renders before
  re-running: `modal volume rm phys4d-gs-output step5 -r`.
- Default ball background on the volume:
  `bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply`.
- Render PNGs are named `{frame:04d}_cam{idx}.png` (e.g. `0235_cam0.png`, not `cam00`).

## Example (ball12 50k-from-30k)

```bash
PY=.venv_pipeline/bin/python
RUN=outputs/bounce_pipeline/<RUN>
SCENE=outputs/bounce_pipeline/run_50k_from30k/_step4a_stage/scene

# 4a — on Modal in practice
modal run modal_app.py --upload-step4a --step4a-dir "$RUN/_step4a_stage"
modal run modal_app.py --step4a --bounce-out-rel wu_ball12_50k_v1/step4a

# 4b + 4c — local
"$PY" scripts/bounce/refit_metric_gt.py \
  --traj "$RUN/step4a/trajectory_smoothed.csv" \
  --gt-poses "$SCENE/object_poses.csv" --out "$RUN/step4b_metric"
"$PY" scripts/bounce/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out "$RUN/step4c/trajectory_predicted.csv" \
  --frame-start 235 --frame-end 360 --anchor-frame 0 --fps 120 --zero-horizontal \
  --gt-poses "$SCENE/object_poses.csv"

SCALE=$("$PY" -c "import json; print(json.load(open('$RUN/step4b_metric/refit_metric.json'))['similarity']['scale'])")
CROP=$(python3 -c "print(round(0.17/float('$SCALE'), 3))")

# 5 — on Modal
modal run modal_app.py --upload-step5 --step5-dir "$RUN/_step5_stage"
modal volume rm phys4d-gs-output step5 -r 2>/dev/null || true
modal run modal_app.py --step5 \
  --canonical-rel object/point_cloud.ply --cfg-args-rel object/cfg_args \
  --bounce-bg-ply-rel bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply \
  --object-scale "$SCALE" --object-crop-radius "$CROP" \
  --object-opacity-boost 4.0 --composite-mode alpha

# 6 — local
"$PY" scripts/bounce/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted.csv" \
  --gt-poses "$SCENE/object_poses.csv" \
  --rendered "$RUN/step5/renders" --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/step6" --fps 120 \
  --extracted-traj "$RUN/step4a/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"
```

Convenience wrappers: `run_workflow_ball12.sh`, `run_workflow_wu75k_scene0003.sh`.
