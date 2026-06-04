#!/usr/bin/env bash
# Run full workflow.md for nativeaspect r0p20 — execute in YOUR terminal (no agent approval).
set -euo pipefail
cd "$(dirname "$0")/../.."
export HOME="${HOME:-$HOME}"
export PYTHONPATH=src
PY="${PY:-.venv_pipeline/bin/python}"
RUN=outputs/bounce_pipeline/wu_debug_ball_r0p20_120fps_0p5s_e0p90_a0p0_wu_object_nativeaspect_fg20_mask1_round0p02_scaleiso5e-2_cloudiso1e-1_init10k_first_surface85_den1500_noreset_bounds0p8_dropempty
MBASE=wu_r0p20_nativeaspect
SCENE=wu_debug_ball_r0p20_120fps_0p5s/scene_0000_e0p90_a0p0

echo "=== step4a upload + run ==="
modal run modal_app.py --upload-step4a --step4a-dir "$RUN/_step4a_stage"
modal run modal_app.py --step4a --bounce-out-rel "$MBASE/step4a"

echo "=== download step4a ==="
mkdir -p "$RUN/step4a"
for f in trajectory_smoothed.csv trajectory_raw.csv step4a_meta.json trajectory_plot.png; do
  modal volume get phys4d-gs-output "$MBASE/step4a/$f" "$RUN/step4a/$f" --force
done

echo "=== refit + predict ==="
"$PY" scripts/bounce/refit_metric_gt.py \
  --traj "$RUN/step4a/trajectory_smoothed.csv" \
  --gt-poses "$SCENE/object_poses.csv" \
  --out "$RUN/step4b_metric" \
  --train-start 0 --train-end 39 --test-start 40 --test-end 60
"$PY" scripts/bounce/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out "$RUN/step4c/trajectory_predicted.csv" \
  --frame-start 40 --frame-end 60 --anchor-frame 0 --fps 120 --zero-horizontal \
  --gt-poses "$SCENE/object_poses.csv"

SCALE=$("$PY" -c "import json; print(json.load(open('$RUN/step4b_metric/refit_metric.json'))['similarity']['scale'])")
CROP=0.33
echo "scale=$SCALE crop=$CROP"

echo "=== step5 stage upload run ==="
mkdir -p "$RUN/_step5_stage/step4a" "$RUN/_step5_stage/step4c"
cp -R "$RUN/_step4a_stage/export" "$RUN/_step5_stage/"
cp -R "$RUN/_step4a_stage/object" "$RUN/_step5_stage/"
cp -R "$RUN/_step4a_stage/scene" "$RUN/_step5_stage/"
cp "$RUN/step4a/trajectory_smoothed.csv" "$RUN/_step5_stage/step4a/"
cp "$RUN/step4c/trajectory_predicted.csv" "$RUN/_step5_stage/step4c/"
modal run modal_app.py --upload-step5 --step5-dir "$RUN/_step5_stage"
modal run modal_app.py --step5 \
  --bounce-out-rel "$MBASE/step5" \
  --canonical-rel object/point_cloud.ply \
  --cfg-args-rel object/cfg_args \
  --bounce-bg-ply-rel bg_wu_huge_3dgs/point_cloud/iteration_7000/point_cloud.ply \
  --object-scale "$SCALE" \
  --object-crop-radius "$CROP"

echo "=== download step5 ==="
mkdir -p "$RUN/step5/renders"
modal volume get phys4d-gs-output "$MBASE/step5/step5_meta.json" "$RUN/step5/step5_meta.json" --force
modal volume ls phys4d-gs-output "$MBASE/step5/renders" 2>/dev/null | while IFS= read -r rel; do
  base="${rel##*/}"
  [[ -n "$base" ]] && modal volume get phys4d-gs-output "$MBASE/step5/renders/$base" "$RUN/step5/renders/$base" --force
done

echo "=== step6 + mp4 ==="
MPLCONFIGDIR=/tmp/mpl "$PY" scripts/bounce/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted.csv" \
  --gt-poses "$SCENE/object_poses.csv" \
  --rendered "$RUN/step5/renders" \
  --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/step6" --fps 120 \
  --extracted-traj "$RUN/step4a/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"

mkdir -p "$RUN/step5/_mp4_frames"
for fr in $(seq 40 60); do
  src=$(printf '%04d_cam10.png' "$fr")
  dst=$(printf "$RUN/step5/_mp4_frames/frame%05d.png" $((fr - 40)))
  cp "$RUN/step5/renders/$src" "$dst" 2>/dev/null || true
done
ffmpeg -y -loglevel error -framerate 120 -i "$RUN/step5/_mp4_frames/frame%05d.png" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
  "$RUN/step5/fused_scene_cam10.mp4" || echo "mp4 skip (missing frames)"
rm -rf "$RUN/step5/_mp4_frames"

bash scripts/bounce/ping_user.sh
echo "DONE $RUN"
