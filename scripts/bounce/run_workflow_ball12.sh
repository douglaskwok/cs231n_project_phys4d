#!/usr/bin/env bash
# Run full workflow.md for a ball12 50k checkpoint — execute in YOUR terminal.
# Usage: run_workflow_ball12.sh <v1|v2>
set -euo pipefail
cd "$(dirname "$0")/../.."
export HOME="${HOME:-$HOME}"
export PYTHONPATH=src
PY="${PY:-.venv_pipeline/bin/python}"
VER="${1:?usage: $0 v1|v2}"

if [[ "$VER" == v1 ]]; then
  RUN=outputs/bounce_pipeline/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k
  MODEL=wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k
  MBASE=wu_ball12_50k_v1
elif [[ "$VER" == v2 ]]; then
  RUN=outputs/bounce_pipeline/wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k_2
  MODEL="wu4dgs_wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter50k_from30k 2"
  MBASE=wu_ball12_50k_v2
else
  echo "unknown version $VER"; exit 1
fi

SRC=outputs/bounce_pipeline/run_50k_from30k/_step4a_stage
SCENE_DIR="$SRC/scene"
GT="$SCENE_DIR/object_poses.csv"

mkdir -p "$RUN/_step4a_stage/export" "$RUN/_step4a_stage/object" "$RUN/_step4a_stage/scene"
cp -R "$SRC/export/." "$RUN/_step4a_stage/export/"
cp -R "$SRC/scene/." "$RUN/_step4a_stage/scene/"
cp "$MODEL/cfg_args" \
   "$MODEL/point_cloud/iteration_50000/point_cloud.ply" \
   "$MODEL/point_cloud/iteration_50000/deformation.pth" \
   "$MODEL/point_cloud/iteration_50000/deformation_table.pth" \
   "$RUN/_step4a_stage/object/"

echo "=== step4a ==="
modal run modal_app.py --upload-step4a --step4a-dir "$RUN/_step4a_stage"
modal run modal_app.py --step4a --bounce-out-rel "$MBASE/step4a"

mkdir -p "$RUN/step4a"
for f in trajectory_smoothed.csv trajectory_raw.csv step4a_meta.json trajectory_plot.png; do
  modal volume get phys4d-gs-output "$MBASE/step4a/$f" "$RUN/step4a/$f" --force
done

echo "=== refit + predict ==="
"$PY" scripts/bounce/refit_metric_gt.py \
  --traj "$RUN/step4a/trajectory_smoothed.csv" \
  --gt-poses "$GT" \
  --out "$RUN/step4b_metric"
"$PY" scripts/bounce/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out "$RUN/step4c/trajectory_predicted.csv" \
  --frame-start 235 --frame-end 360 --anchor-frame 0 --fps 120 --zero-horizontal \
  --gt-poses "$GT"

SCALE=$("$PY" -c "import json; print(json.load(open('$RUN/step4b_metric/refit_metric.json'))['similarity']['scale'])")
CROP=$(python3 -c "print(round(0.17/float('$SCALE'), 3))")
echo "scale=$SCALE crop=$CROP"

mkdir -p "$RUN/_step5_stage/step4a" "$RUN/_step5_stage/step4c"
cp -R "$RUN/_step4a_stage/export" "$RUN/_step5_stage/"
cp -R "$RUN/_step4a_stage/object" "$RUN/_step5_stage/"
cp -R "$RUN/_step4a_stage/scene" "$RUN/_step5_stage/"
cp "$RUN/step4a/trajectory_smoothed.csv" "$RUN/_step5_stage/step4a/"
cp "$RUN/step4c/trajectory_predicted.csv" "$RUN/_step5_stage/step4c/"

modal run modal_app.py --upload-step5 --step5-dir "$RUN/_step5_stage"
modal volume rm phys4d-gs-output step5 -r 2>/dev/null || true
modal run modal_app.py --step5 \
  --canonical-rel object/point_cloud.ply \
  --cfg-args-rel object/cfg_args \
  --bounce-bg-ply-rel bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply \
  --object-scale "$SCALE" \
  --object-crop-radius "$CROP" \
  --object-opacity-boost 4.0 \
  --composite-mode alpha

echo "=== download step5 (volume root step5/) ==="
rm -rf "$RUN/step5"
mkdir -p "$RUN/step5"
modal volume get phys4d-gs-output step5/step5_meta.json "$RUN/step5/step5_meta.json" --force
( cd "$RUN/step5" && modal volume get phys4d-gs-output step5/renders --force )

MPLCONFIGDIR=/tmp/mpl "$PY" scripts/bounce/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted.csv" \
  --gt-poses "$GT" \
  --rendered "$RUN/step5/renders" \
  --gt-rgb-root "$SCENE_DIR/rgb" \
  --out "$RUN/step6" --fps 120 \
  --extracted-traj "$RUN/step4a/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"

mkdir -p "$RUN/step5/_mp4_frames"
for fr in $(seq 235 245); do
  src=$(printf '%04d_cam0.png' "$fr")
  dst=$(printf "$RUN/step5/_mp4_frames/frame%05d.png" $((fr - 235)))
  cp "$RUN/step5/renders/$src" "$dst" 2>/dev/null || true
done
ffmpeg -y -loglevel error -framerate 120 -i "$RUN/step5/_mp4_frames/frame%05d.png" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
  "$RUN/step5/fused_scene_cam00.mp4" 2>/dev/null || echo "mp4 skip"
rm -rf "$RUN/step5/_mp4_frames"

bash scripts/bounce/ping_user.sh
echo "DONE $RUN"
