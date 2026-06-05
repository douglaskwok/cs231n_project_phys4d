#!/usr/bin/env bash
# Full workflow.md (steps 4a-6) for wu75k_scene_0003_e0p93_am5p0_fullcopy.
# 65/35 train/test split over the full 241-frame sequence: train 0-156, test 157-240.
# Angled drop (ball_angle_deg=-5) -> keep horizontal velocity (NO --zero-horizontal).
# Run from YOUR terminal so the modal progress bars are visible.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
PY="${PY:-.venv_pipeline/bin/python}"

SCENE="${SCENE:-wu75k_scene_0003_e0p93_am5p0_fullcopy}"
WU_OUT="${WU_OUT:-$SCENE/4dgs_wu/wu_ball12_2s_blue_e93_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k/wu4dgs_wu_ball12_2s_blue_e93_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k}"
ITER="${ITER:-75000}"
RUN="${RUN:-outputs/bounce_pipeline/$(basename "$SCENE")}"
MBASE="${MBASE:-wu75k_scene0003}"
MODAL="${MODAL:-modal}"
STEP4A_DATA_REL="${STEP4A_DATA_REL:-${MBASE}_step4a_input}"
STEP4A_OUT_REL="${STEP4A_OUT_REL:-$MBASE/step4a}"
STEP5_DATA_REL="${STEP5_DATA_REL:-${MBASE}_step5_input}"
STEP5_OUT_REL="${STEP5_OUT_REL:-${MBASE}_step5}"

# Physics 65/35 split (applied in refit/predict only).
TRAIN_START=0
TRAIN_END=156
TEST_START=157
TEST_END=240
FPS=120
# The Wu model was trained over the full 2.0s clip, so the export's train window
# (which sets t_norm AND the step4a extraction range) must span ALL frames 0-240.
# t_norm = frame/MODEL_LAST must match training; do NOT shrink it to the 65/35 split.
MODEL_LAST=240

GT="$SCENE/object_poses.csv"

echo "=== regenerate DyNeRF export (train 0-$MODEL_LAST for correct t_norm; transforms_test $TEST_START-$TEST_END) ==="
rm -rf "$RUN/_step4a_stage/export"
"$PY" scripts/bounce/make_wu_export.py \
  --scene-dir "$SCENE" \
  --out "$RUN/_step4a_stage/export" \
  --train-end "$MODEL_LAST" --test-start "$TEST_START" --test-end "$TEST_END"

echo "=== (re)stage object (iter $ITER) + scene ==="
mkdir -p "$RUN/_step4a_stage/object" "$RUN/_step4a_stage/scene"
cp -f "$WU_OUT/point_cloud/iteration_$ITER/point_cloud.ply"      "$RUN/_step4a_stage/object/"
cp -f "$WU_OUT/point_cloud/iteration_$ITER/deformation.pth"      "$RUN/_step4a_stage/object/"
cp -f "$WU_OUT/point_cloud/iteration_$ITER/deformation_table.pth" "$RUN/_step4a_stage/object/"
cp -f "$WU_OUT/cfg_args"                                          "$RUN/_step4a_stage/object/"
cp -f "$GT"                                                       "$RUN/_step4a_stage/scene/"
# iter75k canonical has a single non-finite Gaussian; neutralize it in place
# (keeps per-Gaussian indexing aligned with deformation.pth).
"$PY" scripts/bounce/sanitize_canonical_ply.py --ply "$RUN/_step4a_stage/object/point_cloud.ply"

echo "=== step4a: upload + run (GPU) ==="
$MODAL run modal_app.py --upload-step4a --step4a-dir "$RUN/_step4a_stage" \
  --step4a-data-rel "$STEP4A_DATA_REL"
$MODAL run modal_app.py --step4a --step4a-data-rel "$STEP4A_DATA_REL" \
  --bounce-out-rel "$STEP4A_OUT_REL"

echo "=== download step4a (recursive) ==="
rm -rf "$RUN/step4a"
( cd "$RUN" && $MODAL volume get phys4d-gs-output "$STEP4A_OUT_REL" --force )

echo "=== refit (train 0-156 / test 157-240) + predict (157-240, keep horizontal) ==="
"$PY" scripts/bounce/refit_metric_gt.py \
  --traj "$RUN/step4a/trajectory_smoothed.csv" \
  --gt-poses "$GT" \
  --out "$RUN/step4b_metric" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start "$TEST_START" --test-end "$TEST_END"
"$PY" scripts/bounce/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out "$RUN/step4c/trajectory_predicted.csv" \
  --frame-start "$TEST_START" --frame-end "$TEST_END" --anchor-frame "$TRAIN_START" --fps "$FPS" \
  --gt-poses "$GT"

SCALE=$("$PY" -c "import json;print(json.load(open('$RUN/step4b_metric/refit_metric.json'))['similarity']['scale'])")
CROP=$("$PY" -c "print(round(0.17/float('$SCALE'), 3))")
echo "scale=$SCALE crop=$CROP"

echo "=== step5: stage + run (GPU) ==="
rm -rf "$RUN/_step5_stage"
mkdir -p "$RUN/_step5_stage/step4a" "$RUN/_step5_stage/step4c"
cp -R "$RUN/_step4a_stage/export" "$RUN/_step5_stage/"
cp -R "$RUN/_step4a_stage/object" "$RUN/_step5_stage/"
cp -R "$RUN/_step4a_stage/scene"  "$RUN/_step5_stage/"
cp "$RUN/step4a/trajectory_smoothed.csv" "$RUN/_step5_stage/step4a/"
cp "$RUN/step4c/trajectory_predicted.csv" "$RUN/_step5_stage/step4c/"
$MODAL run modal_app.py --upload-step5 --step5-dir "$RUN/_step5_stage" \
  --step5-data-rel "$STEP5_DATA_REL"
$MODAL volume rm phys4d-gs-output "$STEP5_OUT_REL" -r 2>/dev/null || true
$MODAL run modal_app.py --step5 \
  --step5-data-rel "$STEP5_DATA_REL" \
  --step5-out-rel "$STEP5_OUT_REL" \
  --canonical-rel object/point_cloud.ply \
  --cfg-args-rel object/cfg_args \
  --bounce-bg-ply-rel bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply \
  --object-scale "$SCALE" \
  --object-crop-radius "$CROP" \
  --object-opacity-boost 4.0 \
  --composite-mode alpha

echo "=== download step5 (recursive from $STEP5_OUT_REL/) ==="
rm -rf "$RUN/step5"
mkdir -p "$RUN/step5"
$MODAL volume get phys4d-gs-output "$STEP5_OUT_REL/step5_meta.json" "$RUN/step5/step5_meta.json" --force
( cd "$RUN/step5" && $MODAL volume get phys4d-gs-output "$STEP5_OUT_REL/renders" --force )

echo "=== step6 eval ==="
MPLCONFIGDIR=/tmp/mpl "$PY" scripts/bounce/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted.csv" \
  --gt-poses "$GT" \
  --rendered "$RUN/step5/renders" \
  --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/step6" --fps "$FPS" \
  --extracted-traj "$RUN/step4a/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"

echo "=== fused mp4 (test cams 10 & 11) ==="
for cam in cam10 cam11; do
  mkdir -p "$RUN/step5/_mp4_frames"
  i=0
  for fr in $(seq "$TEST_START" "$TEST_END"); do
    src=$(printf '%04d_%s.png' "$fr" "$cam")
    dst=$(printf "$RUN/step5/_mp4_frames/frame%05d.png" "$i")
    if cp "$RUN/step5/renders/$src" "$dst" 2>/dev/null; then i=$((i+1)); fi
  done
  if [[ "$i" -gt 0 ]]; then
    ffmpeg -y -loglevel error -framerate "$FPS" -i "$RUN/step5/_mp4_frames/frame%05d.png" \
      -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
      "$RUN/step5/fused_scene_$cam.mp4" && echo "wrote fused_scene_$cam.mp4 ($i frames)"
  else
    echo "mp4 skip $cam (no frames)"
  fi
  rm -rf "$RUN/step5/_mp4_frames"
done

bash scripts/bounce/ping_user.sh || true
echo "DONE $RUN"
