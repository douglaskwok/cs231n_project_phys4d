#!/usr/bin/env bash
# Full bounce prediction workflow (steps 4a-6) for completed Wu 75k 3x3 grid scenes.
#
# Supported scene ids: 0000, 0001, 0002, 0004, 0005, 0006, 0007, 0008.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src

SCRIPT_START_EPOCH="$(date +%s)"
PHASE_START_EPOCH="$SCRIPT_START_EPOCH"
log_time() {
  local label="$1"
  local now elapsed total
  now="$(date +%s)"
  elapsed=$((now - PHASE_START_EPOCH))
  total=$((now - SCRIPT_START_EPOCH))
  printf '=== timing: %s elapsed=%ss total=%ss at %s ===\n' "$label" "$elapsed" "$total" "$(date)"
  PHASE_START_EPOCH="$now"
}

PY="${PY:-.venv_pipeline/bin/python}"
MODAL="${MODAL:-modal}"
FFMPEG_BIN="${FFMPEG_BIN:-}"
if [[ -z "$FFMPEG_BIN" ]] && command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="$(command -v ffmpeg)"
fi
if [[ -z "$FFMPEG_BIN" ]]; then
  set +e
  FFMPEG_BIN="$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)"
  FFMPEG_STATUS=$?
  set -e
  if [[ "$FFMPEG_STATUS" -ne 0 ]]; then
    FFMPEG_BIN=""
  fi
fi
SCENE_ID="${1:-${SCENE_ID:-0007}}"
ROOT="${ROOT:-dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s}"

case "$SCENE_ID" in
  0000)
    SCENE_NAME="scene_0000_e0p89_am5p0"
    MODEL_NAME="wu_ball12_2s_blue_e89_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=0
    ;;
  0001)
    SCENE_NAME="scene_0001_e0p89_a0p0"
    MODEL_NAME="wu_ball12_2s_blue_e89_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=1
    ;;
  0002)
    SCENE_NAME="scene_0002_e0p89_a5p0"
    MODEL_NAME="wu_ball12_2s_blue_e89_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=0
    ;;
  0004)
    SCENE_NAME="scene_0004_e0p93_a0p0"
    MODEL_NAME="wu_ball12_2s_blue_e93_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k_from50k"
    WU_OUT_OVERRIDE="${WU_OUT_OVERRIDE:-dataset/outputs/wu_debug_ball_marker_all12_table0p40_h0p80_r0p20_120fps_3p0s_bouncy/0601_scene_0000_e0p93_a0p0/4dgs_wu/$MODEL_NAME/wu4dgs_$MODEL_NAME}"
    DEFAULT_ZERO_HORIZONTAL=1
    ;;
  0005)
    SCENE_NAME="scene_0005_e0p93_a5p0"
    MODEL_NAME="wu_ball12_2s_blue_e93_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=0
    ;;
  0006)
    SCENE_NAME="scene_0006_e0p97_am5p0"
    MODEL_NAME="wu_ball12_2s_blue_e97_am5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=0
    ;;
  0007)
    SCENE_NAME="scene_0007_e0p97_a0p0"
    MODEL_NAME="wu_ball12_2s_blue_e97_a0_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=1
    ;;
  0008)
    SCENE_NAME="scene_0008_e0p97_a5p0"
    MODEL_NAME="wu_ball12_2s_blue_e97_a5_fg20_mask1_spill1_area0p1_scaleiso5e-2_den4000_iter75k"
    DEFAULT_ZERO_HORIZONTAL=0
    ;;
  *)
    echo "Unsupported SCENE_ID=$SCENE_ID. Use one of: 0000 0001 0002 0004 0005 0006 0007 0008" >&2
    exit 2
    ;;
esac

SCENE="${SCENE:-$ROOT/$SCENE_NAME}"
WU_OUT="${WU_OUT:-${WU_OUT_OVERRIDE:-$SCENE/4dgs_wu/$MODEL_NAME/wu4dgs_$MODEL_NAME}}"
ITER="${ITER:-75000}"
RUN="${RUN:-outputs/bounce_pipeline/wu75k_${SCENE_NAME}_pred}"
MBASE="${MBASE:-bounce_pred_${SCENE_ID}}"
STEP4A_DATA_REL="${STEP4A_DATA_REL:-step4a_${SCENE_ID}}"
STEP4A_OUT_REL="${STEP4A_OUT_REL:-$MBASE/step4a}"
STEP5_DATA_REL="${STEP5_DATA_REL:-step5_${SCENE_ID}}"
STEP5_OUT_REL="${STEP5_OUT_REL:-step5_${SCENE_ID}}"
ZERO_HORIZONTAL="${ZERO_HORIZONTAL:-$DEFAULT_ZERO_HORIZONTAL}"
ZERO_HORIZONTAL_FLAG=""
if [[ "$ZERO_HORIZONTAL" == "1" || "$ZERO_HORIZONTAL" == "true" ]]; then
  ZERO_HORIZONTAL_FLAG="--zero-horizontal"
fi

# Physics 65/35 split over the full 241-frame sequence.
TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-156}"
TEST_START="${TEST_START:-157}"
TEST_END="${TEST_END:-240}"
FPS="${FPS:-120}"
MODEL_LAST="${MODEL_LAST:-240}"
GT="$SCENE/object_poses.csv"

echo "=== scene $SCENE_ID: $SCENE_NAME ==="
log_time "start scene $SCENE_ID"
echo "RUN=$RUN"
echo "STEP4A_DATA_REL=$STEP4A_DATA_REL STEP4A_OUT_REL=$STEP4A_OUT_REL"
echo "STEP5_DATA_REL=$STEP5_DATA_REL STEP5_OUT_REL=$STEP5_OUT_REL"

echo "=== regenerate DyNeRF export (train 0-$MODEL_LAST for correct t_norm; transforms_test $TEST_START-$TEST_END) ==="
rm -rf "$RUN/_step4a_stage/export"
"$PY" scripts/bounce/make_wu_export.py \
  --scene-dir "$SCENE" \
  --out "$RUN/_step4a_stage/export" \
  --train-end "$MODEL_LAST" --test-start "$TEST_START" --test-end "$TEST_END"
log_time "export"

echo "=== (re)stage object (iter $ITER) + scene ==="
mkdir -p "$RUN/_step4a_stage/object" "$RUN/_step4a_stage/scene"
cp -f "$WU_OUT/point_cloud/iteration_$ITER/point_cloud.ply"      "$RUN/_step4a_stage/object/"
cp -f "$WU_OUT/point_cloud/iteration_$ITER/deformation.pth"      "$RUN/_step4a_stage/object/"
cp -f "$WU_OUT/point_cloud/iteration_$ITER/deformation_table.pth" "$RUN/_step4a_stage/object/"
cp -f "$WU_OUT/cfg_args"                                          "$RUN/_step4a_stage/object/"
cp -f "$GT"                                                       "$RUN/_step4a_stage/scene/"
"$PY" scripts/bounce/sanitize_canonical_ply.py --ply "$RUN/_step4a_stage/object/point_cloud.ply"
log_time "stage object"

echo "=== step4a: upload + run (GPU) ==="
$MODAL run modal_app.py --upload-step4a --step4a-dir "$RUN/_step4a_stage" \
  --step4a-data-rel "$STEP4A_DATA_REL"
log_time "step4a upload"
$MODAL run modal_app.py --step4a --step4a-data-rel "$STEP4A_DATA_REL" \
  --bounce-out-rel "$STEP4A_OUT_REL"
log_time "step4a gpu"

echo "=== download step4a (recursive) ==="
rm -rf "$RUN/step4a"
( cd "$RUN" && $MODAL volume get phys4d-gs-output "$STEP4A_OUT_REL" --force )
log_time "step4a download"

echo "=== refit + predict ==="
"$PY" scripts/bounce/refit_metric_gt.py \
  --traj "$RUN/step4a/trajectory_smoothed.csv" \
  --gt-poses "$GT" \
  --out "$RUN/step4b_metric" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start "$TEST_START" --test-end "$TEST_END"
"$PY" scripts/bounce/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out "$RUN/step4c/trajectory_predicted.csv" \
  --frame-start "$TEST_START" --frame-end "$TEST_END" \
  --anchor-frame "$TRAIN_START" --fps "$FPS" \
  --gt-poses "$GT" \
  $ZERO_HORIZONTAL_FLAG
log_time "refit predict"

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
log_time "step5 upload"
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
log_time "step5 gpu"

echo "=== download step5 (recursive from $STEP5_OUT_REL/) ==="
rm -rf "$RUN/step5"
mkdir -p "$RUN/step5"
$MODAL volume get phys4d-gs-output "$STEP5_OUT_REL/step5_meta.json" "$RUN/step5/step5_meta.json" --force
( cd "$RUN/step5" && $MODAL volume get phys4d-gs-output "$STEP5_OUT_REL/renders" --force )
log_time "step5 download"

echo "=== step6 eval ==="
MPLCONFIGDIR=/tmp/mpl "$PY" scripts/bounce/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted.csv" \
  --gt-poses "$GT" \
  --rendered "$RUN/step5/renders" \
  --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/step6" --fps "$FPS" \
  --extracted-traj "$RUN/step4a/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"
log_time "step6 eval"

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
    if [[ -n "$FFMPEG_BIN" ]]; then
      "$FFMPEG_BIN" -y -loglevel error -framerate "$FPS" -i "$RUN/step5/_mp4_frames/frame%05d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
        "$RUN/step5/fused_scene_$cam.mp4" && echo "wrote fused_scene_$cam.mp4 ($i frames)"
    else
      echo "mp4 skip $cam (ffmpeg not found)"
    fi
  else
    echo "mp4 skip $cam (no frames)"
  fi
  rm -rf "$RUN/step5/_mp4_frames"
done

bash scripts/bounce/ping_user.sh || true
log_time "mp4"
echo "DONE $RUN"
