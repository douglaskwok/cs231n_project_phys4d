#!/usr/bin/env bash
# Rerun only bounce scene 0002 step 5, then download/eval/write MP4s.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src

PY="${PY:-.venv_pipeline/bin/python}"
MODAL="${MODAL:-arch -arm64 modal}"
ROOT="${ROOT:-dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s}"
FPS="${FPS:-120}"
TEST_START="${TEST_START:-157}"
TEST_END="${TEST_END:-240}"

scene="$ROOT/scene_0002_e0p89_a5p0"
run="outputs/bounce_pipeline/wu75k_scene_0002_e0p89_a5p0_pred"
stage="$run/_step5_stage"
step5_rel="${STEP5_REL:-step5_0002}"
scale="${OBJECT_SCALE:-0.5017067866494281}"
crop="${OBJECT_CROP:-0.339}"

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

[[ -f "$stage/export/transforms_test.json" ]] || { echo "missing $stage/export/transforms_test.json"; exit 1; }
[[ -f "$stage/object/point_cloud.ply" ]] || { echo "missing $stage/object/point_cloud.ply"; exit 1; }
[[ -f "$stage/object/cfg_args" ]] || { echo "missing $stage/object/cfg_args"; exit 1; }
[[ -f "$stage/step4c/trajectory_predicted.csv" ]] || { echo "missing $stage/step4c/trajectory_predicted.csv"; exit 1; }

echo "===== 0002 upload staged step5 inputs ====="
$MODAL run modal_app.py --upload-step5 --step5-dir "$stage" --step5-data-rel "$step5_rel"

echo "===== 0002 remote step5 render ====="
$MODAL volume rm phys4d-gs-output "$step5_rel" -r 2>/dev/null || true
$MODAL run modal_app.py --step5 \
  --step5-data-rel "$step5_rel" \
  --step5-out-rel "$step5_rel" \
  --canonical-rel object/point_cloud.ply \
  --cfg-args-rel object/cfg_args \
  --bounce-bg-ply-rel bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply \
  --object-scale "$scale" \
  --object-crop-radius "$crop" \
  --object-opacity-boost 4.0 \
  --composite-mode alpha

echo "===== 0002 download step5 ====="
rm -rf "$run/step5"
mkdir -p "$run/step5"
$MODAL volume get phys4d-gs-output "$step5_rel/step5_meta.json" "$run/step5/step5_meta.json" --force
( cd "$run/step5" && $MODAL volume get phys4d-gs-output "$step5_rel/renders" --force )

echo "===== 0002 step6 eval ====="
MPLCONFIGDIR=/tmp/mpl "$PY" scripts/bounce/step6_eval.py \
  --predicted "$run/step4c/trajectory_predicted.csv" \
  --gt-poses "$scene/object_poses.csv" \
  --rendered "$run/step5/renders" \
  --gt-rgb-root "$scene/rgb" \
  --out "$run/step6" --fps "$FPS" \
  --extracted-traj "$run/step4a/trajectory_smoothed.csv" \
  --refit-metric "$run/step4b_metric/refit_metric.json"

echo "===== 0002 mp4s ====="
for cam in cam10 cam11; do
  mkdir -p "$run/step5/_mp4_frames"
  i=0
  for fr in $(seq "$TEST_START" "$TEST_END"); do
    src=$(printf '%04d_%s.png' "$fr" "$cam")
    dst=$(printf "$run/step5/_mp4_frames/frame%05d.png" "$i")
    if cp "$run/step5/renders/$src" "$dst" 2>/dev/null; then
      i=$((i + 1))
    fi
  done
  if [[ "$i" -gt 0 && -n "$FFMPEG_BIN" ]]; then
    "$FFMPEG_BIN" -y -loglevel error -framerate "$FPS" -i "$run/step5/_mp4_frames/frame%05d.png" \
      -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
      "$run/step5/fused_scene_$cam.mp4"
    echo "wrote $run/step5/fused_scene_$cam.mp4 ($i frames)"
  else
    echo "mp4 skip $cam (frames=$i ffmpeg=${FFMPEG_BIN:-missing})"
  fi
  rm -rf "$run/step5/_mp4_frames"
done

echo "===== DONE 0002 ====="
bash scripts/bounce/ping_user.sh || true
