#!/usr/bin/env bash
# Resume/recover bounce prediction runs that already completed remote work.
#
# - 0008: remote step5 exists; download, eval, and write MP4s.
# - 0002: remote step4a exists; download, refit/predict, render step5, eval, MP4s.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src

PY="${PY:-.venv_pipeline/bin/python}"
MODAL="${MODAL:-arch -arm64 modal}"
ROOT="${ROOT:-dataset/outputs/wu_ball_3x3_e89_e93_e97_angle5_table0p40_h0p80_r0p20_120fps_2p0s}"
FPS="${FPS:-120}"
TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-156}"
TEST_START="${TEST_START:-157}"
TEST_END="${TEST_END:-240}"
MODEL_LAST="${MODEL_LAST:-240}"

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

download_step5() {
  local run="$1"
  local step5_rel="$2"
  rm -rf "$run/step5"
  mkdir -p "$run/step5"
  $MODAL volume get phys4d-gs-output "$step5_rel/step5_meta.json" "$run/step5/step5_meta.json" --force
  ( cd "$run/step5" && $MODAL volume get phys4d-gs-output "$step5_rel/renders" --force )
}

eval_and_mp4() {
  local scene="$1"
  local run="$2"
  local gt="$scene/object_poses.csv"

  MPLCONFIGDIR=/tmp/mpl "$PY" scripts/bounce/step6_eval.py \
    --predicted "$run/step4c/trajectory_predicted.csv" \
    --gt-poses "$gt" \
    --rendered "$run/step5/renders" \
    --gt-rgb-root "$scene/rgb" \
    --out "$run/step6" --fps "$FPS" \
    --extracted-traj "$run/step4a/trajectory_smoothed.csv" \
    --refit-metric "$run/step4b_metric/refit_metric.json"

  for cam in cam10 cam11; do
    mkdir -p "$run/step5/_mp4_frames"
    local i=0
    for fr in $(seq "$TEST_START" "$TEST_END"); do
      local src dst
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
}

resume_0008() {
  echo "===== RECOVER 0008: download remote step5, eval, mp4 ====="
  local scene="$ROOT/scene_0008_e0p97_a5p0"
  local run="outputs/bounce_pipeline/wu75k_scene_0008_e0p97_a5p0_pred"
  download_step5 "$run" "step5_0008"
  eval_and_mp4 "$scene" "$run"
  echo "===== DONE 0008 ====="
}

resume_0002() {
  echo "===== RESUME 0002: download step4a, refit, render, eval, mp4 ====="
  local scene="$ROOT/scene_0002_e0p89_a5p0"
  local run="outputs/bounce_pipeline/wu75k_scene_0002_e0p89_a5p0_pred"
  local gt="$scene/object_poses.csv"
  local step4a_rel="bounce_pred_0002/step4a"
  local step5_data_rel="step5_0002"
  local step5_out_rel="step5_0002"

  rm -rf "$run/step4a"
  mkdir -p "$run"
  ( cd "$run" && $MODAL volume get phys4d-gs-output "$step4a_rel" --force )

  "$PY" scripts/bounce/refit_metric_gt.py \
    --traj "$run/step4a/trajectory_smoothed.csv" \
    --gt-poses "$gt" \
    --out "$run/step4b_metric" \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --test-start "$TEST_START" --test-end "$TEST_END"
  "$PY" scripts/bounce/predict_metric_from_refit.py \
    --refit "$run/step4b_metric/refit_metric.json" \
    --out "$run/step4c/trajectory_predicted.csv" \
    --frame-start "$TEST_START" --frame-end "$TEST_END" \
    --anchor-frame "$TRAIN_START" --fps "$FPS" \
    --gt-poses "$gt"

  local scale crop
  scale=$("$PY" -c "import json;print(json.load(open('$run/step4b_metric/refit_metric.json'))['similarity']['scale'])")
  crop=$("$PY" -c "print(round(0.17/float('$scale'), 3))")
  echo "scale=$scale crop=$crop"

  rm -rf "$run/_step5_stage"
  mkdir -p "$run/_step5_stage/step4a" "$run/_step5_stage/step4c"
  cp -R "$run/_step4a_stage/export" "$run/_step5_stage/"
  cp -R "$run/_step4a_stage/object" "$run/_step5_stage/"
  cp -R "$run/_step4a_stage/scene" "$run/_step5_stage/"
  cp "$run/step4a/trajectory_smoothed.csv" "$run/_step5_stage/step4a/"
  cp "$run/step4c/trajectory_predicted.csv" "$run/_step5_stage/step4c/"

  $MODAL run modal_app.py --upload-step5 --step5-dir "$run/_step5_stage" \
    --step5-data-rel "$step5_data_rel"
  $MODAL volume rm phys4d-gs-output "$step5_out_rel" -r 2>/dev/null || true
  $MODAL run modal_app.py --step5 \
    --step5-data-rel "$step5_data_rel" \
    --step5-out-rel "$step5_out_rel" \
    --canonical-rel object/point_cloud.ply \
    --cfg-args-rel object/cfg_args \
    --bounce-bg-ply-rel bg_ball12blue_med_3dgs/point_cloud/iteration_30000/point_cloud.ply \
    --object-scale "$scale" \
    --object-crop-radius "$crop" \
    --object-opacity-boost 4.0 \
    --composite-mode alpha

  download_step5 "$run" "$step5_out_rel"
  eval_and_mp4 "$scene" "$run"
  echo "===== DONE 0002 ====="
}

resume_0008
resume_0002
bash scripts/bounce/ping_user.sh || true
