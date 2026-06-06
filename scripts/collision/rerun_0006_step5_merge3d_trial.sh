#!/usr/bin/env bash
# Rerun only collision 0006 Step 5 using same-coordinate 3D fusion.
#
# This keeps the existing prediction run intact and writes the trial render to
# outputs/.../step5_merge3d_boost6 by default.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH=src
export MODAL_PROFILE="${MODAL_PROFILE:-simon}"

PY="${PY:-.venv_pipeline/bin/python}"
MODAL="${MODAL:-arch -arm64 modal}"
read -r -a MODAL_CMD <<< "$MODAL"
RUN="${RUN:-outputs/collision_pipeline/scene_0006_col_m440_equal105_r098_event20}"
SCENE="${SCENE:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps/scene_0006_col_m440_equal105_r098}"
BG_NAME="${BG_NAME:-collision_room_med}"
BG_PLY_REL="bg_${BG_NAME}_3dgs/point_cloud/iteration_30000/point_cloud.ply"
OUT_NAME="${OUT_NAME:-step5_merge3d_boost6}"
EVAL_NAME="${EVAL_NAME:-step6_merge3d_boost6}"
OBJECT_OPACITY_BOOST="${OBJECT_OPACITY_BOOST:-6.0}"
FPS="${FPS:-60}"

STAGE="$RUN/_step5_stage"
if [[ ! -f "$STAGE/export/transforms_test.json" ]]; then
  echo "missing staged Step 5 export: $STAGE/export/transforms_test.json" >&2
  exit 2
fi
if [[ ! -f "$RUN/step5/step5_meta.json" ]]; then
  echo "missing baseline Step 5 metadata: $RUN/step5/step5_meta.json" >&2
  exit 2
fi

read -r SCALE0 SCALE1 < <(
  "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(*m["object_scales"])' \
    "$RUN/step5/step5_meta.json"
)

echo "=== collision 0006 merge3d trial ==="
echo "run=$RUN"
echo "out=$RUN/$OUT_NAME"
echo "scales=$SCALE0,$SCALE1 object_opacity_boost=$OBJECT_OPACITY_BOOST"

"${MODAL_CMD[@]}" volume rm phys4d-gs-data step5 -r 2>/dev/null || true
"${MODAL_CMD[@]}" volume put phys4d-gs-data "$STAGE" step5
"${MODAL_CMD[@]}" volume rm phys4d-gs-output step5 -r 2>/dev/null || true
"${MODAL_CMD[@]}" run modal_app.py --collision-step5 \
  --collision-bg-ply-rel "$BG_PLY_REL" \
  --collision-object-scale "$SCALE0,$SCALE1" \
  --collision-object-opacity-boost "$OBJECT_OPACITY_BOOST" \
  --collision-object-crop-box-half-extents "0.21,0.21,0.13125" \
  --collision-composite-mode merge3d

echo "=== download trial Step 5 ==="
rm -rf "$RUN/$OUT_NAME"
mkdir -p "$RUN/$OUT_NAME"
"${MODAL_CMD[@]}" volume get phys4d-gs-output step5/step5_meta.json "$RUN/$OUT_NAME/step5_meta.json" --force
( cd "$RUN/$OUT_NAME" && "${MODAL_CMD[@]}" volume get phys4d-gs-output step5/renders --force )

echo "=== evaluate trial render ==="
"$PY" scripts/collision/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted_obj0.csv,$RUN/step4c/trajectory_predicted_obj1.csv" \
  --gt-poses "$RUN/_poses/object_poses_obj0.csv,$RUN/_poses/object_poses_obj1.csv" \
  --rendered "$RUN/$OUT_NAME/renders" \
  --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/$EVAL_NAME" --fps "$FPS" \
  --extracted-traj "$RUN/step4a/obj0/trajectory_smoothed.csv,$RUN/step4a/obj1/trajectory_smoothed.csv" \
  --refit-metric "$RUN/step4b_metric/refit_metric.json"

FFMPEG_BIN="${FFMPEG_BIN:-}"
if [[ -z "$FFMPEG_BIN" ]] && command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="$(command -v ffmpeg)"
fi
if [[ -n "$FFMPEG_BIN" ]]; then
  for cam in cam10 cam11; do
    rm -rf "$RUN/$OUT_NAME/_mp4_frames"
    mkdir -p "$RUN/$OUT_NAME/_mp4_frames"
    i=0
    for src in $(find "$RUN/$OUT_NAME/renders" -name "*_${cam}.png" | sort); do
      cp "$src" "$RUN/$OUT_NAME/_mp4_frames/frame$(printf '%05d' "$i").png"
      i=$((i + 1))
    done
    if [[ "$i" -gt 0 ]]; then
      "$FFMPEG_BIN" -y -loglevel error -framerate "$FPS" \
        -i "$RUN/$OUT_NAME/_mp4_frames/frame%05d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
        "$RUN/$OUT_NAME/fused_scene_${cam}.mp4"
      echo "wrote $RUN/$OUT_NAME/fused_scene_${cam}.mp4 ($i frames)"
    fi
  done
else
  echo "ffmpeg not found; skipped mp4 generation"
fi

echo "done: $RUN/$OUT_NAME"
