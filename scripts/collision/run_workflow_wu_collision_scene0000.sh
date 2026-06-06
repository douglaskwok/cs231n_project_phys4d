#!/usr/bin/env bash
# Full collision workflow.md (steps 4a-6) for
# wu_collision_scale1p75_v1p4_sidecams_60fps_scene_0000 (two objects, 60 fps, 157 frames).
# Event-aligned split: hold out 20 frames around the second object-object impact.
# Run from YOUR terminal so Modal progress bars are visible.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
PY="${PY:-.venv_pipeline/bin/python}"

SCENE="${SCENE:-dataset/outputs/phys4d_final/collision_scale1p75_v1p4_sidecams_60fps/scene_0000_collision_room}"
WU_A="${WU_A:-$SCENE/4dgs_wu/wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_a_compactA_bg2_area0p2_iso0p05_from50k_to60k}"
WU_B="${WU_B:-$SCENE/4dgs_wu/wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k/wu4dgs_wu_collision_scale1p75_v1p4_object_b_compactA_bg2_area0p2_iso0p05_from50k_to60k}"
ITER="${ITER:-60000}"
ITER_A="${ITER_A:-$ITER}"
ITER_B="${ITER_B:-$ITER}"
RUN="${RUN:-outputs/collision_pipeline/$(basename "$SCENE")}"
MBASE="${MBASE:-wu_collision_scene0000}"
BG_NAME="${BG_NAME:-collision_room_med}"
BG_PLY_REL="bg_${BG_NAME}_3dgs/point_cloud/iteration_30000/point_cloud.ply"
COLLISION_OBJECT_OPACITY_BOOST="${COLLISION_OBJECT_OPACITY_BOOST:-4.0}"
COLLISION_COMPOSITE_MODE="${COLLISION_COMPOSITE_MODE:-alpha}"
COLLISION_COMPOSITE_THRESHOLD="${COLLISION_COMPOSITE_THRESHOLD:-32}"
COLLISION_COMPOSITE_ALPHA_GAMMA="${COLLISION_COMPOSITE_ALPHA_GAMMA:-1.0}"
MODAL="${MODAL:-modal}"
export MODAL_CLI="${MODAL_CLI:-$MODAL}"
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

FPS="${FPS:-60}"
MODEL_LAST="${MODEL_LAST:-156}"

if [[ -z "${TRAIN_START:-}" || -z "${TRAIN_END:-}" || -z "${TEST_START:-}" || -z "${TEST_END:-}" ]]; then
  SPLIT_VARS=$("$PY" scripts/collision/choose_prediction_split.py "$SCENE" \
    --pre-collision-frames "${PRE_COLLISION_FRAMES:-5}" \
    --post-collision-frames "${POST_COLLISION_FRAMES:-14}")
  eval "$SPLIT_VARS"
fi
TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-111}"
TEST_START="${TEST_START:-112}"
TEST_END="${TEST_END:-131}"

GT0="$RUN/_poses/object_poses_obj0.csv"
GT1="$RUN/_poses/object_poses_obj1.csv"
GT_PAIR="$GT0,$GT1"

echo "=== split combined object_poses.csv per object ==="
mkdir -p "$RUN/_poses"
"$PY" - "$SCENE" "$RUN" <<'PY'
import csv
import sys
from pathlib import Path

scene = Path(sys.argv[1])
run = Path(sys.argv[2]) / "_poses"
rows = list(csv.DictReader(open(scene / "object_poses.csv", encoding="utf-8")))
fieldnames = rows[0].keys()
for k in (0, 1):
    sub = [r for r in rows if int(float(r["object_index"])) == k]
    out = run / f"object_poses_obj{k}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sub)
    print(f"wrote {out} ({len(sub)} rows)")
PY

echo "=== regenerate DyNeRF export (train 0-$MODEL_LAST; transforms_test $TEST_START-$TEST_END) ==="
rm -rf "$RUN/_step4a_stage/export"
"$PY" scripts/bounce/make_wu_export.py \
  --scene-dir "$SCENE" \
  --out "$RUN/_step4a_stage/export" \
  --train-end "$MODEL_LAST" --test-start "$TEST_START" --test-end "$TEST_END"

echo "=== stage both objects (obj0 iter $ITER_A, obj1 iter $ITER_B) + per-object GT poses ==="
for k in 0 1; do
  WU="$WU_A"
  OBJ_ITER="$ITER_A"
  if [[ "$k" == "1" ]]; then
    WU="$WU_B"
    OBJ_ITER="$ITER_B"
  fi
  mkdir -p "$RUN/_step4a_stage/obj$k" "$RUN/_step4a_stage/scene"
  cp -f "$WU/point_cloud/iteration_$OBJ_ITER/point_cloud.ply"       "$RUN/_step4a_stage/obj$k/"
  cp -f "$WU/point_cloud/iteration_$OBJ_ITER/deformation.pth"       "$RUN/_step4a_stage/obj$k/"
  cp -f "$WU/point_cloud/iteration_$OBJ_ITER/deformation_table.pth" "$RUN/_step4a_stage/obj$k/"
  cp -f "$WU/cfg_args"                                              "$RUN/_step4a_stage/obj$k/"
  "$PY" scripts/bounce/sanitize_canonical_ply.py --ply "$RUN/_step4a_stage/obj$k/point_cloud.ply"
done
cp -f "$GT0" "$RUN/_step4a_stage/scene/"
cp -f "$GT1" "$RUN/_step4a_stage/scene/"

echo "=== step4a: upload + run (GPU) ==="
$MODAL volume rm phys4d-gs-data step4a -r 2>/dev/null || true
$MODAL volume put phys4d-gs-data "$RUN/_step4a_stage" step4a
$MODAL run modal_app.py --collision-step4a --collision-out-rel "$MBASE/step4a"

echo "=== download step4a ==="
rm -rf "$RUN/step4a"
( cd "$RUN" && $MODAL volume get phys4d-gs-output "$MBASE/step4a" --force )

echo "=== refit + predict (train $TRAIN_START-$TRAIN_END / test $TEST_START-$TEST_END) ==="
"$PY" scripts/collision/refit_metric_gt.py \
  --traj "$RUN/step4a/obj0/trajectory_smoothed.csv,$RUN/step4a/obj1/trajectory_smoothed.csv" \
  --gt-poses "$GT_PAIR" \
  --config "$SCENE/config.json" \
  --out "$RUN/step4b_metric" \
  --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
  --test-start "$TEST_START" --test-end "$TEST_END" \
  --fit-mass-ratio --fit-walls
"$PY" scripts/collision/predict_metric_from_refit.py \
  --refit "$RUN/step4b_metric/refit_metric.json" \
  --out-dir "$RUN/step4c" \
  --frame-start "$TEST_START" --frame-end "$TEST_END" --anchor-frame "$TRAIN_START" --fps "$FPS" \
  --gt-poses "$GT_PAIR"

read -r SCALE0 SCALE1 CROP0 CROP1 < <(
  "$PY" -c "
import json
r = json.load(open('$RUN/step4b_metric/refit_metric.json'))
s0, s1 = r['similarity'][0]['scale'], r['similarity'][1]['scale']
# box half-extent 0.21 m in XY
print(s0, s1, round(0.21/s0, 3), round(0.21/s1, 3))
"
)
echo "scale0=$SCALE0 scale1=$SCALE1 crop0=$CROP0 crop1=$CROP1"

echo "=== background: union object masks + export + train (if missing on volume) ==="
BG_SCENE="$RUN/_bg_scene"
rm -rf "$BG_SCENE"
mkdir -p "$BG_SCENE/masks"
cp "$SCENE/config.json" "$SCENE/cameras.json" "$BG_SCENE/"
ln -sfn "$(pwd)/$SCENE/rgb" "$BG_SCENE/rgb"
"$PY" - "$SCENE" "$BG_SCENE/masks" <<'PY'
import imageio.v2 as imageio
import numpy as np
import sys
from pathlib import Path

scene = Path(sys.argv[1])
out = Path(sys.argv[2])
for cam_a in sorted((scene / "masks_object_a").glob("cam*")):
    cam = cam_a.name
    cam_b = scene / "masks_object_b" / cam
    out_cam = out / cam
    out_cam.mkdir(parents=True, exist_ok=True)
    for ma in sorted(cam_a.glob("frame*.png")):
        mb = cam_b / ma.name
        a = np.asarray(imageio.imread(ma))
        b = np.asarray(imageio.imread(mb)) if mb.is_file() else 0
        if a.ndim == 3:
            a = a[..., 0]
        if isinstance(b, np.ndarray) and b.ndim == 3:
            b = b[..., 0]
        union = ((a > 0) | (b > 0)).astype(np.uint8) * 255
        imageio.imwrite(out_cam / ma.name, union)
print(f"wrote union masks -> {out}")
PY
"$PY" scripts/bounce/make_background_export.py \
  --scene-dir "$BG_SCENE" \
  --out "$RUN/_bg_export" \
  --inpaint temporal-median \
  --cameras all --frame-start 0 --frame-end "$MODEL_LAST" --frame-stride 3
if ! $MODAL volume ls phys4d-gs-output "$BG_PLY_REL" >/dev/null 2>&1; then
  $MODAL run modal_app.py --upload-bg --bg-dir "$RUN/_bg_export" --bg-name "$BG_NAME"
  $MODAL run modal_app.py --train-bg-job --bg-name "$BG_NAME" --iterations 30000
else
  echo "background PLY already on volume: $BG_PLY_REL"
fi

echo "=== step5: stage + run (GPU) ==="
rm -rf "$RUN/_step5_stage"
mkdir -p "$RUN/_step5_stage/step4a/obj0" "$RUN/_step5_stage/step4a/obj1" "$RUN/_step5_stage/step4c" "$RUN/_step5_stage/scene" "$RUN/_step5_stage/obj0" "$RUN/_step5_stage/obj1"
cp -R "$RUN/_step4a_stage/export" "$RUN/_step5_stage/"
for k in 0 1; do
  cp -f "$RUN/_step4a_stage/obj$k/point_cloud.ply" "$RUN/_step5_stage/obj$k/"
  cp -f "$RUN/_step4a_stage/obj$k/cfg_args"       "$RUN/_step5_stage/obj$k/"
  cp "$RUN/step4a/obj$k/trajectory_smoothed.csv" "$RUN/_step5_stage/step4a/obj$k/"
  cp "$RUN/step4c/trajectory_predicted_obj$k.csv" "$RUN/_step5_stage/step4c/"
done
cp -f "$GT0" "$GT1" "$RUN/_step5_stage/scene/"
$MODAL volume rm phys4d-gs-data step5 -r 2>/dev/null || true
$MODAL volume put phys4d-gs-data "$RUN/_step5_stage" step5
$MODAL volume rm phys4d-gs-output step5 -r 2>/dev/null || true
$MODAL run modal_app.py --collision-step5 \
  --collision-bg-ply-rel "$BG_PLY_REL" \
  --collision-object-scale "$SCALE0,$SCALE1" \
  --collision-object-opacity-boost "$COLLISION_OBJECT_OPACITY_BOOST" \
  --collision-object-crop-box-half-extents "0.21,0.21,0.13125" \
  --collision-composite-mode "$COLLISION_COMPOSITE_MODE" \
  --collision-composite-threshold "$COLLISION_COMPOSITE_THRESHOLD" \
  --collision-composite-alpha-gamma "$COLLISION_COMPOSITE_ALPHA_GAMMA"

echo "=== download step5 ==="
rm -rf "$RUN/step5"
mkdir -p "$RUN/step5"
$MODAL volume get phys4d-gs-output step5/step5_meta.json "$RUN/step5/step5_meta.json" --force
( cd "$RUN/step5" && $MODAL volume get phys4d-gs-output step5/renders --force )

echo "=== step6 eval ==="
MPLCONFIGDIR=/tmp/mpl "$PY" scripts/collision/step6_eval.py \
  --predicted "$RUN/step4c/trajectory_predicted_obj0.csv,$RUN/step4c/trajectory_predicted_obj1.csv" \
  --gt-poses "$GT_PAIR" \
  --rendered "$RUN/step5/renders" \
  --gt-rgb-root "$SCENE/rgb" \
  --out "$RUN/step6" --fps "$FPS" \
  --extracted-traj "$RUN/step4a/obj0/trajectory_smoothed.csv,$RUN/step4a/obj1/trajectory_smoothed.csv" \
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
echo "DONE $RUN"
