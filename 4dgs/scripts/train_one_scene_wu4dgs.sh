#!/usr/bin/env bash
# Export one generated Phys4D scene to DyNeRF, train hustvl/4DGaussians
# (Wu et al.) on Modal, optionally render with Wu's native renderer, and
# download the resulting model folder.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-phys_sim/bin/python}"
MODAL_CMD="${MODAL_CMD:-arch -arm64 modal}"

SCENE_DIR=""
RUN_NAME=""
MASK_SUBDIR="masks"
MODE="object"
BACKGROUND="black"
EXPORT_DIR=""
DOWNLOAD_DIR=""
MODEL_REL=""
ITERATIONS="15000"
COARSE_ITERATIONS="3000"
TIME_RESOLUTION="75"
BOUNDS="1.6"
ALL_TRAIN="1"
DROP_INVISIBLE_FRAMES="0"
TRIM_EMPTY_TIME_ENDS="0"
MIN_VISIBLE_CAMERAS="1"
MIN_MASK_PIXELS="0"
SKIP_EXPORT="0"
SKIP_UPLOAD="0"
SKIP_TRAIN="0"
RENDER="0"

usage() {
  cat <<'EOF'
Usage:
  bash 4dgs/scripts/train_one_scene_wu4dgs.sh SCENE_DIR RUN_NAME [options]

This trains the Wu/HUSTVL 4DGaussians implementation on the same DyNeRF export
format used by our current 4DGS pipeline.

Required:
  SCENE_DIR      Generated scene directory containing config.json, rgb/, masks/.
  RUN_NAME       Stable local name, e.g. ball_drop_h0p55_e0p90_a0p0_wu_object.

Options:
  --mask-subdir NAME       Mask folder inside SCENE_DIR. Default: masks.
  --mode MODE              object, background, or full. Default: object.
  --background COLOR       black or white for masked-out pixels. Default: black.
  --iterations N           Wu fine iterations. Default: 15000.
  --coarse-iterations N    Wu coarse iterations. Default: 3000.
  --time-resolution N      HexPlane temporal grid resolution. Default: 75.
  --bounds X               Wu scene bounds. Default: 1.6.
  --render                 Run Wu render.py after training.
  --no-all-train           Use scene train/test split instead of all 12 cameras.
  --drop-invisible-frames  Drop timestamps weakly visible across cameras.
  --trim-empty-time-ends   Trim only leading/trailing weak timestamps.
  --min-visible-cameras N  Required visible cameras for temporal filtering.
  --min-mask-pixels N      Treat masks with <= N pixels as invisible.
  --skip-export            Reuse the existing DyNeRF export.
  --skip-upload            Reuse existing phys4d-gs-data:/4d_scene.
  --skip-train             Only download/render an existing Wu model.
  -h, --help               Show this help.

Outputs:
  4dgs/experiments/wu4dgs/runs/<RUN_NAME>/       DyNeRF export
  <SCENE_DIR>/4dgs_wu/<RUN_NAME>/                downloaded Wu model folder

Example:
  bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
    dataset/outputs/phys4d_final/ball_drop_h0p55_single_60fps/scene_0000_e0p90_a0p0 \
    ball_drop_h0p55_e0p90_a0p0_wu_object \
    --render
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

sanitize_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", sys.argv[1]).strip("_"))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mask-subdir)
      MASK_SUBDIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --background)
      BACKGROUND="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --coarse-iterations)
      COARSE_ITERATIONS="$2"
      shift 2
      ;;
    --time-resolution)
      TIME_RESOLUTION="$2"
      shift 2
      ;;
    --bounds)
      BOUNDS="$2"
      shift 2
      ;;
    --render)
      RENDER="1"
      shift
      ;;
    --no-all-train)
      ALL_TRAIN="0"
      shift
      ;;
    --drop-invisible-frames)
      DROP_INVISIBLE_FRAMES="1"
      shift
      ;;
    --trim-empty-time-ends)
      TRIM_EMPTY_TIME_ENDS="1"
      shift
      ;;
    --min-visible-cameras)
      MIN_VISIBLE_CAMERAS="$2"
      shift 2
      ;;
    --min-mask-pixels)
      MIN_MASK_PIXELS="$2"
      shift 2
      ;;
    --skip-export)
      SKIP_EXPORT="1"
      shift
      ;;
    --skip-upload)
      SKIP_UPLOAD="1"
      shift
      ;;
    --skip-train)
      SKIP_TRAIN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      if [[ -z "$SCENE_DIR" ]]; then
        SCENE_DIR="$1"
      elif [[ -z "$RUN_NAME" ]]; then
        RUN_NAME="$(sanitize_name "$1")"
      else
        die "Unexpected positional argument: $1"
      fi
      shift
      ;;
  esac
done

[[ -n "$SCENE_DIR" ]] || { usage >&2; exit 2; }
[[ -n "$RUN_NAME" ]] || { usage >&2; exit 2; }
[[ -d "$SCENE_DIR" ]] || die "Missing scene dir: $SCENE_DIR"
[[ -f "$SCENE_DIR/config.json" ]] || die "Missing config.json under $SCENE_DIR"

if [[ -z "$EXPORT_DIR" ]]; then
  EXPORT_DIR="4dgs/experiments/wu4dgs/runs/${RUN_NAME}"
fi
if [[ -z "$DOWNLOAD_DIR" ]]; then
  DOWNLOAD_DIR="${SCENE_DIR}/4dgs_wu/${RUN_NAME}"
fi
if [[ -z "$MODEL_REL" ]]; then
  MODEL_REL="wu4dgs_${RUN_NAME}"
fi

MASK_ROOT="${SCENE_DIR}/${MASK_SUBDIR}"
if [[ "$MODE" != "full" && ! -d "$MASK_ROOT" ]]; then
  die "Missing mask root: $MASK_ROOT"
fi

echo "Wu/HUSTVL 4DGaussians one-scene training"
echo "Scene:        $SCENE_DIR"
echo "Run:          $RUN_NAME"
echo "Mask:         $MASK_SUBDIR"
echo "Mode:         $MODE"
echo "Export dir:   $EXPORT_DIR"
echo "Model rel:    $MODEL_REL"
echo "Download:     $DOWNLOAD_DIR"
echo "Iterations:   coarse=$COARSE_ITERATIONS fine=$ITERATIONS"
echo "Temporal:     all_train=$ALL_TRAIN drop_frames=$DROP_INVISIBLE_FRAMES trim_ends=$TRIM_EMPTY_TIME_ENDS min_visible_cameras=$MIN_VISIBLE_CAMERAS min_mask_pixels=$MIN_MASK_PIXELS"

if [[ "$SKIP_EXPORT" != "1" ]]; then
  echo
  echo "==> Exporting DyNeRF scene"
  "$PYTHON_BIN" 4dgs/experiments/object_only/export_object_only_dynerf.py \
    --config "$SCENE_DIR/config.json" \
    --masks-root "$MASK_ROOT" \
    --output "$EXPORT_DIR" \
    --mode "$MODE" \
    --background "$BACKGROUND" \
    --min-visible-cameras "$MIN_VISIBLE_CAMERAS" \
    --min-mask-pixels "$MIN_MASK_PIXELS" \
    $(if [[ "$ALL_TRAIN" == "1" ]]; then printf '%s' "--all-train"; fi) \
    $(if [[ "$DROP_INVISIBLE_FRAMES" == "1" ]]; then printf '%s' "--drop-invisible-frames"; fi) \
    $(if [[ "$TRIM_EMPTY_TIME_ENDS" == "1" ]]; then printf '%s' "--trim-empty-time-ends"; fi)
fi

if [[ "$SKIP_UPLOAD" != "1" ]]; then
  echo
  echo "==> Uploading DyNeRF scene to Modal"
  "$PYTHON_BIN" 4dgs/scripts/upload_4d_scene_to_modal.py \
    "$EXPORT_DIR" \
    --modal-cmd "$MODAL_CMD"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo
  echo "==> Training Wu 4DGS on Modal"
  $MODAL_CMD run modal_app.py \
    --train-wu-4d \
    --train-wu-4d-model "$MODEL_REL" \
    --wu-iterations "$ITERATIONS" \
    --wu-coarse-iterations "$COARSE_ITERATIONS" \
    --wu-time-resolution "$TIME_RESOLUTION" \
    --wu-bounds "$BOUNDS"
fi

if [[ "$RENDER" == "1" ]]; then
  echo
  echo "==> Rendering Wu 4DGS on Modal"
  $MODAL_CMD run modal_app.py \
    --render-wu-4d \
    --render-wu-4d-model "$MODEL_REL" \
    --render-wu-4d-iteration "$ITERATIONS" \
    --wu-time-resolution "$TIME_RESOLUTION" \
    --wu-bounds "$BOUNDS"
fi

echo
echo "==> Downloading Wu model folder"
mkdir -p "$DOWNLOAD_DIR"
$MODAL_CMD volume get phys4d-gs-output \
  "$MODEL_REL" \
  "$DOWNLOAD_DIR" \
  --force

echo
echo "Done."
echo "Local model:  $DOWNLOAD_DIR"
echo "Remote model: phys4d-gs-output:/$MODEL_REL"
echo "PLY:          $DOWNLOAD_DIR/$MODEL_REL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
echo "Deformation:  $DOWNLOAD_DIR/$MODEL_REL/point_cloud/iteration_${ITERATIONS}/deformation.pth"
echo "Checkpoint:   $DOWNLOAD_DIR/$MODEL_REL/chkpnt_fine_${ITERATIONS}.pth"
