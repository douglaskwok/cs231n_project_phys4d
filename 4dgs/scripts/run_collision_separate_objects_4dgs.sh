#!/usr/bin/env bash
# Train object_a and visibility-filtered object_b as separate object-only 4DGS
# models, then composite their rendered PNGs into one collision preview video.
#
# Default is a quality 15k run. Use --quick for the 1k smoke-test config.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-phys_sim/bin/python}"
MODAL_CMD="${MODAL_CMD:-arch -arm64 modal}"

CONFIG="room_physics_4dgs_2p6s.yaml"
CHECKPOINT="chkpnt15000.pth"
SUFFIX="15k"
RENDER_FPS="60"
SKIP_TRAIN="0"
SKIP_RENDER="0"

usage() {
  cat <<'EOF'
Usage:
  bash 4dgs/scripts/run_collision_separate_objects_4dgs.sh [options]

Options:
  --quick        Use 1k quick config/checkpoint instead of 15k quality config.
  --skip-train   Skip upload/train and only render/download from existing remote checkpoint.
  --skip-render  Skip Modal render/download; only composite existing local downloads.
  --fps N        Render/build video FPS. Default: 60.
  -h, --help     Show this help.

Outputs:
  latest_collision_object_a_<suffix>/
  latest_collision_object_b_<suffix>/
  latest_collision_object_ab_composited_<suffix>/latest/videos/train_cams_grid.mp4

Notes:
  This script runs object A then object B sequentially because the Modal app
  overwrites /data/4d_scene, /outputs/4dgs_sphere_bounce, and
  /outputs/4dgs_renders/latest on each run.

  Object B uses collision_room_object_b_visible_min50, which drops camera-time
  pairs where the blue object mask is invisible/tiny while preserving original
  timestamps and frame IDs for render alignment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      CONFIG="room_physics_4dgs_quick_2p6s.yaml"
      CHECKPOINT="chkpnt1000.pth"
      SUFFIX="quick"
      shift
      ;;
    --skip-train)
      SKIP_TRAIN="1"
      shift
      ;;
    --skip-render)
      SKIP_RENDER="1"
      shift
      ;;
    --fps)
      RENDER_FPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

prepare_download_dir() {
  local path="$1"
  if [[ -e "$path" && ! -d "$path" ]]; then
    echo "Download target exists but is not a directory: $path" >&2
    echo "Move or delete it, then rerun." >&2
    exit 1
  fi
  mkdir -p "$path"
}

run_one_object() {
  local label="$1"
  local scene_dir="$2"
  local download_dir="$3"

  require_dir "$scene_dir"

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    echo
    echo "==> Uploading ${label}: ${scene_dir}"
    "$PYTHON_BIN" 4dgs/scripts/upload_4d_scene_to_modal.py \
      "$scene_dir" \
      --modal-cmd "$MODAL_CMD"

    echo
    echo "==> Training ${label} with ${CONFIG}"
    $MODAL_CMD run modal_app.py \
      --train-4d \
      --train-4d-config "$CONFIG"
  else
    echo
    echo "==> Skipping upload/train for ${label}"
  fi

  if [[ "$SKIP_RENDER" != "1" ]]; then
    echo
    echo "==> Rendering ${label} checkpoint ${CHECKPOINT}"
    $MODAL_CMD run modal_app.py \
      --render-4d \
      --render-4d-config "$CONFIG" \
      --render-4d-checkpoint "$CHECKPOINT" \
      --render-fps "$RENDER_FPS"

    prepare_download_dir "$download_dir"
    echo
    echo "==> Downloading ${label} renders to ${download_dir}"
    $MODAL_CMD volume get phys4d-gs-output \
      4dgs_renders/latest \
      "$download_dir" \
      --force
  else
    echo
    echo "==> Skipping Modal render/download for ${label}; expecting ${download_dir}/latest"
  fi

  require_dir "$download_dir/latest"
}

OBJECT_A_SCENE="4dgs/experiments/object_only/runs/collision_room_object_a"
OBJECT_B_SCENE="4dgs/experiments/object_only/runs/collision_room_object_b_visible_min50"
OBJECT_A_DOWNLOAD="latest_collision_object_a_${SUFFIX}"
OBJECT_B_DOWNLOAD="latest_collision_object_b_${SUFFIX}"
COMPOSITE_DIR="latest_collision_object_ab_composited_${SUFFIX}/latest"

echo "Collision separate-object 4DGS pipeline"
echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "FPS:        ${RENDER_FPS}"

run_one_object "object_a" "$OBJECT_A_SCENE" "$OBJECT_A_DOWNLOAD"
run_one_object "object_b" "$OBJECT_B_SCENE" "$OBJECT_B_DOWNLOAD"

echo
echo "==> Compositing object_a + object_b renders"
"$PYTHON_BIN" 4dgs/scripts/composite_4dgs_renders.py \
  "$OBJECT_A_DOWNLOAD/latest" \
  "$OBJECT_B_DOWNLOAD/latest" \
  --out-dir "$COMPOSITE_DIR" \
  --mode max \
  --frame-policy reference

echo
echo "==> Building MP4 previews"
"$PYTHON_BIN" 4dgs/scripts/build_4dgs_render_videos.py \
  "$COMPOSITE_DIR" \
  --fps "${RENDER_FPS%.*}"

echo
echo "Done."
echo "Open:"
echo "  open ${COMPOSITE_DIR}/videos/train_cams_grid.mp4"
echo
echo "Single-object previews:"
echo "  ${OBJECT_A_DOWNLOAD}/latest"
echo "  ${OBJECT_B_DOWNLOAD}/latest"
