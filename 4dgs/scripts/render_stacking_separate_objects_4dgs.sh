#!/usr/bin/env bash
# Render the separately trained stacking block 4DGS models, then composite them
# into one preview while preserving original camera/frame IDs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-phys_sim/bin/python}"
MODAL_CMD="${MODAL_CMD:-arch -arm64 modal}"

CONFIG="room_physics_4dgs_5p0s.yaml"
CHECKPOINT="chkpnt15000.pth"
RENDER_FPS="60"
SKIP_MODAL="0"

usage() {
  cat <<'EOF'
Usage:
  bash 4dgs/scripts/render_stacking_separate_objects_4dgs.sh [options]

Options:
  --quick        Use quick config/checkpoint: room_physics_4dgs_quick_5p0s.yaml + chkpnt1000.pth.
  --skip-modal   Skip Modal upload/render/download and only composite existing local render folders.
  --fps N        Render/build video FPS. Default: 60.
  -h, --help     Show this help.

Inputs expected on Modal output volume:
  /outputs/4dgs_stacking_big_block_00/chkpnt15000.pth
  /outputs/4dgs_stacking_big_block_01/chkpnt15000.pth
  /outputs/4dgs_stacking_big_block_02/chkpnt15000.pth

Local outputs:
  latest_stacking_big_block_00_render/latest/
  latest_stacking_big_block_01_render/latest/
  latest_stacking_big_block_02_render/latest/
  latest_stacking_big_blocks_composited/latest/videos/train_cams_grid.mp4

Notes:
  This runs sequentially because each render needs its matching DyNeRF scene
  uploaded to Modal at /data/4d_scene. The composite step aligns objects by
  camera ID and original frame ID, so blocks that start later remain black
  before their first visible frame.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      CONFIG="room_physics_4dgs_quick_5p0s.yaml"
      CHECKPOINT="chkpnt1000.pth"
      shift
      ;;
    --skip-modal)
      SKIP_MODAL="1"
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

render_one_block() {
  local label="$1"
  local scene_dir="$2"
  local model_rel="$3"
  local download_dir="$4"

  require_dir "$scene_dir"

  if [[ "$SKIP_MODAL" != "1" ]]; then
    echo
    echo "==> Uploading ${label} scene: ${scene_dir}"
    "$PYTHON_BIN" 4dgs/scripts/upload_4d_scene_to_modal.py \
      "$scene_dir" \
      --modal-cmd "$MODAL_CMD"

    echo
    echo "==> Rendering ${label}: ${model_rel}/${CHECKPOINT}"
    $MODAL_CMD run modal_app.py \
      --render-4d \
      --render-4d-config "$CONFIG" \
      --render-4d-model "$model_rel" \
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

BLOCK_00_SCENE="4dgs/experiments/object_only/runs/stacking_big_block_00"
BLOCK_01_SCENE="4dgs/experiments/object_only/runs/stacking_big_block_01"
BLOCK_02_SCENE="4dgs/experiments/object_only/runs/stacking_big_block_02"

BLOCK_00_MODEL="4dgs_stacking_big_block_00"
BLOCK_01_MODEL="4dgs_stacking_big_block_01"
BLOCK_02_MODEL="4dgs_stacking_big_block_02"

BLOCK_00_RENDER="latest_stacking_big_block_00_render"
BLOCK_01_RENDER="latest_stacking_big_block_01_render"
BLOCK_02_RENDER="latest_stacking_big_block_02_render"

COMPOSITE_DIR="latest_stacking_big_blocks_composited/latest"

echo "Stacking separate-object render/composite pipeline"
echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "FPS:        ${RENDER_FPS}"

render_one_block "block_00" "$BLOCK_00_SCENE" "$BLOCK_00_MODEL" "$BLOCK_00_RENDER"
render_one_block "block_01" "$BLOCK_01_SCENE" "$BLOCK_01_MODEL" "$BLOCK_01_RENDER"
render_one_block "block_02" "$BLOCK_02_SCENE" "$BLOCK_02_MODEL" "$BLOCK_02_RENDER"

echo
echo "==> Compositing block_00 + block_01 + block_02 renders"
"$PYTHON_BIN" 4dgs/scripts/composite_4dgs_renders.py \
  "$BLOCK_00_RENDER/latest" \
  "$BLOCK_01_RENDER/latest" \
  "$BLOCK_02_RENDER/latest" \
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
