#!/usr/bin/env bash
# Export one final PyBullet scene to DyNeRF, train 4DGS on Modal, download the
# checkpoint/PLY folder, and record wall-clock timing.
#
# Default target is object-only 4DGS using the scene's dynamic-object mask
# folder (`masks`). For collision/stacking per-object training, pass
# --mask-subdir masks_object_a, --mask-subdir masks_block_00, etc.

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
QUALITY="quality"
CONFIG=""
CHECKPOINT=""
MODEL_REL=""
EXPORT_DIR=""
DOWNLOAD_DIR=""
SKIP_EXPORT="0"
SKIP_UPLOAD="0"
SKIP_TRAIN="0"
NO_DOWNLOAD="0"
ALL_TRAIN="0"

usage() {
  cat <<'EOF'
Usage:
  bash 4dgs/scripts/train_one_final_scene_4dgs.sh SCENE_DIR RUN_NAME [options]

Required:
  SCENE_DIR      Generated scene directory containing config.json, rgb/, masks/.
  RUN_NAME       Stable output name, e.g. ball_drop_e0p78_a0p0_object.

Options:
  --mask-subdir NAME   Mask folder inside SCENE_DIR. Default: masks.
                       Examples: masks_object_a, masks_block_00, masks_soft_torus.
  --mode MODE          object, background, or full. Default: object.
  --background COLOR   black or white for masked-out pixels. Default: black.
  --quick              Use 1000-iteration quick config/checkpoint.
  --config NAME        Override auto-selected 4DGS config.
  --checkpoint NAME    Override checkpoint to report/download metadata for.
  --model-rel NAME     Remote Modal output folder. Default: 4dgs_<RUN_NAME>.
  --export-dir PATH    Local DyNeRF export folder.
                       Default: 4dgs/experiments/object_only/runs/<RUN_NAME>.
  --download-dir PATH  Local downloaded model folder.
                       Default: <SCENE_DIR>/4dgs/<RUN_NAME>.
  --skip-export        Reuse existing export-dir.
  --skip-upload        Reuse existing phys4d-gs-data:/4d_scene.
  --skip-train         Do not train; useful for downloading an existing model.
  --no-download        Do not download model folder after training.
  --all-train          Ignore scene train/test split; train on all cameras and
                       all frames. Recommended for final reconstruction/viewing.
  -h, --help           Show this help.

Outputs:
  <export-dir>/                         DyNeRF train/test JSON + images
  <download-dir>/                       downloaded Modal model folder
  <download-dir>/timing.json            export/upload/train/download timings

Config auto-selection:
  <= 2.7s scene: room_physics_4dgs_2p6s.yaml
  <= 4.1s scene: room_physics_4dgs_4p0s.yaml
  otherwise:    room_physics_4dgs_5p0s.yaml

Examples:
  # First benchmark: final ball-drop anchor, object-only, quality 15k.
  bash 4dgs/scripts/train_one_final_scene_4dgs.sh \
    dataset/outputs/phys4d_final/ball_drop_3x3_60fps/scene_0004_e0p78_a0p0 \
    ball_drop_e0p78_a0p0_object

  # Quick collision object A smoke test.
  bash 4dgs/scripts/train_one_final_scene_4dgs.sh \
    dataset/outputs/phys4d_final/collision_base_60fps/scene_0000_collision_room \
    collision_object_a_quick \
    --mask-subdir masks_object_a \
    --quick
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing directory: $path"
}

sanitize_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", sys.argv[1]).strip("_"))
PY
}

scene_duration() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path
scene = Path(sys.argv[1])
metadata = scene / "metadata.json"
config = scene / "config.json"
duration = None
if metadata.is_file():
    data = json.loads(metadata.read_text())
    duration = data.get("duration_s") or data.get("duration_sec")
if duration is None and config.is_file():
    data = json.loads(config.read_text())
    duration = data.get("simulation", {}).get("duration_s")
if duration is None:
    raise SystemExit("could not infer scene duration")
print(float(duration))
PY
}

select_config() {
  local scene="$1"
  local quality="$2"
  "$PYTHON_BIN" - "$scene" "$quality" <<'PY'
import json
import sys
from pathlib import Path
scene = Path(sys.argv[1])
quality = sys.argv[2]
metadata = scene / "metadata.json"
config = scene / "config.json"
duration = None
if metadata.is_file():
    data = json.loads(metadata.read_text())
    duration = data.get("duration_s") or data.get("duration_sec")
if duration is None and config.is_file():
    data = json.loads(config.read_text())
    duration = data.get("simulation", {}).get("duration_s")
if duration is None:
    raise SystemExit("could not infer scene duration for config selection")
duration = float(duration)
prefix = "room_physics_4dgs_quick" if quality == "quick" else "room_physics_4dgs"
if duration <= 2.7:
    suffix = "2p6s"
elif duration <= 4.1:
    suffix = "4p0s"
else:
    suffix = "5p0s"
print(f"{prefix}_{suffix}.yaml")
PY
}

json_quote() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

record_timing() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
keys = [
    "run_name",
    "scene_dir",
    "export_dir",
    "mask_subdir",
    "mode",
    "config",
    "checkpoint",
    "model_rel",
    "download_dir",
    "quality",
]
timing_keys = [
    "export_seconds",
    "upload_seconds",
    "train_seconds",
    "download_seconds",
    "total_seconds",
]
data = {key: os.environ.get(key.upper(), "") for key in keys}
data["timings"] = {
    key: float(os.environ[key.upper()])
    for key in timing_keys
    if os.environ.get(key.upper()) not in (None, "")
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
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
    --quick)
      QUALITY="quick"
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --model-rel)
      MODEL_REL="$2"
      shift 2
      ;;
    --export-dir)
      EXPORT_DIR="$2"
      shift 2
      ;;
    --download-dir)
      DOWNLOAD_DIR="$2"
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
    --no-download)
      NO_DOWNLOAD="1"
      shift
      ;;
    --all-train)
      ALL_TRAIN="1"
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

require_dir "$SCENE_DIR"
[[ -f "$SCENE_DIR/config.json" ]] || die "Missing config.json under $SCENE_DIR"

if [[ "$MODE" != "object" && "$MODE" != "background" && "$MODE" != "full" ]]; then
  die "--mode must be object, background, or full"
fi
if [[ "$BACKGROUND" != "black" && "$BACKGROUND" != "white" ]]; then
  die "--background must be black or white"
fi

if [[ -z "$CONFIG" ]]; then
  CONFIG="$(select_config "$SCENE_DIR" "$QUALITY")"
fi
if [[ -z "$CHECKPOINT" ]]; then
  if [[ "$QUALITY" == "quick" ]]; then
    CHECKPOINT="chkpnt1000.pth"
  else
    CHECKPOINT="chkpnt15000.pth"
  fi
fi
if [[ -z "$MODEL_REL" ]]; then
  MODEL_REL="4dgs_${RUN_NAME}"
fi
if [[ -z "$EXPORT_DIR" ]]; then
  EXPORT_DIR="4dgs/experiments/object_only/runs/${RUN_NAME}"
fi
if [[ -z "$DOWNLOAD_DIR" ]]; then
  DOWNLOAD_DIR="${SCENE_DIR}/4dgs/${RUN_NAME}"
fi

MASK_ROOT="${SCENE_DIR}/${MASK_SUBDIR}"
if [[ "$MODE" != "full" ]]; then
  require_dir "$MASK_ROOT"
fi

echo "4DGS one-scene training"
echo "Scene:      $SCENE_DIR"
echo "Run:        $RUN_NAME"
echo "Mask:       $MASK_SUBDIR"
echo "Mode:       $MODE"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Model rel:  $MODEL_REL"
echo "Export dir: $EXPORT_DIR"
echo "Download:   $DOWNLOAD_DIR"
echo "Duration:   $(scene_duration "$SCENE_DIR") seconds"

TOTAL_START=$SECONDS
EXPORT_SECONDS=0
UPLOAD_SECONDS=0
TRAIN_SECONDS=0
DOWNLOAD_SECONDS=0

if [[ "$SKIP_EXPORT" != "1" ]]; then
  echo
  echo "==> Exporting DyNeRF scene"
  START=$SECONDS
  "$PYTHON_BIN" 4dgs/experiments/object_only/export_object_only_dynerf.py \
    --config "$SCENE_DIR/config.json" \
    --masks-root "$MASK_ROOT" \
    --output "$EXPORT_DIR" \
    --mode "$MODE" \
    --background "$BACKGROUND" \
    $(if [[ "$ALL_TRAIN" == "1" ]]; then printf '%s' "--all-train"; fi)
  EXPORT_SECONDS=$((SECONDS - START))
else
  echo
  echo "==> Skipping export; using $EXPORT_DIR"
fi
require_dir "$EXPORT_DIR"

if [[ "$SKIP_UPLOAD" != "1" ]]; then
  echo
  echo "==> Uploading DyNeRF scene to Modal"
  START=$SECONDS
  "$PYTHON_BIN" 4dgs/scripts/upload_4d_scene_to_modal.py \
    "$EXPORT_DIR" \
    --modal-cmd "$MODAL_CMD"
  UPLOAD_SECONDS=$((SECONDS - START))
else
  echo
  echo "==> Skipping upload; using existing phys4d-gs-data:/4d_scene"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo
  echo "==> Training 4DGS on Modal"
  START=$SECONDS
  $MODAL_CMD run modal_app.py \
    --train-4d \
    --train-4d-config "$CONFIG" \
    --train-4d-model "$MODEL_REL"
  TRAIN_SECONDS=$((SECONDS - START))
else
  echo
  echo "==> Skipping training; using existing phys4d-gs-output:/$MODEL_REL"
fi

if [[ "$NO_DOWNLOAD" != "1" ]]; then
  echo
  echo "==> Downloading model folder"
  START=$SECONDS
  if [[ -e "$DOWNLOAD_DIR" && ! -d "$DOWNLOAD_DIR" ]]; then
    die "Download target exists but is not a directory: $DOWNLOAD_DIR"
  fi
  mkdir -p "$DOWNLOAD_DIR"
  $MODAL_CMD volume get phys4d-gs-output \
    "$MODEL_REL" \
    "$DOWNLOAD_DIR" \
    --force
  DOWNLOAD_SECONDS=$((SECONDS - START))
fi

TOTAL_SECONDS=$((SECONDS - TOTAL_START))

export RUN_NAME SCENE_DIR EXPORT_DIR MASK_SUBDIR MODE CONFIG CHECKPOINT MODEL_REL DOWNLOAD_DIR QUALITY
export EXPORT_SECONDS UPLOAD_SECONDS TRAIN_SECONDS DOWNLOAD_SECONDS TOTAL_SECONDS

TIMING_PATH="${DOWNLOAD_DIR}/timing.json"
if [[ "$NO_DOWNLOAD" == "1" ]]; then
  TIMING_PATH="${EXPORT_DIR}/timing.json"
fi
record_timing "$TIMING_PATH"

echo
echo "Done."
echo "Timing:"
echo "  export:   ${EXPORT_SECONDS}s"
echo "  upload:   ${UPLOAD_SECONDS}s"
echo "  train:    ${TRAIN_SECONDS}s"
echo "  download: ${DOWNLOAD_SECONDS}s"
echo "  total:    ${TOTAL_SECONDS}s"
echo "Wrote timing: $TIMING_PATH"
echo
echo "Gaussian artifacts:"
echo "  local:  ${DOWNLOAD_DIR}"
echo "  remote: phys4d-gs-output:/${MODEL_REL}"
echo "  PLY:    ${DOWNLOAD_DIR}/${MODEL_REL}/point_cloud/exported/point_cloud.ply"
