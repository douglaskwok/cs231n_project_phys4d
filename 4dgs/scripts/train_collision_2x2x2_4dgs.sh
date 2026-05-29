#!/usr/bin/env bash
# Train separate 4DGS models for object_a and object_b across all 8 scenes
# in the collision 2x2x2 dataset.
#
# Calls train_one_final_scene_4dgs.sh twice per scene (once per object),
# producing 16 run folders total under each scene's 4dgs/ subdirectory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-phys_sim/bin/python}"
MODAL_CMD="${MODAL_CMD:-arch -arm64 modal}"

SCENES_DIR="dataset/outputs/phys4d_final/collision_2x2x2_60fps"
QUICK_FLAG=""

usage() {
  cat <<'EOF'
Usage:
  bash 4dgs/scripts/train_collision_2x2x2_4dgs.sh [options]

Options:
  --quick        Use 1k quick config/checkpoint instead of quality 30k.
  --scenes-dir PATH  Override default scenes directory.
                     Default: dataset/outputs/phys4d_final/collision_2x2x2_60fps
  -h, --help     Show this help.

Outputs per scene (16 total):
  <scene_dir>/4dgs/<short_name>_object_a_all12_allframes/
  <scene_dir>/4dgs/<short_name>_object_b_all12_allframes/

Example run names:
  col_m100_v060_r025_object_a_all12_allframes
  col_m100_v060_r025_object_b_all12_allframes
  ...
  col_m440_v140_r075_object_b_all12_allframes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK_FLAG="--quick"
      shift
      ;;
    --scenes-dir)
      SCENES_DIR="$2"
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

[[ -d "$SCENES_DIR" ]] || { echo "Error: scenes directory not found: $SCENES_DIR" >&2; exit 1; }

scenes=( "$SCENES_DIR"/scene_*/ )
[[ ${#scenes[@]} -gt 0 ]] || { echo "Error: no scene directories found under $SCENES_DIR" >&2; exit 1; }

echo "Collision 2x2x2 4DGS training"
echo "Scenes dir: $SCENES_DIR"
echo "Scenes:     ${#scenes[@]}"
echo "Objects:    object_a, object_b"
echo "Total runs: $(( ${#scenes[@]} * 2 ))"
echo

for scene_dir in "${scenes[@]}"; do
  scene_name="$(basename "$scene_dir")"
  # Strip leading scene_XXXX_ index to get e.g. col_m100_v060_r025
  short_name="${scene_name#scene_????_}"

  for obj in object_a object_b; do
    run_name="${short_name}_${obj}_all12_allframes"
    echo "========================================================"
    echo "Scene:  $scene_name"
    echo "Object: $obj  ->  run: $run_name"
    echo "========================================================"
    bash 4dgs/scripts/train_one_final_scene_4dgs.sh \
      "$scene_dir" \
      "$run_name" \
      --mask-subdir "masks_${obj}" \
      --all-train \
      ${QUICK_FLAG:+"$QUICK_FLAG"}
    echo
  done
done

echo "All collision 2x2x2 scenes trained."
