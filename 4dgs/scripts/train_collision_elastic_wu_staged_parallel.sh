#!/usr/bin/env bash
# Train Wu 4DGS collision scenes with the staged recipe that worked best:
# a gentler 50k base fit, followed by compact-A continuation to 60k.
#
# Scene pipelines run in limited parallelism. Inside each scene, objects and
# stages run sequentially so checkpoint dependencies are simple and reliable.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${DATASET_ROOT:-dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps}"
MODAL_PROFILE="${MODAL_PROFILE:-simon}"
SCENE_ORDER="${SCENE_ORDER:-1 3 5}"
OBJECTS="${OBJECTS:-a b}"
BASE_ITERATIONS="${BASE_ITERATIONS:-50000}"
COMPACT_ITERATIONS="${COMPACT_ITERATIONS:-60000}"
MAX_PARALLEL_SCENES="${MAX_PARALLEL_SCENES:-2}"
LOG_DIR="${LOG_DIR:-/tmp/wu4dgs_collision_logs}"

COMMON_ARGS=(
  --mode object
  --background black
  --coarse-iterations 1000
  --time-resolution 120
  --bounds 1.2
  --foreground-loss-weight 20
  --mask-loss-weight 1
  --wu-densify-until-iter 1000
  --wu-opacity-reset-interval 300000
  --drop-invisible-frames
  --min-visible-cameras 6
  --min-mask-pixels 50
  --init-points 20000
  --init-center-mode first
  --init-surface-ratio 0.8
)

BASE_ARGS=(
  --bg-spill-loss-weight 1
  --area-loss-weight 0.1
  --scale-isotropy-loss-weight 0.05
)

COMPACT_ARGS=(
  --bg-spill-loss-weight 2
  --area-loss-weight 0.2
  --scale-isotropy-loss-weight 0.05
)

mkdir -p "$LOG_DIR"

train_scene() {
  local scene_idx="$1"
  local scene_dir
  scene_dir="$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "$(printf 'scene_%04d_*' "$scene_idx")" | sort | head -n 1)"
  if [[ -z "$scene_dir" ]]; then
    echo "Missing scene index $scene_idx under $DATASET_ROOT" >&2
    return 1
  fi

  echo "==> Scene $scene_idx: $scene_dir"
  for object in $OBJECTS; do
    local mask_subdir object_tag
    case "$object" in
      a|object_a)
        mask_subdir="masks_object_a"
        object_tag="object_a"
        ;;
      b|object_b)
        mask_subdir="masks_object_b"
        object_tag="object_b"
        ;;
      *)
        echo "Unknown object '$object'; use a/b or object_a/object_b." >&2
        return 1
        ;;
    esac

    local base_run compact_run base_remote_checkpoint
    base_run="wu_collision_elastic_s$(printf '%04d' "$scene_idx")_${object_tag}_base${BASE_ITERATIONS}"
    compact_run="wu_collision_elastic_s$(printf '%04d' "$scene_idx")_${object_tag}_compactA_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}"
    base_remote_checkpoint="wu4dgs_${base_run}/chkpnt_fine_${BASE_ITERATIONS}.pth"

    echo "==> Base fit $base_run"
    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$scene_dir" \
      "$base_run" \
      --mask-subdir "$mask_subdir" \
      --iterations "$BASE_ITERATIONS" \
      "${COMMON_ARGS[@]}" \
      "${BASE_ARGS[@]}"

    echo "==> Compact continuation $compact_run from $base_remote_checkpoint"
    MODAL_PROFILE="$MODAL_PROFILE" bash 4dgs/scripts/train_one_scene_wu4dgs.sh \
      "$scene_dir" \
      "$compact_run" \
      --mask-subdir "$mask_subdir" \
      --iterations "$COMPACT_ITERATIONS" \
      "${COMMON_ARGS[@]}" \
      "${COMPACT_ARGS[@]}" \
      --wu-start-checkpoint "$base_remote_checkpoint" \
      --render
  done
}

pids=()
pid_scenes=()
status=0

for scene_idx in $SCENE_ORDER; do
  while [[ "${#pids[@]}" -ge "$MAX_PARALLEL_SCENES" ]]; do
    pid="${pids[0]}"
    scene="${pid_scenes[0]}"
    if ! wait "$pid"; then
      echo "Scene $scene failed" >&2
      status=1
    fi
    pids=("${pids[@]:1}")
    pid_scenes=("${pid_scenes[@]:1}")
  done

  log_path="$LOG_DIR/collision_elastic_s$(printf '%04d' "$scene_idx")_staged_${BASE_ITERATIONS}_to${COMPACT_ITERATIONS}.log"
  echo "Launching staged scene $scene_idx -> $log_path"
  train_scene "$scene_idx" >"$log_path" 2>&1 &
  pids+=("$!")
  pid_scenes+=("$scene_idx")
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Scene ${pid_scenes[$i]} failed" >&2
    status=1
  fi
done

exit "$status"
