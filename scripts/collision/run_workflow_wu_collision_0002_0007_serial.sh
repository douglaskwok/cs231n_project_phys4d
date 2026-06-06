#!/usr/bin/env bash
# Serial event20 collision prediction queue for elastic scenes 0002-0007.
# Uses the event-aligned 20-frame split chosen by choose_prediction_split.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_PROFILE="${MODAL_PROFILE:-simon}"
export MODAL="${MODAL:-arch -arm64 modal}"
export PY="${PY:-.venv_pipeline/bin/python}"
export PRE_COLLISION_FRAMES="${PRE_COLLISION_FRAMES:-5}"
export POST_COLLISION_FRAMES="${POST_COLLISION_FRAMES:-14}"
export FPS="${FPS:-60}"

ROOT="dataset/outputs/phys4d_final/collision_scale1p75_elastic_2x2x2_60fps"
SCENES="${SCENES:-0002 0003 0004 0005 0006 0007}"

want_scene() {
  local needle="$1"
  for scene_id in $SCENES; do
    [[ "$scene_id" == "$needle" ]] && return 0
  done
  return 1
}

run_scene() {
  local scene_id="$1"
  local scene="$2"
  local wu_a="$3"
  local iter_a="$4"
  local wu_b="$5"
  local iter_b="$6"

  local base
  base="$(basename "$scene")"
  echo
  echo "================================================================"
  echo "START scene_${scene_id} ${base} $(date)"
  echo "SCENE=${scene}"
  echo "WU_A=${wu_a} ITER_A=${iter_a}"
  echo "WU_B=${wu_b} ITER_B=${iter_b}"
  echo "================================================================"

  SCENE="$scene" \
  WU_A="$wu_a" \
  WU_B="$wu_b" \
  ITER_A="$iter_a" \
  ITER_B="$iter_b" \
  RUN="outputs/collision_pipeline/${base}_event20" \
  MBASE="wu_collision_${base}_event20_$(date +%Y%m%d)" \
  bash scripts/collision/run_workflow_wu_collision_scene0000.sh

  echo "DONE scene_${scene_id} ${base} $(date)"
}

if want_scene "0002"; then
run_scene "0002" \
  "$ROOT/scene_0002_col_m220_equal105_r098" \
  "$ROOT/scene_0002_col_m220_equal105_r098/4dgs_wu/wu_collision_elastic_s0002_object_a_compactA_iter60000/wu4dgs_wu_collision_elastic_s0002_object_a_compactA_iter60000" \
  "60000" \
  "$ROOT/scene_0002_col_m220_equal105_r098/4dgs_wu/wu_collision_elastic_s0002_object_b_compactA_iter60000/wu4dgs_wu_collision_elastic_s0002_object_b_compactA_iter60000" \
  "60000"
fi

if want_scene "0003"; then
run_scene "0003" \
  "$ROOT/scene_0003_col_m220_equal105_r090" \
  "$ROOT/scene_0003_col_m220_equal105_r090/4dgs_wu/wu_collision_elastic_s0003_object_a_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0003_object_a_compactA_50000_to60000" \
  "60000" \
  "$ROOT/scene_0003_col_m220_equal105_r090/4dgs_wu/wu_collision_elastic_s0003_object_b_anchor005_desaturn_70000_to80000/wu4dgs_wu_collision_elastic_s0003_object_b_anchor005_desaturn_70000_to80000" \
  "80000"
fi

if want_scene "0004"; then
run_scene "0004" \
  "$ROOT/scene_0004_col_m440_asym140_r098" \
  "$ROOT/scene_0004_col_m440_asym140_r098/4dgs_wu/wu_collision_elastic_s0004_object_a_cleanup_60000_to70000/wu4dgs_wu_collision_elastic_s0004_object_a_cleanup_60000_to70000" \
  "70000" \
  "$ROOT/scene_0004_col_m440_asym140_r098/4dgs_wu/wu_collision_elastic_s0004_object_b_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0004_object_b_compactA_50000_to60000" \
  "60000"
fi

if want_scene "0005"; then
run_scene "0005" \
  "$ROOT/scene_0005_col_m440_asym140_r090" \
  "$ROOT/scene_0005_col_m440_asym140_r090/4dgs_wu/wu_collision_elastic_s0005_object_a_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0005_object_a_compactA_50000_to60000" \
  "60000" \
  "$ROOT/scene_0005_col_m440_asym140_r090/4dgs_wu/wu_collision_elastic_s0005_object_b_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0005_object_b_compactA_50000_to60000" \
  "60000"
fi

if want_scene "0006"; then
run_scene "0006" \
  "$ROOT/scene_0006_col_m440_equal105_r098" \
  "$ROOT/scene_0006_col_m440_equal105_r098/4dgs_wu/wu_collision_elastic_s0006_object_a_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0006_object_a_compactA_50000_to60000" \
  "60000" \
  "$ROOT/scene_0006_col_m440_equal105_r098/4dgs_wu/wu_collision_elastic_s0006_object_b_all30k_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0006_object_b_all30k_compactA_50000_to60000" \
  "60000"
fi

if want_scene "0007"; then
run_scene "0007" \
  "$ROOT/scene_0007_col_m440_equal105_r090" \
  "$ROOT/scene_0007_col_m440_equal105_r090/4dgs_wu/wu_collision_elastic_s0007_object_a_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0007_object_a_compactA_50000_to60000" \
  "60000" \
  "$ROOT/scene_0007_col_m440_equal105_r090/4dgs_wu/wu_collision_elastic_s0007_object_b_compactA_50000_to60000/wu4dgs_wu_collision_elastic_s0007_object_b_compactA_50000_to60000" \
  "60000"
fi

echo
echo "ALL DONE $(date)"
