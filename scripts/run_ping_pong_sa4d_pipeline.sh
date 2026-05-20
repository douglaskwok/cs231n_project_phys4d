#!/usr/bin/env bash
# Full ping-pong -> 4DGS -> SA4D-inspired identity table pipeline.
#
# Usage:
#   bash scripts/run_ping_pong_sa4d_pipeline.sh
#
# If your Modal CLI has an Apple Silicon architecture issue, try:
#   MODAL_CMD="arch -arm64 modal" bash scripts/run_ping_pong_sa4d_pipeline.sh

set -euo pipefail

PY_SIM="${PY_SIM:-phys_sim/bin/python}"
MODAL_CMD="${MODAL_CMD:-modal}"

SCENE_ID="${SCENE_ID:-scene_0003_e0p90_a15p0}"
SCENE_ROOT="dataset/outputs/ping_pong_12view_restitution_ball_angle_poc/${SCENE_ID}"
DYNERF_OUT="segmentation/01_mask_guided_gaussian_binding/runs/ping_pong_${SCENE_ID}_full_dynerf"
DOWNLOADED_PLY="${DOWNLOADED_PLY:-exported/point_cloud.ply}"
SA4D_OUT="segmentation/sa4d/runs/ping_pong_${SCENE_ID}"

echo "== 1/7 Generate POC dataset =="
"${PY_SIM}" dataset/export_ping_pong_12view.py --variation-set poc

echo "== 2/7 Export ${SCENE_ID} to full-scene 4DGS/DyNeRF =="
"${PY_SIM}" scripts/export_4dgs_dataset.py \
  --config "${SCENE_ROOT}/config.json" \
  --output "${DYNERF_OUT}"

echo "== 3/7 Upload DyNeRF scene to Modal volume =="
python scripts/upload_4d_scene_to_modal.py \
  "${DYNERF_OUT}" \
  --modal-cmd "${MODAL_CMD}"

echo "== 4/7 Train full-scene 4DGS on Modal =="
${MODAL_CMD} run modal_app.py --train-4d

echo "== 5/7 Export trained 4DGS checkpoint to PLY on Modal =="
${MODAL_CMD} run modal_app.py --export-4d-ply

echo "== 6/7 Download exported PLY =="
${MODAL_CMD} volume get phys4d-gs-output 4dgs_sphere_bounce/point_cloud/exported . --force

echo "== 7/7 Build SA4D-inspired identity table =="
"${PY_SIM}" segmentation/sa4d/build_identity_table.py \
  --ply "${DOWNLOADED_PLY}" \
  --masks-root "${SCENE_ROOT}/masks" \
  --cameras-json "${SCENE_ROOT}/cameras.json" \
  --frame-start 0 \
  --frame-end 90 \
  --frame-step 10 \
  --min-camera-hits 2 \
  --out-dir "${SA4D_OUT}"

echo "Done. Results: ${SA4D_OUT}"
