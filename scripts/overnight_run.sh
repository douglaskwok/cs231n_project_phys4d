#!/usr/bin/env bash
# Autonomous overnight: train CNN+MLP, ablations, eval, E2E pipeline (if 3DGS present).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/overnight/run.log
mkdir -p outputs/overnight outputs/ablations outputs/param_predictor
echo "" >> "$LOG"
exec >> "$LOG" 2>&1

echo "=== overnight start $(date) ==="
export PYTHONPATH=src

if command -v conda >/dev/null 2>&1 && conda env list | grep -q '^phys4d '; then
  PYTHON="conda run --no-capture-output -n phys4d python"
else
  PYTHON=python
fi

MANIFEST=outputs/sphere_bounce_batch/dataset_manifest.json
if [[ ! -f "$MANIFEST" ]]; then
  $PYTHON scripts/build_param_id_splits.py \
    --batch-root outputs/sphere_bounce_batch \
    --out "$MANIFEST"
fi

$PYTHON -c "import torch; d='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'); print(d)" > outputs/overnight/device.txt || echo cpu > outputs/overnight/device.txt
DEVICE=$(cat outputs/overnight/device.txt)

EPOCHS="${OVERNIGHT_EPOCHS:-40}"

echo "=== train CNN ($DEVICE, $EPOCHS epochs) ==="
$PYTHON scripts/train_param_predictor.py \
  --manifest "$MANIFEST" \
  --data-root outputs \
  --epochs "$EPOCHS" \
  --model cnn \
  --max-views 6 \
  --out-dir outputs/param_predictor \
  --device "$DEVICE"

echo "=== train pose MLP ==="
$PYTHON scripts/train_param_predictor.py \
  --manifest "$MANIFEST" \
  --data-root outputs \
  --epochs "$EPOCHS" \
  --model mlp \
  --out-dir outputs/ablations/mlp \
  --device "$DEVICE"

echo "=== eval test split ==="
$PYTHON scripts/eval_param_predictor.py \
  --manifest "$MANIFEST" \
  --data-root outputs \
  --checkpoint outputs/param_predictor/param_predictor_best.pt \
  --model cnn \
  --max-views 6 \
  --split test \
  --out-json outputs/overnight/eval_cnn_test.json \
  --device "$DEVICE"

$PYTHON scripts/eval_param_predictor.py \
  --manifest "$MANIFEST" \
  --data-root outputs \
  --checkpoint outputs/ablations/mlp/pose_mlp_best.pt \
  --model mlp \
  --split test \
  --out-json outputs/overnight/eval_mlp_test.json \
  --device "$DEVICE"

echo "=== view sweep K=4 (shorter train) ==="
$PYTHON scripts/train_param_predictor.py \
  --manifest "$MANIFEST" \
  --data-root outputs \
  --epochs $((EPOCHS / 2)) \
  --model cnn \
  --max-views 4 \
  --out-dir outputs/overnight/cnn_k4 \
  --device "$DEVICE" || true

if [[ -f gs_sphere_bounce/point_cloud/iteration_7000/point_cloud.ply ]]; then
  echo "=== E2E pipeline ==="
  $PYTHON scripts/run_param_id_pipeline.py \
    --checkpoint outputs/param_predictor/param_predictor_best.pt \
    --device "$DEVICE" || true
  $PYTHON scripts/eval_warped_crop_mae.py \
    --poses-csv outputs/param_id_pipeline/object_poses_predicted.csv || true
else
  echo "SKIP E2E: no gs_sphere_bounce PLY (run Modal --upload --train)"
fi

echo "=== overnight done $(date) ==="
