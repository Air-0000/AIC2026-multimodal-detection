#!/bin/bash
set -e

BASE_DIR="/home/ailab/ssd/bigdata/liuzhengyang"
DEIM_DIR="$BASE_DIR/deimv2"
PY="/home/ailab40/anaconda3/envs/rfdetr/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export TORCH_CUDA_MATMUL_NO_CUBLASLT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

cd "$DEIM_DIR"
mkdir -p outputs/sovereign_v30_cssp_ailab40

START_CKPT="$BASE_DIR/weights/sovereign_v30_cssp_best_stg2.pth"
if [ ! -f "$START_CKPT" ]; then
    START_CKPT="$BASE_DIR/weights/sovereign_v20_pure_lmm_soup_585.pth"
fi

echo "=========================================================================="
echo "=== Launching Sovereign V30 CSSP + ABC Fine-Tuning on ailab40 (2x V100) ==="
echo "=========================================================================="
echo "Start Checkpoint:  $START_CKPT"
echo "Config:            configs/deimv2/sovereign_v30_cssp_ailab40.yml"
echo "Output Directory:  outputs/sovereign_v30_cssp_ailab40"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "=========================================================================="

$PY train.py \
  -c configs/deimv2/sovereign_v30_cssp_ailab40.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a outputs/sovereign_v30_cssp_ailab40/train_console.log

echo "=== Sovereign V30 Tuning on ailab40 Completed at $(date) ==="
