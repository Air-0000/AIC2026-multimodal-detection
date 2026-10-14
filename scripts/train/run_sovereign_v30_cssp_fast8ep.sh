#!/bin/bash
set -e

ROOT="/home/constant/CodeBuddy/Dev/AIC"
PY="/home/constant/miniconda3/envs/deimv2/bin/python"

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/scripts/lib:$ROOT/vendor/DEIMv2:${PYTHONPATH:-}"

cd "$ROOT/vendor/DEIMv2"
mkdir -p outputs/sovereign_v30_cssp_fast8ep

START_CKPT="$ROOT/vendor/DEIMv2/outputs/sovereign_v28_disparity_awakening_2048/best_stg1.pth"
if [ ! -f "$START_CKPT" ]; then
    echo "[-] Error: Start checkpoint $START_CKPT not found!"
    exit 1
fi

echo "=========================================================================="
echo "=== Launching Sovereign V30 CSSP + ABC Fast Fine-Tuning (8 Epochs)   ==="
echo "=========================================================================="
echo "Start Checkpoint:  $START_CKPT"
echo "Config:            configs/deimv2/sovereign_v30_cssp_fast8ep.yml"
echo "Output Directory:  outputs/sovereign_v30_cssp_fast8ep"
echo "CSSP Module Params: 30.24k (Dedicated LR: 1e-4, Zero-Init Step 0 Identity)"
echo "Target:            8 Epochs Fast Adaptation & Saliency Wake-Up"
echo "Timestamp:         $(date)"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "=========================================================================="

$PY train.py \
  -c configs/deimv2/sovereign_v30_cssp_fast8ep.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a outputs/sovereign_v30_cssp_fast8ep/train_console.log

echo "=== Sovereign V30 Fast Tuning Completed at $(date) ==="
