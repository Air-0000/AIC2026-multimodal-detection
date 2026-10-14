#!/bin/bash
set -e

BASE_DIR="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
DEIM_DIR="$BASE_DIR/vendor/DEIMv2"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 必须使用稳定健康的 GPU 1 (Tesla V100 32GB)
export CUDA_VISIBLE_DEVICES=1

cd "$DEIM_DIR"
mkdir -p outputs/sovereign_v30_cssp_v29peak_2560

START_CKPT="$DEIM_DIR/outputs/champions/sovereign_v29_alltime_peak_c5_5756.pth"
if [ ! -f "$START_CKPT" ]; then
    START_CKPT="$DEIM_DIR/outputs/sovereign_v29_cycle5_stage2_anneal_2560/best_stg2.pth"
fi

if [ ! -f "$START_CKPT" ]; then
    echo "[-] Error: Start checkpoint $START_CKPT not found!"
    exit 1
fi

echo "=========================================================================="
echo "=== Launching Sovereign V30 Tuning on V29 Cycle 5 Peak (58.08% / UAV 100%) ==="
echo "=========================================================================="
echo "Base Checkpoint:   $START_CKPT"
echo "Config:            configs/deimv2/sovereign_v30_cssp_v29peak_2560.yml"
echo "Output Directory:  outputs/sovereign_v30_cssp_v29peak_2560"
echo "Resolution:        1440 x 2560 (Native 2.5K Ultra-High Resolution)"
echo "Core Module:       CSSP Cross-Spectral Diff + ABC Local Contrast Attention"
echo "Epochs:            8 (Fast Adaptation & Saliency Wake-Up)"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "=========================================================================="

$PY -u train.py \
  -c configs/deimv2/sovereign_v30_cssp_v29peak_2560.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  --output-dir outputs/sovereign_v30_cssp_v29peak_2560 \
  2>&1 | tee -a outputs/sovereign_v30_cssp_v29peak_2560/train_console.log

echo "=== Sovereign V30 Tuning on V29 Peak Completed at $(date) ==="
