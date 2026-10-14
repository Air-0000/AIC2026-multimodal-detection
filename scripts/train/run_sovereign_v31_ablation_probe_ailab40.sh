#!/bin/bash
# =============================================================================
# Sovereign V31 Ablation Probe: Controlled One-to-Many Auxiliary Supervision Study
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, GPU 1)
# =============================================================================

set -e

BASE_DIR="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
DEIM_DIR="$BASE_DIR/vendor/DEIMv2"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 使用稳定健康的 GPU 1 (Tesla V100 32GB)
export CUDA_VISIBLE_DEVICES=1

cd "$DEIM_DIR"

START_CKPT="$DEIM_DIR/outputs/champions/sovereign_v29_alltime_peak_c5_5756.pth"
OUT_DIR="outputs/sovereign_v31_ablation_probe_2560"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "=== Launching Sovereign V31 Controlled Ablation Probe (2 Epochs) ==="
echo "=========================================================================="
echo "Base Checkpoint:   $START_CKPT (Exact same starting base as V30)"
echo "Config:            configs/deimv2/sovereign_v31_ablation_probe.yml"
echo "Output Directory:  $OUT_DIR"
echo "Resolution:        1440 x 2560 (Native 2.5K)"
echo "Variable under test: Co-ATSS One-to-Many Auxiliary Supervision (use_co_aux: True)"
echo "Control Baseline:  Sovereign V30 (Exact same seed=42, lr, 2.5K without Co-Aux)"
echo "Epochs:            2 (Fast Controlled Validation Showdown)"
echo "=========================================================================="

$PY -u train.py \
  -c configs/deimv2/sovereign_v31_ablation_probe.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train_console.log"

echo "=== Sovereign V31 Ablation Probe Completed at $(date) ==="
