#!/usr/bin/env bash
# =============================================================================
# AIC Clean Baseline v1: 50-Epoch Production Training on ailab40 (GPU 1)
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, GPU 1)
# 机制: 纯净基准 14 项通用正确性修复后首次端到端完整训练
#   - 纯净老标注: 1800 train / 200 val (instances_train.json)
#   - 严格单卡纯 FP32 (禁用 AMP, 禁用 cuDNN, 杜绝 Volta SM_70 精度异常)
#   - 单调 Clean Lifecycle (无中途回滚, 原子保存 last/best/final)
#   - 初始 IR Stem 严格从加载后的 COCO 权重复制
#   - DINOv3 教师与 COCO 底模严格 SHA256 锁死
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ROOT="/home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${ROOT}/scripts/lib:${ROOT}/vendor/DEIMv2:${PYTHONPATH:-}"

cd "${ROOT}/vendor/DEIMv2"

mkdir -p "${ROOT}/logs"
mkdir -p "outputs/aic_clean_baseline_v1"

echo "=========================================================================="
echo "=== Launching AIC Clean Baseline v1 Training (50 Epochs, Pure FP32) ==="
echo "=========================================================================="
echo "Start Time:        $(date)"
echo "Target GPU:        Tesla V100 32GB (CUDA_VISIBLE_DEVICES=1)"
echo "Working Directory: $(pwd)"
echo "Config:            configs/deimv2/aic_clean_baseline_v1.yml"
echo "Log File:          ${ROOT}/logs/aic_clean_baseline_v1_train.log"
echo "Output Directory:  outputs/aic_clean_baseline_v1"
echo "=========================================================================="

exec $PY -u train.py \
  -c configs/deimv2/aic_clean_baseline_v1.yml \
  -u use_amp=False \
  2>&1 | tee -a "${ROOT}/logs/aic_clean_baseline_v1_train.log"
