#!/bin/bash
# =============================================================================
# Sovereign V32 Production: Physical Inductive Bias & Modality Grounding (PIG)
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, GPU 1)
# 核心机制:
#   1. 承接 Sovereign V31 Production Peak 巅峰检查点 (best_stg2.pth, mAP 0.5729)
#   2. 激活 Physical Inductive Gating: 局域热辐射差分 (Delta T) 与深度共面性残差
#   3. 激活 Physical Consistency Loss: 严厉惩罚冷背景虚假冒充 (海报清零)
#   4. 物理增益激活: 强热斑正向强化 (拯救低对比度白机)
#   5. 保持 Co-ATSS 一对多辅助密集监督与 2.5K 超清原生尺度 (1440 x 2560)
# =============================================================================

set -e

BASE_DIR="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
DEIM_DIR="$BASE_DIR/vendor/DEIMv2"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

cd "$DEIM_DIR"

START_CKPT="$DEIM_DIR/outputs/sovereign_v31_production_full8ep_2560/best_stg2.pth"
OUT_DIR="outputs/sovereign_v32_physics_grounded_2560"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "=== Launching Sovereign V32 Physical Inductive Grounding (PIG) 8-Epoch Training ==="
echo "=========================================================================="
echo "Start Time:        $(date)"
echo "Base Checkpoint:   $START_CKPT"
echo "Config:            configs/deimv2/sovereign_v32_physics_grounded_2560.yml"
echo "Output Directory:  $OUT_DIR"
echo "Resolution:        1440 x 2560 (Native 2.5K)"
echo "Physics Module:    Physical Inductive Gating (PIG) + Consistency Loss"
echo "Aux Head:          Co-ATSS One-to-Many Auxiliary Supervision Head"
echo "Epochs:            8 (Full Production Annealing)"
echo "Schedule:          flatcosine (flat_epoch=2, lr_gamma=0.15)"
echo "=========================================================================="

$PY -u train.py \
  -c configs/deimv2/sovereign_v32_physics_grounded_2560.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train_console.log"

echo "=========================================================================="
echo "=== Sovereign V32 Production Training Completed at $(date) ==="
echo "=========================================================================="
