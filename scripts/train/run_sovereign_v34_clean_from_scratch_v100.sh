#!/bin/bash
# =============================================================================
# Sovereign V34 Clean-Slate: 50-Epoch Full Production Training from Open COCO Base
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, 稳定健康的 GPU 1)
# 机制:
#   1. 纯血开源底模: ckpts/dfine_x_obj2coco.pth (零参赛历史污染)
#   2. 知识蒸馏: Meta DINOv3-ViT-Large (300M) 在线特征微创蒸馏 (帮助 RGB 快速学习)
#   3. 密集正反馈: Co-ATSS 一对多辅助密集监督头 (小样本快速收敛，推理零开销切除)
#   4. 极简骨干: 纯 HGNetv2-X + DCDA 视差补偿 + DOG 深度门控 (切除冗余分支)
#   5. 纯净老标: 100% 官方原始标注 (instances_train.json 1800 图)
#   6. 50 轮全量生产排程: 前 46 轮多模态特征联合学习，后 4 轮关停增广几何吸附退火
# =============================================================================

set -e

BASE_DIR="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
DEIM_DIR="$BASE_DIR/vendor/DEIMv2"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 锁定使用稳定健康的 GPU 1 (Tesla V100 32GB)
export CUDA_VISIBLE_DEVICES=1

cd "$DEIM_DIR"

START_CKPT="$DEIM_DIR/ckpts/dfine_x_obj2coco.pth"
OUT_DIR="outputs/sovereign_v34_clean_from_scratch_2560"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "=== Launching Sovereign V34 Clean-Slate Full 50-Epoch Training on V100 ==="
echo "=========================================================================="
echo "Start Time:        $(date)"
echo "Base Checkpoint:   $START_CKPT (Pure Open COCO Base, NO competition inheritance)"
echo "Config:            configs/deimv2/sovereign_v34_clean_from_scratch_2560.yml"
echo "Output Directory:  $OUT_DIR"
echo "Resolution:        1440 x 2560 (Native 2.5K)"
echo "Knowledge Distill: Meta DINOv3-ViT-Large 300M (Online Feature Distillation)"
echo "Dense Aux Head:    Co-ATSS One-to-Many Auxiliary Supervision"
echo "Parallax Align:    DCDA Depth-Conditioned Deformable Alignment"
echo "Annotations:       100% Official instances_train.json (1,800 images)"
echo "Epochs:            50 (Full Production Clean Slate)"
echo "Schedule:          flatcosine (flat_epoch=32, no_aug_epoch=4)"
echo "=========================================================================="

$PY -u train.py \
  -c configs/deimv2/sovereign_v34_clean_from_scratch_2560.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train_console.log"

echo "=========================================================================="
echo "=== Sovereign V34 Clean-Slate Training Completed at $(date) ==="
echo "=========================================================================="
