#!/bin/bash
# =============================================================================
# Sovereign V35 Distinct Queries: Full Production Clean-Slate 50-Epoch Training
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, 稳定健康的 GPU 1)
# 架构范式: 选项 B (保留 Co-ATSS 辅助密集监督 + 解码器入口 3x3 空间局部极值排斥)
# 机制:
#   1. 纯血开源底模: ckpts/dfine_x_obj2coco.pth (零参赛历史污染)
#   2. 选项 B 破局: 保留 Co-ATSS (96.1% 极限召回)，引入入口空间排斥 (杜绝同目标多框)
#   3. 知识蒸馏: Meta DINOv3-ViT-Large (300M) 在线特征微创蒸馏 (帮助 RGB 快速学习)
#   4. 极简骨干: 纯 HGNetv2-X + DCDA 视差补偿 + DOG 深度门控 (切除冗余分支)
#   5. 纯净老标: 100% 官方原始标注 (instances_train.json 1800 图)
#   6. 50 轮全量生产排程: flat_epoch=30, lr_gamma=0.5 退火, no_aug_epoch=4 几何吸附
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
OUT_DIR="outputs/sovereign_v35_distinct_queries_2560"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "=== Launching Sovereign V35 Distinct-Query Full 50-Epoch Training on V100 ==="
echo "=========================================================================="
echo "Start Time:        $(date)"
echo "Base Checkpoint:   $START_CKPT (Pure Open COCO Base, NO competition inheritance)"
echo "Config:            configs/deimv2/sovereign_v35_distinct_queries_2560.yml"
echo "Output Directory:  $OUT_DIR"
echo "Resolution:        1440 x 2560 (Native 2.5K)"
echo "Paradigm:          Option B (Co-ATSS Dense Aux + Distinct Query Selection)"
echo "Knowledge Distill: Meta DINOv3-ViT-Large 300M (Online Feature Distillation)"
echo "Dense Aux Head:    Co-ATSS One-to-Many Auxiliary Supervision"
echo "Distinct Filter:   DFINETransformer SDQS (3x3 Local-Max Spatial Filter)"
echo "Parallax Align:    DCDA Depth-Conditioned Deformable Alignment"
echo "Annotations:       100% Official instances_train.json (1,800 images)"
echo "Epochs:            50 (Full Production Clean Slate)"
echo "Schedule:          flatcosine (flat_epoch=30, lr_gamma=0.5, no_aug_epoch=4)"
echo "=========================================================================="

$PY -u train.py \
  -c configs/deimv2/sovereign_v35_distinct_queries_2560.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train_console.log"

echo "=========================================================================="
echo "=== Sovereign V35 Distinct-Query Training Completed at $(date) ==="
echo "=========================================================================="
