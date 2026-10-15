#!/bin/bash
# =============================================================================
# Sovereign V36: P2 (Stride=4) High-Resolution Direct Injection & Guidance
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, 稳定健康的 GPU 0)
# 架构范式: 方案 C (P2 Stride=4 释放 + P2-CSRG+ 小波高频保峰 + 空间排斥独占查询 + Co-ATSS + DINOv3)
# 机制:
#   1. 纯血开源底模: ckpts/dfine_x_obj2coco.pth (零参赛历史污染)
#   2. 方案 C 核心: 释放 Stage 0 (Stride 4, P2, 360x640 高分辨率特征)，专攻微小无人机、远景小行人
#   3. P2-CSRG+ 直通引导: Haar 小波分解 + LMM 正交双向十字卷积 (3x7 水平 + 7x3 垂直) + CSR 空间门控
#   4. 空间排斥独占查询: DFINETransformer SDQS (3x3 空间局部极值排斥，彻底清除同目标多框)
#   5. 知识蒸馏: Meta DINOv3-ViT-Large (300M) 在线特征微创蒸馏 (帮助 RGB 快速学习)
#   6. 密集正反馈: Co-ATSS 一对多辅助密集监督头 (保障 96%+ 极限召回)
#   7. 纯净老标: 100% 官方原始标注 (instances_train.json 1800 图)
#   8. 50 轮全量生产排程: flat_epoch=30, lr_gamma=0.5 退火, no_aug_epoch=4 几何吸附
# =============================================================================

set -e

BASE_DIR="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
DEIM_DIR="$BASE_DIR/vendor/DEIMv2"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 锁定使用空闲的 GPU 0 (Tesla V100 32GB)
export CUDA_VISIBLE_DEVICES=0

cd "$DEIM_DIR"

START_CKPT="$DEIM_DIR/ckpts/dfine_x_obj2coco.pth"
OUT_DIR="outputs/sovereign_v36_p2_direct_injection_2560"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "=== Launching Sovereign V36 P2 Direct-Injection 50-Epoch Training (GPU 0) ==="
echo "=========================================================================="
echo "Start Time:        $(date)"
echo "Target GPU:        Tesla V100 32GB (GPU 0)"
echo "Base Checkpoint:   $START_CKPT (Pure Open COCO Base, NO competition inheritance)"
echo "Config:            configs/deimv2/sovereign_v36_p2_direct_injection_2560.yml"
echo "Output Directory:  $OUT_DIR"
echo "Resolution:        1440 x 2560 (Native 2.5K)"
echo "Paradigm:          Option C (P2 Stride=4 Direct Injection + Co-ATSS + SDQS)"
echo "P2 Module:         P2-CSRG+ (MWHL Haar Wavelet + LMM 3x7+7x3 + CSR Spatial Gate)"
echo "Trunk Bridge:      Stage 0 (Stride 4) -> Stage 1 P3 (Stride 8) Direct Residual"
echo "Knowledge Distill: Meta DINOv3-ViT-Large 300M (Online Feature Distillation)"
echo "Dense Aux Head:    Co-ATSS One-to-Many Auxiliary Supervision"
echo "Distinct Filter:   DFINETransformer SDQS (3x3 Local-Max Spatial Filter)"
echo "Parallax Align:    DCDA Depth-Conditioned Deformable Alignment"
echo "Annotations:       100% Official instances_train.json (1,800 images)"
echo "Epochs:            50 (Full Production Clean Slate)"
echo "Schedule:          flatcosine (flat_epoch=30, lr_gamma=0.5, no_aug_epoch=4)"
echo "=========================================================================="

# 关键：GPU 0 采用纯 FP32 训练，规避 Volta GPC 5 硬件寄存器缺陷 (峰值显存 17.9GB < 32GB)
$PY -u train.py \
  -c configs/deimv2/sovereign_v36_p2_direct_injection_2560.yml \
  -t "$START_CKPT" \
  --seed 42 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train_console.log"

echo "=========================================================================="
echo "=== Sovereign V36 P2 Direct-Injection Training Completed at $(date) ==="
echo "=========================================================================="
