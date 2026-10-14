#!/usr/bin/env bash
# =============================================================================
# run_dog_ablation.sh - 单测 3: 深度阶跃遮挡边界门控 (DOG) 单变量消融实验
# -----------------------------------------------------------------------------
# 严格单变量受控天梯对照:
#   1. Base: HGNetv2-B5 + 1ch IR + CSSA (无视差对齐) -> 锁定 0.5413 AP
#   2. 单测 1 (DCDA): 开启深度引导视差可变形对齐 -> 锁定 0.5490 AP (+0.77%)
#   3. 单测 2 (3D-Lift): 固化 DCDA + 开启 3D 物理空间坐标升维 -> 锁定 0.5585 AP (+1.72%)
#   4. 单测 3 (DOG): 固化 DCDA + 3D-Lift + 仅开启 use_dog: True
#      (计算深度梯度模长 |∇Z|，阻断前后景重叠处的红外热晕泼溅)
# -----------------------------------------------------------------------------
# 起始权重: outputs/deim_hgnetv2_x_aic_3dlift_mm/best_stg1.pth (锁定 0.5585 成果)
# =============================================================================
set -e

REPO="/home/constant/CodeBuddy/Dev/AIC"
V2="$REPO/vendor/DEIMv2"
PY="/home/constant/miniconda3/envs/deimv2/bin/python"
LOG_DIR="$REPO/logs"
START_CKPT="$V2/outputs/deim_hgnetv2_x_aic_3dlift_mm/best_stg1.pth"

mkdir -p "$LOG_DIR"
cd "$V2"
export PYTHONPATH="$V2:$REPO/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === DOG ABLATION RUNNER INITIALIZED ==="
echo "Base Checkpoint (3D-Lift 0.5585): $START_CKPT"

# ---- 启动单测 3 (DOG 微调) ----
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Single Test 3 (DCDA + 3D-Lift + DOG enabled)..."
$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_dog_mm.yml \
    -t "$START_CKPT" \
    --use-amp 2>&1 | tee "$LOG_DIR/dog_ablation_train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished. Evaluating on 200 validation images..."
BEST_DOG="$V2/outputs/deim_hgnetv2_x_aic_dog_mm/best_stg2.pth"
if [ ! -f "$BEST_DOG" ]; then
    BEST_DOG="$V2/outputs/deim_hgnetv2_x_aic_dog_mm/best_stg1.pth"
fi
if [ ! -f "$BEST_DOG" ]; then
    BEST_DOG="$V2/outputs/deim_hgnetv2_x_aic_dog_mm/last.pth"
fi

$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_dog_mm.yml \
    -r "$BEST_DOG" \
    --use-amp \
    --test-only 2>&1 | tee "$LOG_DIR/dog_ablation_eval.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === DOG ABLATION COMPLETE ==="
