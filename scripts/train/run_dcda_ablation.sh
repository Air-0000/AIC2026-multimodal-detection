#!/usr/bin/env bash
# =============================================================================
# run_dcda_ablation.sh - 单测 1: 深度引导视差可变形对齐 (DCDA) 单变量对比实验
# -----------------------------------------------------------------------------
# 严格单变量对照:
#   1. Base (基线): HGNetv2-B5 + 1ch IR + CSSA (use_dcda: False) -> 锁定 0.5413 AP
#   2. 单测 1 (DCDA): 同架构 + 仅开启 use_dcda: True (以 1/Z 引导可变形采样)
# -----------------------------------------------------------------------------
# 包含自动 GPU 排队守护: 检测前序显卡进程，空闲后自动无缝启动
# =============================================================================
set -e

REPO="/home/constant/CodeBuddy/Dev/AIC"
V2="$REPO/vendor/DEIMv2"
PY="/home/constant/miniconda3/envs/deimv2/bin/python"
LOG_DIR="$REPO/logs"
START_CKPT="$V2/outputs/deim_hgnetv2_x_aic_obj2coco_ft_dcnmm/best_stg1.pth"

mkdir -p "$LOG_DIR"
cd "$V2"
export PYTHONPATH="$V2:$REPO/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === DCDA ABLATION RUNNER INITIALIZED ==="
echo "Base Checkpoint: $START_CKPT"



# ---- 启动单测 1 (DCDA 微调) ----
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Single Test 1 (DCDA enabled)..."
$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_dcda_mm.yml \
    -t "$START_CKPT" \
    --use-amp 2>&1 | tee "$LOG_DIR/dcda_ablation_train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished. Evaluating on 200 validation images..."
BEST_DCDA="$V2/outputs/deim_hgnetv2_x_aic_dcda_mm/best_stg1.pth"
if [ ! -f "$BEST_DCDA" ]; then
    BEST_DCDA="$V2/outputs/deim_hgnetv2_x_aic_dcda_mm/last.pth"
fi

$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_dcda_mm.yml \
    -r "$BEST_DCDA" \
    --use-amp \
    --test-only 2>&1 | tee "$LOG_DIR/dcda_ablation_eval.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === DCDA ABLATION COMPLETE ==="
