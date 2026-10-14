#!/usr/bin/env bash
# =============================================================================
# run_3dlift_ablation.sh - 单测 2: 3D 物理空间坐标升维 (3D-Lift) 单变量消融实验
# -----------------------------------------------------------------------------
# 严格单变量受控天梯对照:
#   1. Base: HGNetv2-B5 + 1ch IR + CSSA (无视差对齐) -> 锁定 0.5413 AP
#   2. 单测 1 (DCDA): 开启深度引导视差可变形对齐 -> 锁定 0.5490 AP (+0.77%, 路灯+2.58%)
#   3. 单测 2 (3D-Lift): 固化 DCDA + 仅开启 use_3d_lift: True
#      (基于真实 16-bit 深度 Z 反投影物理点云 (X,Y,Z)，注入几何空间场)
# -----------------------------------------------------------------------------
# 起始权重: outputs/deim_hgnetv2_x_aic_dcda_mm/best_stg2.pth (锁定 0.5490 成果)
# =============================================================================
set -e

REPO="/home/constant/CodeBuddy/Dev/AIC"
V2="$REPO/vendor/DEIMv2"
PY="/home/constant/miniconda3/envs/deimv2/bin/python"
LOG_DIR="$REPO/logs"
START_CKPT="$V2/outputs/deim_hgnetv2_x_aic_dcda_mm/best_stg2.pth"

mkdir -p "$LOG_DIR"
cd "$V2"
export PYTHONPATH="$V2:$REPO/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 3D-LIFT ABLATION RUNNER INITIALIZED ==="
echo "Base DCDA Checkpoint: $START_CKPT"

# ---- GPU 排队守护机制: 检测前序显卡推理进程 ----
TARGET_PID=22960
if ps -p $TARGET_PID > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU is currently generating submission package (PID $TARGET_PID). Waiting..."
    while ps -p $TARGET_PID > /dev/null 2>&1; do
        sleep 5
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submission generation completed! Claiming GPU for 3D-Lift ablation..."
    sleep 3
fi

# ---- 启动单测 2 (3D-Lift 微调) ----
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Single Test 2 (DCDA + 3D-Lift enabled)..."
$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_3dlift_mm.yml \
    -t "$START_CKPT" \
    --use-amp 2>&1 | tee "$LOG_DIR/3dlift_ablation_train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished. Evaluating on 200 validation images..."
BEST_3D="$V2/outputs/deim_hgnetv2_x_aic_3dlift_mm/best_stg2.pth"
if [ ! -f "$BEST_3D" ]; then
    BEST_3D="$V2/outputs/deim_hgnetv2_x_aic_3dlift_mm/best_stg1.pth"
fi
if [ ! -f "$BEST_3D" ]; then
    BEST_3D="$V2/outputs/deim_hgnetv2_x_aic_3dlift_mm/last.pth"
fi

$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_3dlift_mm.yml \
    -r "$BEST_3D" \
    --use-amp \
    --test-only 2>&1 | tee "$LOG_DIR/3dlift_ablation_eval.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 3D-LIFT ABLATION COMPLETE ==="
