#!/usr/bin/env bash
# =============================================================================
# train_puremal_refine_giou35.sh - 方案 A: 12-Epoch 纯净边界精炼 (GIoU 3.5x + FGL 0.20x)
# =============================================================================
set -e

REPO="/home/constant/CodeBuddy/Dev/AIC"
V2="$REPO/vendor/DEIMv2"
PY="/home/constant/miniconda3/envs/deimv2/bin/python"
LOG_DIR="$REPO/logs"
OUT_DIR="$V2/outputs/deim_hgnetv2_x_aic_puremal_refine_giou35"
START_CKPT="$V2/outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal/best_stg2.pth"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

cd "$V2"
export PYTHONPATH="$V2:$REPO/scripts/lib:${PYTHONPATH:-}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === STARTING SCHEME A: CLEAN BOUNDARY REFINEMENT (12 EPOCHS) ==="
echo "Base Checkpoint: $START_CKPT"
echo "Config: configs/deimv2/deim_hgnetv2_x_aic_puremal_refine_giou35.yml"

# 1. 启动 12-Epoch 边界精炼训练
$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_puremal_refine_giou35.yml \
    -t "$START_CKPT" \
    --use-amp \
    --seed 0 2>&1 | tee "$LOG_DIR/dfine_x_puremal_refine_giou35_train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training complete! Checking for best checkpoint..."

BEST_CKPT="$OUT_DIR/best_stg1.pth"
if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT="$OUT_DIR/last.pth"
fi
echo "Selected checkpoint: $BEST_CKPT"

# 2. 评测验证集
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating refined model on validation set..."
$PY train.py \
    -c configs/deimv2/deim_hgnetv2_x_aic_puremal_refine_giou35_val_orig.yml \
    -r "$BEST_CKPT" \
    --use-amp \
    --test-only > "$LOG_DIR/dfine_x_puremal_refine_giou35_eval_orig.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Validation evaluation finished. Log: $LOG_DIR/dfine_x_puremal_refine_giou35_eval_orig.log"

# 3. 自动生成提交包
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating official submission package..."
SUB_DIR="$REPO/work_dir/submissions/submission_puremal_refine"
mkdir -p "$SUB_DIR"

$PY "$REPO/scripts/gen/make_submission_clean_single.py" \
    --config configs/deimv2/deim_hgnetv2_x_aic_puremal_refine_giou35_val_orig.yml \
    --ckpt "$BEST_CKPT" \
    --out-dir "$SUB_DIR" \
    --zip-name submission_puremal_refine.zip \
    --max-dets 70 \
    --score-thresh 0.01

cp "$SUB_DIR/submission_puremal_refine.zip" "$REPO/submission_puremal_refine.zip" 2>/dev/null || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === SCHEME A ALL PIPELINES COMPLETED SUCCESSFULLY ==="
