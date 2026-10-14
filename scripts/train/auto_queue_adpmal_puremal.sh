#!/usr/bin/env bash
# =============================================================================
# auto_queue_adpmal_puremal.sh - 自动化流水线
# 1. 守候 PID 28677 (Class-Adaptive MAL 训练) 完成
# 2. 评测 adpmal 并自动生成 Phase 2 提交包 submission_adpmal.zip
# 3. 立即启动缺失的消融实验: ft_puremal (纯 MAL2 无 DCN)
# 4. 评测 ft_puremal 并汇总完整 2x2 消融矩阵
# =============================================================================
set -u
ulimit -n 65535

REPO="/home/constant/CodeBuddy/Dev/AIC"
V2="$REPO/vendor/DEIMv2"
PY="/home/constant/miniconda3/envs/deimv2/bin/python"
LOG_DIR="$REPO/logs"
QUEUE_LOG="$LOG_DIR/auto_queue.log"

export PYTHONPATH="$V2:$REPO/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$V2"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

log "=== AUTO QUEUE MONITOR STARTED ==="
TARGET_PID=28677
log "Waiting for PID $TARGET_PID (AdpMAL training) to finish..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 30
done

log "PID $TARGET_PID finished! Checking AdpMAL output..."

# 1. 查找 AdpMAL 最优权重
ADPMAL_DIR="$V2/outputs/deim_hgnetv2_x_aic_obj2coco_ft_adpmal"
ADPMAL_CKPT="$ADPMAL_DIR/best_stg2.pth"
if [ ! -f "$ADPMAL_CKPT" ]; then
    ADPMAL_CKPT="$ADPMAL_DIR/best_stg1.pth"
fi
if [ ! -f "$ADPMAL_CKPT" ]; then
    ADPMAL_CKPT=$(ls -t "$ADPMAL_DIR"/*.pth 2>/dev/null | head -1)
fi

if [ -n "$ADPMAL_CKPT" ] && [ -f "$ADPMAL_CKPT" ]; then
    log "AdpMAL checkpoint found: $ADPMAL_CKPT"
    
    # 评测 AdpMAL 验证集
    log "Running AdpMAL evaluation on validation set..."
    $PY train.py -c configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_adpmal_val_orig.yml \
        -r "$ADPMAL_CKPT" --use-amp --test-only > "$LOG_DIR/dfine_x_adpmal_eval_orig.log" 2>&1
    log "AdpMAL val eval finished. Log: $LOG_DIR/dfine_x_adpmal_eval_orig.log"
    
    # 生成 AdpMAL Phase 2 提交包
    log "Generating submission_adpmal.zip for Phase 2..."
    mkdir -p "$REPO/work_dir/submissions/submission_adpmal"
    $PY "$REPO/scripts/gen/make_submission_opt_ab.py" \
        --config configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_adpmal_val_orig.yml \
        --ckpt "$ADPMAL_CKPT" \
        --out-dir "$REPO/work_dir/submissions/submission_adpmal" \
        --zip-name submission_adpmal.zip > "$LOG_DIR/gen_submission_adpmal.log" 2>&1
    log "AdpMAL Phase 2 submission generated successfully!"
else
    log "WARNING: No AdpMAL checkpoint found in $ADPMAL_DIR!"
fi

# 2. 启动缺失的消融实验: ft_puremal (纯 MAL2 无 DCN)
log "=== STARTING MISSING ABLATION: ft_puremal (Pure MAL2, No DCN) ==="
PUREMAL_CFG="configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal.yml"
PUREMAL_OUT="./outputs/deim_hgnetv2_x_aic_obj2coco_ft_puremal"
BASE_CKPT="ckpts/dfine_x_obj2coco.pth"

$PY train.py -c "$PUREMAL_CFG" -t "$BASE_CKPT" --use-amp --seed 0 > "$LOG_DIR/dfine_x_puremal_train.log" 2>&1
PUREMAL_RC=$?
log "ft_puremal training finished with rc=$PUREMAL_RC"

if [ $PUREMAL_RC -eq 0 ]; then
    PUREMAL_CKPT="$PUREMAL_OUT/best_stg2.pth"
    [ ! -f "$PUREMAL_CKPT" ] && PUREMAL_CKPT="$PUREMAL_OUT/best_stg1.pth"
    [ ! -f "$PUREMAL_CKPT" ] && PUREMAL_CKPT=$(ls -t "$PUREMAL_OUT"/*.pth 2>/dev/null | head -1)
    
    if [ -n "$PUREMAL_CKPT" ] && [ -f "$PUREMAL_CKPT" ]; then
        log "Evaluating ft_puremal on validation set..."
        $PY train.py -c configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_puremal_val_orig.yml \
            -r "$PUREMAL_CKPT" --use-amp --test-only > "$LOG_DIR/dfine_x_puremal_eval_orig.log" 2>&1
        log "ft_puremal evaluation finished."
    fi
fi

# 3. 汇总消融指标
log "Generating unified ablation matrix summary..."
$PY "$REPO/scripts/eval/summarize_ablation_matrix.py" >> "$QUEUE_LOG" 2>&1 || true

log "=== ALL TASKS COMPLETED SUCCESSFULLY ==="
