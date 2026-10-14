#!/usr/bin/env bash
# =============================================================================
# Sovereign V29 Ultra-High Resolution 2048 无限循环迭代精炼引擎
# 机制:
#   1. 承接 Cycle 4 终态冠军权重 sovereign_v29_champion.pth
#   2. 升级至满血 1152 x 2048 超高分辨率 (Batch 2 + Accum 2, 等效物理 Batch 4)
#   3. Cycle 5+:
#      Step 1: sovereign_v29_highres2048_distill.yml (6 轮 2048 大蒸馏)
#      Step 2: sovereign_v29_highres2048_anneal.yml  (6 轮 2048 无偏退火)
#   4. 实时更新 sovereign_v29_champion.pth
# =============================================================================

set -eo pipefail

AIC_ROOT="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
cd "$AIC_ROOT/vendor/DEIMv2"
export PYTHONPATH="$AIC_ROOT/scripts/lib:$AIC_ROOT/vendor/DEIMv2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=1
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

LOG_FILE="$AIC_ROOT/continuous_cycle_2048.log"
STOP_FLAG="$AIC_ROOT/STOP_LOOP_2048"
rm -f "$STOP_FLAG"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

log "======================================================================="
log "Sovereign V29 High-Res 2048 Continuous Engine Initialized"
log "Hardware: ailab40 (Tesla V100 32GB, GPU 1 [Healthy SM_70])"
log "Resolution: 1152 x 2048 | Micro-batch: 2 | Accum: 2 (Effective Batch: 4)"
log "Control: Touch '$STOP_FLAG' anytime to gracefully stop after current stage"
log "======================================================================="

CHAMPION="$AIC_ROOT/vendor/DEIMv2/outputs/champions/sovereign_v29_champion.pth"

# 阶段 1: 守候当前正在运行的 Cycle 4 会话完全收官
log "[Phase 1] Waiting for Cycle 4 daemon to cleanly conclude..."
while tmux has-session -t v29_daemon 2>/dev/null; do
    sleep 15
done

log "[Phase 1] Cycle 4 daemon finished! Baseline Champion Checkpoint locked."
log "Champion Weight: $CHAMPION (Size: $(du -h "$CHAMPION" | cut -f1))"

# 阶段 2: 启动 2048 超高分辨率迭代循环 (Cycle 5, Cycle 6 ...)
CYCLE_NUM=5

while true; do
    if [ -f "$STOP_FLAG" ]; then
        log "======================================================================="
        log "[STOP SIGNAL DETECTED] Stop flag received. Gracefully exiting 2048 loop."
        log "Final Champion Weight Preserved at: $CHAMPION"
        log "======================================================================="
        break
    fi

    log "======================================================================="
    log ">>> ENTERING 2048 HIGH-RES CYCLE ${CYCLE_NUM} <<<"
    log "Current Base Checkpoint: $CHAMPION"
    log "Resolution: [1152, 2048] | Effective Batch: 4"
    log "======================================================================="

    # --- Step 1: 2048 DINOv3 多模态大蒸馏 ---
    DIR_DISTILL="outputs/sovereign_v29_cycle${CYCLE_NUM}_stage1_distill_2048"
    mkdir -p "$DIR_DISTILL"

    log "[Cycle ${CYCLE_NUM} (2048) - Step 1] Launching 6-epoch 2048 DINOv3 multimodal distillation..."
    $PY -u train.py \
        -c configs/deimv2/sovereign_v29_highres2048_distill.yml \
        -t "$CHAMPION" \
        --use-amp --seed 42 \
        --output-dir "$DIR_DISTILL" \
        2>&1 | tee "$DIR_DISTILL/train.log"

    if [ -f "$DIR_DISTILL/best_stg2.pth" ]; then
        CKPT_S1="$DIR_DISTILL/best_stg2.pth"
    elif [ -f "$DIR_DISTILL/best_stg1.pth" ]; then
        CKPT_S1="$DIR_DISTILL/best_stg1.pth"
    else
        CKPT_S1="$CHAMPION"
        log "WARNING: Step 1 did not produce new best, retaining previous champion."
    fi
    log "[Cycle ${CYCLE_NUM} (2048) - Step 1 Complete] Best Checkpoint: $CKPT_S1"

    if [ -f "$STOP_FLAG" ]; then
        log "[STOP SIGNAL DETECTED] Stopping between steps."
        cp -f "$CKPT_S1" "$CHAMPION"
        break
    fi

    # --- Step 2: 2048 纯净无偏退火收官 ---
    DIR_ANNEAL="outputs/sovereign_v29_cycle${CYCLE_NUM}_stage2_anneal_2048"
    mkdir -p "$DIR_ANNEAL"

    log "[Cycle ${CYCLE_NUM} (2048) - Step 2] Launching 6-epoch 2048 pure unbiased annealing..."
    $PY -u train.py \
        -c configs/deimv2/sovereign_v29_highres2048_anneal.yml \
        -t "$CKPT_S1" \
        --use-amp --seed 42 \
        --output-dir "$DIR_ANNEAL" \
        2>&1 | tee "$DIR_ANNEAL/train.log"

    if [ -f "$DIR_ANNEAL/best_stg2.pth" ]; then
        CKPT_S2="$DIR_ANNEAL/best_stg2.pth"
        cp -f "$CKPT_S2" "$CHAMPION"
        log "[Cycle ${CYCLE_NUM} (2048) - Step 2 Complete] New Champion Updated: $CHAMPION"
    else
        log "WARNING: Step 2 did not produce new best, retaining Step 1 checkpoint."
        cp -f "$CKPT_S1" "$CHAMPION"
    fi

    log "[CYCLE ${CYCLE_NUM} (2048) COMPLETE] Ready for next cycle."
    CYCLE_NUM=$((CYCLE_NUM + 1))
done

log "All 2048 operations completed cleanly. Final delivered model: $CHAMPION"
