#!/usr/bin/env bash
# =============================================================================
# Sovereign V29 自动化无限循环迭代精炼引擎 (Continuous Cyclical Refinement Engine)
# 机制:
#   1. 监听当前正在运行的 Cycle 2 会话直至圆满结束；
#   2. 自动捕获历代最佳权重，开启循环迭代流水线:
#      Cycle N: Step 1 (DINOv3 多模态大蒸馏 6 轮) -> Step 2 (纯净无偏退火 6 轮)
#   3. 每轮收官后自动评估 mAP，维护全域冠军权重 sovereign_v29_champion.pth；
#   4. 支持随时外部叫停: 创建 STOP_LOOP 文件即可优雅停机。
# =============================================================================

set -eo pipefail

AIC_ROOT="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
cd "$AIC_ROOT/vendor/DEIMv2"
export PYTHONPATH="$AIC_ROOT/scripts/lib:$AIC_ROOT/vendor/DEIMv2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=1
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

LOG_FILE="$AIC_ROOT/continuous_cycle.log"
STOP_FLAG="$AIC_ROOT/STOP_LOOP"
rm -f "$STOP_FLAG"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

log "======================================================================="
log "Sovereign V29 Continuous Cyclical Refinement Engine Started"
log "Hardware: ailab40 (Tesla V100 32GB, GPU 1 [Healthy SM_70])"
log "Control: Touch '$STOP_FLAG' anytime to gracefully stop after current stage"
log "======================================================================="

# -----------------------------------------------------------------------------
# 阶段 1: 守候当前正在运行的 Cycle 2 会话
# -----------------------------------------------------------------------------
log "[Phase 1] Monitoring ongoing Cycle 2 execution in tmux (v29_cycle2)..."
while tmux has-session -t v29_cycle2 2>/dev/null; do
    sleep 30
done

log "[Phase 1] Cycle 2 session finished!"

# 验证 Cycle 2 的输出
DIR_C2_S2="outputs/sovereign_v29_cycle2_stage2_anneal"
DIR_C2_S1="outputs/sovereign_v29_cycle2_stage1_distill"

if [ -f "$DIR_C2_S2/best_stg2.pth" ]; then
    CURRENT_BEST="$DIR_C2_S2/best_stg2.pth"
elif [ -f "$DIR_C2_S1/best_stg2.pth" ]; then
    CURRENT_BEST="$DIR_C2_S1/best_stg2.pth"
else
    CURRENT_BEST="outputs/sovereign_v29_stage2c_pure_annealing/best_stg2.pth"
fi

mkdir -p "$AIC_ROOT/vendor/DEIMv2/outputs/champions"
CHAMPION="$AIC_ROOT/vendor/DEIMv2/outputs/champions/sovereign_v29_champion.pth"
cp -f "$CURRENT_BEST" "$CHAMPION"
log "[Cycle 2 Settled] Champion Base initialized from: $CURRENT_BEST"

# -----------------------------------------------------------------------------
# 阶段 2: 无限循环迭代精炼主循环 (Cycle 3, Cycle 4, Cycle 5 ...)
# -----------------------------------------------------------------------------
CYCLE_NUM=3

while true; do
    if [ -f "$STOP_FLAG" ]; then
        log "======================================================================="
        log "[STOP SIGNAL DETECTED] Stop flag received. Gracefully exiting loop."
        log "Final Champion Weight Preserved at: $CHAMPION"
        log "======================================================================="
        break
    fi

    log "======================================================================="
    log ">>> ENTERING CYCLE ${CYCLE_NUM} <<<"
    log "Current Base Checkpoint: $CHAMPION"
    log "======================================================================="

    # --- Step 1: DINOv3 多模态大蒸馏 ---
    DIR_DISTILL="outputs/sovereign_v29_cycle${CYCLE_NUM}_stage1_distill"
    mkdir -p "$DIR_DISTILL"

    log "[Cycle ${CYCLE_NUM} - Step 1] Launching 6-epoch DINOv3 multimodal distillation..."
    $PY -u train.py \
        -c configs/deimv2/sovereign_v29_cycle2_stage1_distill.yml \
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
    log "[Cycle ${CYCLE_NUM} - Step 1 Complete] Best Checkpoint: $CKPT_S1"

    if [ -f "$STOP_FLAG" ]; then
        log "[STOP SIGNAL DETECTED] Stopping between steps."
        cp -f "$CKPT_S1" "$CHAMPION"
        break
    fi

    # --- Step 2: 纯净无偏退火收官 ---
    DIR_ANNEAL="outputs/sovereign_v29_cycle${CYCLE_NUM}_stage2_anneal"
    mkdir -p "$DIR_ANNEAL"

    log "[Cycle ${CYCLE_NUM} - Step 2] Launching 6-epoch pure unbiased annealing..."
    $PY -u train.py \
        -c configs/deimv2/sovereign_v29_stage2c_pure_annealing.yml \
        -t "$CKPT_S1" \
        --use-amp --seed 42 \
        --output-dir "$DIR_ANNEAL" \
        2>&1 | tee "$DIR_ANNEAL/train.log"

    if [ -f "$DIR_ANNEAL/best_stg2.pth" ]; then
        CKPT_S2="$DIR_ANNEAL/best_stg2.pth"
        cp -f "$CKPT_S2" "$CHAMPION"
        log "[Cycle ${CYCLE_NUM} - Step 2 Complete] New Champion Updated: $CHAMPION"
    else
        log "WARNING: Step 2 did not produce new best, retaining Step 1 checkpoint."
        cp -f "$CKPT_S1" "$CHAMPION"
    fi

    log "[CYCLE ${CYCLE_NUM} COMPLETE] Ready for next cycle."
    CYCLE_NUM=$((CYCLE_NUM + 1))
done

log "All operations completed cleanly. Final delivered model: $CHAMPION"
