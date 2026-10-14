#!/usr/bin/env bash
set -eo pipefail

AIC_ROOT="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
cd "$AIC_ROOT/vendor/DEIMv2"
export PYTHONPATH="$AIC_ROOT/scripts/lib:$AIC_ROOT/vendor/DEIMv2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=1
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

LOG_FILE="$AIC_ROOT/cycle2_pipeline.log"
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

log "==============================================================="
log "Sovereign V29 Cycle 2 Multi-Modal Pipeline: Distill -> Anneal"
log "Hardware: ailab40 (Tesla V100 32GB, GPU 1 [Healthy SM_70])"
log "==============================================================="

# Base Checkpoint from Cycle 1 Stage 2-C (mAP 0.5167)
CKPT_BASE="outputs/sovereign_v29_stage2c_pure_annealing/best_stg2.pth"
if [ ! -f "$CKPT_BASE" ]; then
    log "ERROR: Base checkpoint $CKPT_BASE not found!"
    exit 1
fi
log "[Base Checkpoint Verified] $CKPT_BASE"

# ==============================================================
# Cycle 2 - Step 1: 多模态全要素 + DINOv3-ViT-L 大模型语义自蒸馏 (6 轮)
# ==============================================================
DIR_C2_S1="outputs/sovereign_v29_cycle2_stage1_distill"
mkdir -p "$DIR_C2_S1"

log "==============================================================="
log "[Cycle 2 - Step 1] Starting 6-epoch DINOv3 teacher semantic distillation..."
log "[Cycle 2 - Step 1] Chaining from: $CKPT_BASE"

$PY -u train.py \
    -c configs/deimv2/sovereign_v29_cycle2_stage1_distill.yml \
    -t "$CKPT_BASE" \
    --use-amp --seed 42 \
    2>&1 | tee "$DIR_C2_S1/train.log"

if [ -f "$DIR_C2_S1/best_stg2.pth" ]; then
    CKPT_C2_S1="$DIR_C2_S1/best_stg2.pth"
elif [ -f "$DIR_C2_S1/best_stg1.pth" ]; then
    CKPT_C2_S1="$DIR_C2_S1/best_stg1.pth"
elif [ -f "$DIR_C2_S1/last.pth" ]; then
    CKPT_C2_S1="$DIR_C2_S1/last.pth"
else
    log "ERROR: Cycle 2 Step 1 did not produce a valid checkpoint!"
    exit 1
fi
log "[Cycle 2 - Step 1 Complete] Best Checkpoint: $CKPT_C2_S1"

# ==============================================================
# Cycle 2 - Step 2: 纯净无偏退火收官 (6 轮)
# ==============================================================
DIR_C2_S2="outputs/sovereign_v29_cycle2_stage2_anneal"
mkdir -p "$DIR_C2_S2"

log "==============================================================="
log "[Cycle 2 - Step 2] Starting 6-epoch pure annealing..."
log "[Cycle 2 - Step 2] Chaining from: $CKPT_C2_S1"

$PY -u train.py \
    -c configs/deimv2/sovereign_v29_stage2c_pure_annealing.yml \
    -t "$CKPT_C2_S1" \
    --use-amp --seed 42 \
    2>&1 | tee "$DIR_C2_S2/train.log"

if [ -f "$DIR_C2_S2/best_stg2.pth" ]; then
    CKPT_C2_S2="$DIR_C2_S2/best_stg2.pth"
elif [ -f "$DIR_C2_S2/best_stg1.pth" ]; then
    CKPT_C2_S2="$DIR_C2_S2/best_stg1.pth"
elif [ -f "$DIR_C2_S2/last.pth" ]; then
    CKPT_C2_S2="$DIR_C2_S2/last.pth"
else
    log "ERROR: Cycle 2 Step 2 did not produce a valid checkpoint!"
    exit 1
fi

log "==============================================================="
log "[CYCLE 2 COMPLETE] Sovereign V29 Cycle 2 Training Finished!"
log "Final Delivered Model: $CKPT_C2_S2"
log "==============================================================="
