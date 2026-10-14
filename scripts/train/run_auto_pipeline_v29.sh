#!/usr/bin/env bash
set -eo pipefail

AIC_ROOT="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
cd "$AIC_ROOT/vendor/DEIMv2"
export PYTHONPATH="$AIC_ROOT/scripts/lib:$AIC_ROOT/vendor/DEIMv2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=1
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

LOG_FILE="$AIC_ROOT/auto_pipeline.log"
log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

log "==============================================================="
log "Sovereign V29 Auto-Chained Pipeline: Stage 2-B -> Stage 2-C"
log "Hardware: ailab40 (Tesla V100 32GB, GPU 1 [Healthy SM_70])"
log "==============================================================="

# ==============================================================
# Base Checkpoint Verification from Stage 2-A
# ==============================================================
DIR_2A="outputs/sovereign_v29_stage2a_disparity_alignment"
if [ -f "$DIR_2A/best_stg2.pth" ]; then
    CKPT_2A="$DIR_2A/best_stg2.pth"
elif [ -f "$DIR_2A/best_stg1.pth" ]; then
    CKPT_2A="$DIR_2A/best_stg1.pth"
elif [ -f "$DIR_2A/last.pth" ]; then
    CKPT_2A="$DIR_2A/last.pth"
else
    echo "ERROR: No Stage 2-A checkpoint found!"
    exit 1
fi
log "[Stage 2-A Verified] Base Checkpoint: $CKPT_2A"

# ==============================================================
# Stage 2-B: 3D 物理空间升维与深层语义全融合 (12 轮)
# ==============================================================
DIR_2B="outputs/sovereign_v29_stage2b_deep_fusion"
mkdir -p "$DIR_2B"
if [ -f "$DIR_2B/train.log" ]; then
    mv "$DIR_2B/train.log" "$DIR_2B/train_prev_$(date +%Y%m%d_%H%M%S).log"
fi

log "==============================================================="
log "[Stage 2-B] Starting 12-epoch 3D lift & deep multimodal fusion..."
log "[Stage 2-B] Chaining from: $CKPT_2A"

$PY -u train.py \
    -c configs/deimv2/sovereign_v29_stage2b_deep_fusion.yml \
    -t "$CKPT_2A" \
    --use-amp --seed 42 \
    2>&1 | tee "$DIR_2B/train.log"

if [ -f "$DIR_2B/best_stg2.pth" ]; then
    CKPT_2B="$DIR_2B/best_stg2.pth"
elif [ -f "$DIR_2B/best_stg1.pth" ]; then
    CKPT_2B="$DIR_2B/best_stg1.pth"
elif [ -f "$DIR_2B/last.pth" ]; then
    CKPT_2B="$DIR_2B/last.pth"
else
    log "ERROR: Stage 2-B did not produce a valid checkpoint!"
    exit 1
fi
log "[Stage 2-B Complete] Best Checkpoint: $CKPT_2B"

# ==============================================================
# Stage 2-C: 纯净无偏退火与高精度边界框收官 (6 轮)
# ==============================================================
DIR_2C="outputs/sovereign_v29_stage2c_pure_annealing"
mkdir -p "$DIR_2C"
if [ -f "$DIR_2C/train.log" ]; then
    mv "$DIR_2C/train.log" "$DIR_2C/train_prev_$(date +%Y%m%d_%H%M%S).log"
fi

log "==============================================================="
log "[Stage 2-C] Starting 6-epoch pure unbiased annealing..."
log "[Stage 2-C] Chaining from: $CKPT_2B"

$PY -u train.py \
    -c configs/deimv2/sovereign_v29_stage2c_pure_annealing.yml \
    -t "$CKPT_2B" \
    --use-amp --seed 42 \
    2>&1 | tee "$DIR_2C/train.log"

if [ -f "$DIR_2C/best_stg2.pth" ]; then
    CKPT_2C="$DIR_2C/best_stg2.pth"
elif [ -f "$DIR_2C/best_stg1.pth" ]; then
    CKPT_2C="$DIR_2C/best_stg1.pth"
elif [ -f "$DIR_2C/last.pth" ]; then
    CKPT_2C="$DIR_2C/last.pth"
else
    log "ERROR: Stage 2-C did not produce a valid checkpoint!"
    exit 1
fi

log "==============================================================="
log "[ALL COMPLETE] Sovereign V29 Full Training Pipeline Finished!"
log "Final Delivered Model: $CKPT_2C"
log "==============================================================="
