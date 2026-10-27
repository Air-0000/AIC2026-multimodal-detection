#!/usr/bin/env bash
# =============================================================================
# Launch a Co-DINO pure-RGB run (MODEL=swin_l or vit_l, configs co_dino_<model>_aic_rgb_1800.py) after
# smoke_codino.sh passed. NGPUS (1, 2 or 4) GPUs on this machine, 1 image per GPU, accumulative_counts = 4 / NGPUS
# so every optimizer step sees 4 images. Pick the GPUs with CUDA_VISIBLE_DEVICES.
# Refuses an existing WORK_DIR unless RESUME=1 (mmengine then continues from WORK_DIR's last checkpoint).
# Checkpoints hold weights, EMA and AdamW state (Swin-L ~2.7 GB, ViT-L ~5 GB; 2 kept plus best), so on seeta point
# WORK_DIR at the large /autodl-fs disk. Log: logs/<config>_train.log.
# Usage: ROOT=/root/autodl-tmp/AIC_clean_v1 MODEL=vit_l NGPUS=4 WORK_DIR=/autodl-fs/data/codino/vit_l_rgb_1800 \
#        [RESUME=1] nohup bash scripts/train/codino/run_codino_aic.sh > /dev/null 2>&1 &
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/codino_env.sh"
MODEL=${MODEL:?set MODEL to swin_l or vit_l}
NGPUS=${NGPUS:-1}
NAME=$(codino_config "$MODEL")
ACCUM=$(codino_accum)
WORK_DIR=${WORK_DIR:-$ROOT/work_dir/codino/$NAME}
mkdir -p "$ROOT/logs"
LOG=$ROOT/logs/${NAME}_train.log

ARGS=(--work-dir "$WORK_DIR" --cfg-options optim_wrapper.accumulative_counts="$ACCUM")
if [ "${RESUME:-0}" = 1 ]; then
  ARGS+=(--resume)
elif [ -e "$WORK_DIR" ]; then
  echo "refusing: $WORK_DIR exists (set RESUME=1 to continue it)"; exit 1
fi

cd "$ROOT"
echo "$(date '+%F %T') start $NAME NGPUS=$NGPUS accumulative_counts=$ACCUM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} WORK_DIR=$WORK_DIR RESUME=${RESUME:-0} commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)" | tee -a "$LOG"
codino_train "$CODINO_DIR/$NAME.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
