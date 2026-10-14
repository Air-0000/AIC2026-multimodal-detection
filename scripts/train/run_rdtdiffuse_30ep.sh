#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Quick validation of the hybrid backbone: 30 epochs, then measure how much the
# IR channel actually contributes (same protocol as the 5ch/4ch ablations).
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" >> /tmp/rdd30_progress.log; }

cd "$DEIM" || exit 1

log "start 30ep training"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_30ep.yml --use-amp \
    > /tmp/rdd30_train.log 2>&1
log "training exit=$?"

OUT=$DEIM/outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep
if [ -f "$OUT/best_stg2.pth" ]; then CKPT="$OUT/best_stg2.pth"
elif [ -f "$OUT/best_stg1.pth" ]; then CKPT="$OUT/best_stg1.pth"
elif [ -f "$OUT/last.pth" ]; then CKPT="$OUT/last.pth"
else log "NO CHECKPOINT -> abort"; exit 2; fi
log "ckpt=$CKPT"

log "baseline AIC val (full modalities)"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_30ep.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/rdd30_val_base.log 2>&1

log "IR ablation (IR -> all black)"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_30ep_val_noir.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/rdd30_val_noir.log 2>&1

log "RDTDIFUSE-30EP COMPLETE"
