#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Hybrid backbone (RDTTrack de-redundancy + DiffFuse feature-level injection):
# wait for the AIC-2 segformer job to free the GPU, then
#   train -> IR ablation on AIC val -> LLVIP eval.
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" >> /tmp/rdtdiffuse_progress.log; }

log "waiting for AIC-2 (segformer) to release the GPU"
while pgrep -f "envs/aic-2" >/dev/null 2>&1; do sleep 60; done
log "GPU free -> start training"

cd "$DEIM" || exit 1
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse.yml --use-amp \
    > /tmp/rdtdiffuse_train.log 2>&1
log "training exit=$?"

OUT=$DEIM/outputs/deimv2_dinov3_l_aic_rdtdiffuse
if [ -f "$OUT/best_stg2.pth" ]; then CKPT="$OUT/best_stg2.pth"
elif [ -f "$OUT/best_stg1.pth" ]; then CKPT="$OUT/best_stg1.pth"
elif [ -f "$OUT/last.pth" ]; then CKPT="$OUT/last.pth"
else log "NO CHECKPOINT -> abort"; exit 2; fi
log "using ckpt $CKPT"

log "IR ablation on AIC val (FEATURE -> 0, valid protocol)"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_val_featnoir.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/rdtdiffuse_val_noir.log 2>&1
log "ir-ablation exit=$?"

log "depth ablation on AIC val (FEATURE -> 0)"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_val_featnodepth.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/rdtdiffuse_val_nodepth.log 2>&1
log "depth-ablation exit=$?"

log "both-aux ablation on AIC val (FEATURE -> 0)"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_val_featnoaux.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/rdtdiffuse_val_noaux.log 2>&1
log "noaux-ablation exit=$?"

log "LLVIP eval"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdtdiffuse_llvip.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/rdtdiffuse_llvip.log 2>&1
log "llvip exit=$?"

log "RDTDIFUSE QUEUE COMPLETE"
