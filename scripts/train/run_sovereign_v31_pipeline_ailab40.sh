#!/bin/bash
# =============================================================================
# Sovereign V31 Automated Pipeline: From V30 Peak to V31 Co-DETR One-to-Many
# 硬件平台: 实验室科研服务器 ailab40 (Tesla V100 32GB, GPU 1)
# =============================================================================

set -e

BASE_DIR="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
DEIM_DIR="$BASE_DIR/vendor/DEIMv2"
PY="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

cd "$DEIM_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=========================================================================="
log "=== Sovereign V31 Pipeline Initialized ==="
log "=========================================================================="

V30_DIR="outputs/sovereign_v30_cssp_v29peak_2560"

# 1. Wait for V30 to complete
log "Monitoring Sovereign V30 training process..."
while pgrep -f "sovereign_v30_cssp_v29peak_2560.yml" > /dev/null; do
    log "V30 training in progress... Sleeping for 60s."
    sleep 60
done

log ">>> Sovereign V30 Training Finished! <<<"

# 2. Verify V30 best checkpoint
V30_BEST="$V30_DIR/best_stg2.pth"
if [ ! -f "$V30_BEST" ]; then
    log "Warning: $V30_BEST not found, checking champions..."
    V30_BEST="outputs/champions/sovereign_v29_alltime_peak_c5_5756.pth"
fi

log "Base Checkpoint for V31 selected: $V30_BEST ($(du -h "$V30_BEST" | cut -f1))"

# 3. Create V31 output directory
V31_DIR="outputs/sovereign_v31_co_one2many_2560"
mkdir -p "$V31_DIR"

log "=========================================================================="
log "=== Launching Sovereign V31 Training (Co-DETR One-to-Many Supervision) ==="
log "=========================================================================="
log "Base Model:       $V30_BEST"
log "Config:           configs/deimv2/sovereign_v31_co_one2many_2560.yml"
log "Output Dir:       $V31_DIR"
log "Resolution:       1440 x 2560 (Native 2.5K)"
log "Supervision:      One-to-Many ATSS Aux Head (6~18 Positive Anchors per Object)"
log "Loss Objective:   TADL + FGL + CMAL + loss_co_cls (1.0) + loss_co_reg (2.0)"
log "=========================================================================="

$PY -u train.py \
  -c configs/deimv2/sovereign_v31_co_one2many_2560.yml \
  -t "$V30_BEST" \
  --use-amp \
  --seed 42 \
  --output-dir "$V31_DIR" \
  2>&1 | tee -a "$V31_DIR/train_console.log"

log "=== Sovereign V31 Training Finished at $(date) ==="
