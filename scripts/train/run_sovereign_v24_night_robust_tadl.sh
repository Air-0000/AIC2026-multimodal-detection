#!/bin/bash
set -e
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/autodl-tmp/AIC/scripts/lib:/root/autodl-tmp/AIC/vendor/DEIMv2

cd /root/autodl-tmp/AIC/vendor/DEIMv2
mkdir -p outputs/sovereign_v24_night_robust_tadl_2048

START_CKPT="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v23_tadl_oldlabel1800_2048/best_stg2.pth"

echo "================================================================="
echo "=== Launching Sovereign V24 Night Robust TADL Training        ==="
echo "================================================================="
echo "Base Checkpoint: $START_CKPT"
echo "Config: configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml"
echo "Timestamp: $(date)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Launch background disk monitor to ensure we never run out of disk
(
  while true; do
    sleep 60
    AVAIL=$(df /root/autodl-tmp | awk 'NR==2 {print $4}')
    if [ "$AVAIL" -lt 2500000 ]; then
      echo "[Disk Monitor $(date)] Free space < 2.5GB ($AVAIL KB), pruning old intermediate checkpoints..."
      find outputs/sovereign_v24_night_robust_tadl_2048/ -name "checkpoint000*.pth" -mmin +20 -delete 2>/dev/null || true
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

/root/miniconda3/bin/python train.py \
  -c configs/deimv2/sovereign_v24_night_robust_tadl_2048.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a outputs/sovereign_v24_night_robust_tadl_2048/train_console.log

echo "=== Sovereign V24 Night Robust TADL Training Completed at $(date) ==="
