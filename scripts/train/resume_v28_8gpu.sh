#!/usr/bin/env bash
set -e
set -u

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}/vendor/DEIMv2"
ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/vendor/Depth-Anything-V2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"

mkdir -p /dev/shm/aic_cache
ln -sfn "${AIC_ROOT}/dataset/train/visible" /dev/shm/aic_cache/visible
ln -sfn "${AIC_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/train_aligned_ir
ln -sfn "${AIC_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/train_lbg_ir
ln -sfn "${AIC_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/depth_p0
echo "[$(date)] /dev/shm/aic_cache ready!"

CFG="configs/deimv2/sovereign_v28_full72ep_zoom_awakening_8gpu.yml"
OUT="./outputs/checkpoints/sovereign_v28_full72ep_zoom_awakening_8gpu"
RES="${OUT}/last.pth"
LOG_DIR="${AIC_ROOT}/logs"

echo "============================================================================"
echo " Resuming Sovereign V28 from ${RES} (Epoch 58 -> 72) on 8 x RTX PRO 6000"
echo " Date: $(date)"
echo "============================================================================"

# 守护磁盘空间，防盘满
(
  while true; do
    sleep 60
    AVAIL=$(df "${AIC_ROOT}" | awk "NR==2 {print \$4}")
    if [ "$AVAIL" -lt 3000000 ]; then
      echo "[Disk Monitor $(date)] Free space < 3GB, pruning old intermediate checkpoints..."
      find "${OUT}" -name "checkpoint00*.pth" -mmin +20 -delete 2>/dev/null || true
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

/root/miniconda3/bin/torchrun --nproc_per_node=8 train.py \
  -c "${CFG}" \
  -r "${RES}" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a "${OUT}/train_console.log" | tee -a "${LOG_DIR}/v28_full72ep_zoom_awakening_8gpu.log"

echo "============================================================================"
echo "Sovereign V28 full 72 epochs completed successfully at: $(date)"
echo "============================================================================"
