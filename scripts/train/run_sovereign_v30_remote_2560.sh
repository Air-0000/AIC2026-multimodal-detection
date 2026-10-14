#!/bin/bash
set -e

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/autodl-tmp/AIC/scripts/lib:/root/autodl-tmp/AIC/vendor/DEIMv2:${PYTHONPATH:-}

cd /root/autodl-tmp/AIC/vendor/DEIMv2
mkdir -p outputs/sovereign_v30_cssp_2560_remote

# 1. 确保高速 shm 内存缓存链路正常
mkdir -p /dev/shm/aic_cache
ln -sfn /root/autodl-tmp/AIC/dataset/train/visible /dev/shm/aic_cache/visible
ln -sfn /root/autodl-tmp/AIC/work_dir/train_aligned_ir /dev/shm/aic_cache/train_aligned_ir
ln -sfn /root/autodl-tmp/AIC/work_dir/train_lbg_ir /dev/shm/aic_cache/train_lbg_ir
ln -sfn /root/autodl-tmp/AIC/dataset/train/depth_p0 /dev/shm/aic_cache/depth_p0

# 2. 热启动基底检查 (优先使用 V28-A 或 V29 巅峰)
START_CKPT="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v28_disparity_awakening_2048/best_stg1.pth"
if [ ! -f "$START_CKPT" ]; then
    START_CKPT="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v29_cycle5_stage2_anneal_2560/best_stg2.pth"
fi
if [ ! -f "$START_CKPT" ]; then
    START_CKPT="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v27_pure_unbiased_2048/best_stg2.pth"
fi

echo "=========================================================================="
echo "=== Launching Sovereign V30 CSSP + ABC Remote 2.5K Fine-Tuning (8 Ep) ==="
echo "=========================================================================="
echo "Base Checkpoint:   $START_CKPT"
echo "Config:            configs/deimv2/sovereign_v30_cssp_2560_remote.yml"
echo "Core Innovation:   CSSP Cross-Spectral Diff |F_rgb - F_ir| + ABC Local Contrast Attention"
echo "Resolution:        1440 x 2560 (2.5K Native Ultra-High Resolution)"
echo "Epochs:            8 (Fast 8-Epoch Tuning)"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "=========================================================================="

/root/miniconda3/bin/python train.py \
  -c configs/deimv2/sovereign_v30_cssp_2560_remote.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a outputs/sovereign_v30_cssp_2560_remote/train_console.log

echo "=== Sovereign V30 2.5K Remote Tuning Completed at $(date) ==="
