#!/bin/bash
set -e
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/autodl-tmp/AIC/scripts/lib:/root/autodl-tmp/AIC/vendor/DEIMv2

cd /root/autodl-tmp/AIC/vendor/DEIMv2
mkdir -p outputs/sovereign_v27_pure_unbiased_2048

# 确保 shm 内存高速缓存链路正常
mkdir -p /dev/shm/aic_cache
ln -sfn /root/autodl-tmp/AIC/dataset/train/visible /dev/shm/aic_cache/visible
ln -sfn /root/autodl-tmp/AIC/work_dir/train_aligned_ir /dev/shm/aic_cache/train_aligned_ir
ln -sfn /root/autodl-tmp/AIC/work_dir/train_lbg_ir /dev/shm/aic_cache/train_lbg_ir
ln -sfn /root/autodl-tmp/AIC/dataset/train/depth_p0 /dev/shm/aic_cache/depth_p0

# 黄金热启动基底: Sovereign V23 TADL 成熟权重 (80+ 轮全模态物理收敛产物)
START_CKPT="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs/sovereign_v23_tadl_oldlabel1800_2048/best_stg2.pth"

echo "================================================================="
echo "=== Launching Sovereign V27 Pure Unbiased End-to-End Training ==="
echo "================================================================="
echo "Base Checkpoint: $START_CKPT"
echo "Config: configs/deimv2/sovereign_v27_pure_unbiased_2048.yml"
echo "Core Principle: All cls_neg_scale = 1.0 (Zero Artificial Bias)"
echo "Timestamp: $(date)"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 守护磁盘空间，防盘满
(
  while true; do
    sleep 60
    AVAIL=$(df /root/autodl-tmp | awk "NR==2 {print \$4}")
    if [ "$AVAIL" -lt 2500000 ]; then
      echo "[Disk Monitor $(date)] Free space < 2.5GB ($AVAIL KB), pruning old intermediate checkpoints..."
      find outputs/sovereign_v27_pure_unbiased_2048/ -name "checkpoint000*.pth" -mmin +20 -delete 2>/dev/null || true
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

/root/miniconda3/bin/python train.py \
  -c configs/deimv2/sovereign_v27_pure_unbiased_2048.yml \
  -t "$START_CKPT" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a outputs/sovereign_v27_pure_unbiased_2048/train_console.log

echo "=== Sovereign V27 Pure Unbiased Training Completed at $(date) ==="
