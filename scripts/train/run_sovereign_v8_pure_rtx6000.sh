#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 小目标巅峰破局旗舰 V8 (Sovereign Multimodal V8 - Pure MSRB 16-Epoch Deep Annealing)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. MSRB 主干微残差直通桥: Stage 0 (160x160) 128 通道深层特征直通 P3 (80x80)
#   2. 保峰双路池化 (Peak Salience Dual Pooling): 杜绝小目标几何边缘被均值滤波冲淡
#   3. Mamba-CCFM 2D-SSM 全局全图关联 (P3/P4 跨模态无损长程上下文)
#   4. 纯净目标优化机制 (Pure Balanced Criterion): 彻底移除 cls_loss_weights 虚警干扰
#   5. 16 个 Epoch 深度淬火: flat_epoch: 8，给予 MSRB 充足的双倍探索与收敛周期
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v6_mamba_ssm_rtx6000/best_stg1.pth (57.14% 干净无偏基准)
# =============================================================================
set -e
set -u

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}/vendor/DEIMv2"

ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/vendor/Depth-Anything-V2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 确保 /dev/shm 内存盘全模态高速就绪
if [ ! -d "/dev/shm/aic_cache/visible" ]; then
    echo "Preloading dataset into /dev/shm RAM disk..."
    mkdir -p /dev/shm/aic_cache
    cp -r "${AIC_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/dataset/train/visible" /dev/shm/aic_cache/ &
    wait
    echo "Dataset preloaded to /dev/shm successfully!"
fi

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v8_msrb_pure_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v8_msrb_pure_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v6_mamba_ssm_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V8 (Pure MSRB 16-Epoch) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (57.14% all-time peak baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v8_msrb_pure_train.log"

echo "Training finished: $(date)"
