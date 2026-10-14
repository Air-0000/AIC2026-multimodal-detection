#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 暗区边界锐化破局旗舰 V9 (Sovereign Multimodal V9 - SNR-Aware FDR Head)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. [方案 4: 暗区信噪比感知边界分布锐化头 (SNR-Aware FDR Head)]:
#      - Query 驱动自适应降温 (tau < 1.0)，强制收敛弥散边界概率分布，终结暗区漂移！
#   2. MSRB 主干微残差直通桥: Stage 0 (160x160) 128 通道深层特征直通 P3 (80x80)
#   3. 保峰双路池化 (Peak Salience Dual Pooling): 杜绝小目标几何边缘被均值滤波冲淡
#   4. Mamba-CCFM 2D-SSM 全局长程关联 (P3/P4 跨模态无损长程上下文)
#   5. 严格零初始化 (Zero-Init): 100.00% 绝对无损热启自 Sovereign V6 (57.14% 历史最高战果)
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v6_mamba_ssm_rtx6000/best_stg1.pth
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v6_mamba_ssm_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V9 (SNR-Aware FDR Head) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (57.14% all-time peak baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v9_snr_fdr_train.log"

echo "Training finished: $(date)"
