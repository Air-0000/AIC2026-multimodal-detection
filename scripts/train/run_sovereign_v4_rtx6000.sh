#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 终极双频域残差傅里叶引导主干网络 V4 (Sovereign Multimodal V4 - RFGM)
# 基于 AAAI 2026 顶级论文: Beyond Illumination: Fine-Grained Detail Preservation
# 模态: RGB(3) + Aligned IR(1) + LBG IR(1) + P0 16-bit 真实深度(1) -> 6 通道
# 两大新架构:
#   1. [Scheme A] 前端频域暗光照度恢复模块 (Frontend RFGM):
#      - 2D-rFFT 幅度/相位分离 + 通道 Gram 相似度矩阵 (MagGuidedFusion)
#   2. [Scheme B] RF-ProG 浅层相位几何直通 Prompt (Stage 0 -> Stage 1 / Neck P3):
#      - 频域全图连续闭合相位补偿几何外框，彻底攻坚 sign (常温微小) 与 boat (水面暗光)
# 起点热启: outputs/deim_hgnetv2_x_aic_sovereign_v3_rtx6000/best_stg1.pth (56.38% 历史最高战绩)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v4_rfgm_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v4_rfgm_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v3_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V4 (RFGM) Training on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (56.38% baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v4_rfgm_train.log"

echo "Training finished: $(date)"
