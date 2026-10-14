#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 终极纯净边界精炼攻坚模型 V5 (Sovereign Multimodal V5 - Clean GIoU 3.5x + MAL2)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. loss_giou: 3.5 + cost_giou: 3.5 (全面收紧边框，直指 63%+ AP75)
#   2. loss_mal: 2.0 (Model-Aware 负样本动态强惩罚)
#   3. 纯净单图无畸变精炼 (彻底移除 Mosaic/Crop 物理碎块扰动)
#   4. 超平滑微调: 8e-6 (主干 2e-6), 8 Epochs
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v4_rfgm_rtx6000/best_stg2.pth (56.67% 峰值战果)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v5_refine_giou35_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v5_refine_giou35_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v4_rfgm_rtx6000/best_stg2.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V5 (Clean GIoU 3.5x + MAL2) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (56.67% peak baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v5_giou35_train.log"

echo "Training finished: $(date)"
