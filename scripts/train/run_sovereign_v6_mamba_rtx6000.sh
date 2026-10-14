#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 终极异构混血架构 V6 (Sovereign Multimodal V6 - Mamba-CCFM 2D-SSM)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. Mamba-CCFM 混血特征颈: P3 (80x80) 与 P4 (40x40) 2D-SSM 4向交叉线性全景扫描
#   2. 零初始化（Zero-Init）机制: 保证从 Sovereign V5 (56.77% mAP) 100.00% 绝对无损热启
#   3. 攻克小目标 AP_small = 29.04% 物理断层落差
#   4. loss_giou: 3.5, cost_giou: 3.5, loss_mal: 2.0 (全景难负样本动态感知惩罚)
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v5_refine_giou35_rtx6000/best_stg1.pth (56.77% 历史最高战果)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v6_mamba_ssm_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v6_mamba_ssm_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v5_refine_giou35_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V6 (Mamba-CCFM 2D-SSM) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (56.77% all-time peak baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v6_mamba_ssm_train.log"

echo "Training finished: $(date)"
