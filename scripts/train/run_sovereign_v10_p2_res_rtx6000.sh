#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 尺度破局终极旗舰 V10 (Sovereign Multimodal V10 - P2 Cross-Scale Residual Guidance)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. [P2 (160x160) 浅层高频与立柱相位直通指南 (P2-CSRG in HybridEncoder)]:
#      - Backbone 释放 Stage 0 (stride 4, 160x160, 128ch) 原生尺度
#      - HybridEncoder 引入 P2CrossScaleRefinement: 双路保峰池化 (Avg + Max) 捕捉单像素标牌立柱
#      - 将 160x160 亚像素高频轮廓无损注入 P3 (80x80) Top-Down 颈部特征，杜绝小目标几何特征塌陷！
#      - 严格零初始化 (Zero-Init): Step 0 与 Sovereign V9 达成 100.000% 绝对数值等价
#   2. MSRB 主干微残差直通桥: Stage 0 (160x160) 128 通道深层特征直通 P3 (80x80)
#   3. AAAI 2026 RFGM 傅里叶相位-幅度几何直通
#   4. 暗区信噪比感知边界分布锐化头 (SNR-Aware FDR Head)
#   5. Mamba-CCFM 2D-SSM 线性全局长程交互 (P3/P4 跨模态无损长程上下文)
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000/best_stg1.pth
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v10_p2_res_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v9_snr_fdr_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V10 (P2 Cross-Scale Residual Guidance) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (Sovereign V9 baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v10_p2_res_train.log"

echo "Training finished: $(date)"
