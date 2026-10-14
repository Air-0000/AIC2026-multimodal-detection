#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 纯净顶会生产架构 8 轮训练 (Sovereign Pure Production RTX6000)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心拓扑:
#   1. [骨干纯净]: 纯 HGNetv2-X + DCDA/DOG/3DLift + 0参数 MixStyle + MSRB 直通桥 (剔除所有冗余自注意力门控)
#   2. [特征颈裁撤]: 裁撤 MHLA 双注意力冗余，保留 Mamba 2D-SSM + 微软 StarConv (CVPR 2024)
#   3. [高频去噪引导]: P2-CSRG+ (Haar小波纵向核 + Spatial-PBD 空间水空门控 + Point-Preserving LFP)
#   4. [解码头]: DIAM 局部几何交互注意力 + SNR-Aware FDR 边界分布锐化
#   5. [推断原生]: 3-View CAA-SAHI 完整内生化于 forward()
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v14_soup/best_stg1.pth (58.30% 历史绝对最高点)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_pure_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_pure_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v14_soup/best_stg1.pth"

mkdir -p "${OUT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rtx6000_sovereign_pure_train.log"

echo "============================================================================="
echo "Starting Sovereign Pure Production 8-Epoch Training on RTX PRO 6000 Blackwell"
echo "Config: ${CFG}"
echo "Tuning Checkpoint: ${TUN} (Current Peak: 58.30% mAP)"
echo "Output: ${OUT}"
echo "Log: ${LOG_FILE}"
echo "============================================================================="

python3 train.py \
    -c "${CFG}" \
    -t "${TUN}" \
    --use-amp \
    --seed 42 \
    2>&1 | tee "${LOG_FILE}"

echo "Sovereign Pure Training Complete!"
