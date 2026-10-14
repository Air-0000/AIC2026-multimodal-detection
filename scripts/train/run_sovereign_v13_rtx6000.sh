#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 顶会模块终极合流 V13 (Sovereign Multimodal V13 - Sovereign Ascendant)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏 (三阶实证模块全量落地):
#   1. [V1-231 LFP 低频引导净化模块 (CVPR 2026) in P2-CSRG+]:
#      - 语义锚点反向抑制平坦水面与高空云层高频噪声无序扰动
#      - 保护 3px 极细立柱边界不被平滑削弱
#   2. [V1创新038 PBD 峰值-背景能量正交解耦 (Peak-Background Decoupling) in P3]:
#      - 强制检测头聚焦离散显著峰值流，漫反射背景流被特征吸收
#      - 物理根除沿岸公园长椅与水面反光假阳性 (FP)
#   3. [V1-185 MixStyle 跨域泛化模块 (ICLR / 1区TOP) in HGNetv2MultiModal]:
#      - 0 额外参数，Batch 内特征统计量 (均值与方差) 动态插值
#      - 彻底消除昼夜/天气光照漂移 (Domain Shift)，全天候抗崩盘
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth (58.29% mAP 历史最高)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v13_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v13_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V13 (Sovereign Ascendant) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (Sovereign V12 baseline: 58.29% mAP)"
echo "Batch Size: 24 (LFP + PBD + MixStyle + Dual-Core Neck + DIAM Attention)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v13_train.log"

echo "Training finished: $(date)"
