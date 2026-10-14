#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 终极全功能多模态主干网络 V2 (Sovereign Multimodal V2)
# 模态: RGB(3) + Aligned IR(1) + LBG IR(1) + P0 16-bit 真实深度(1) -> 6 通道
# 三大新特性:
#   1. 动态模态置信度门控 (DynamicModalityConfidenceGating)
#   2. Stage 1 (Stride 4, 160x160) 浅层高频边缘注入 (HighFrequencyEdgeShallowFusion)
#   3. ImageNet 预训练 RGB Stem 权重迁移初始化 IR ModalityStem
# 固化成果: DCDA + DOG + 3D-Lift + Bilateral Pooling + 1D NUC 去条纹
# 彻底拔除: 错位 Query-level KD 干扰噪声
# 起点热启: outputs/deim_hgnetv2_x_aic_dog_mm/best_stg1.pth (56.18% 纯净成果)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v2_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v2_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_dog_mm/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V2 Training on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (56.18% clean baseline)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v2_train.log"

echo "Training finished: $(date)"
