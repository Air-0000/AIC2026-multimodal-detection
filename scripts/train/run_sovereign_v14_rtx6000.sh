#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 顶会模块终极合流 V14 (Sovereign Multimodal V14 - Sovereign Imperium)
# 模态: 6 通道 (RGB + Aligned IR + LBG IR + P0 16-bit Metric Depth)
# 核心杀手锏:
#   1. [空间分区峰值-背景解耦 (Spatial-PBD)]: 陆地豁免长椅 (seat)，水面消灭反光 (boat 49%+)
#   2. [点源保护低频净化 (Point-Preserving LFP)]: 保护路灯 (light) 与小球 (ball)
#   3. [【V1-011】StarConv 微软星形运算 (CVPR 2024)]: P4 隐式无限维非线性交互
#   4. [V1-185 MixStyle 跨域泛化]: 全天候防光照漂移
# 起步热启: outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth (58.29% mAP 基线)
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
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_v14_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_v14_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal V14 (Sovereign Imperium) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN} (Sovereign V12 baseline: 58.29% mAP)"
echo "Batch Size: 24 (Spatial-PBD + Point-LFP + StarConv + MixStyle)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v14_train.log"

echo "Training finished: $(date)"
