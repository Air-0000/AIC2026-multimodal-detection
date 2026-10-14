#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 终极大融合主战攻坚模型 (Sovereign Multimodal Distillation)
# 模态: RGB(3) + Aligned IR(1) + LBG IR(1) + 16-bit Depth(1) -> 6 通道
# 架构: HGNetv2-B5 + DCDA + 3D-Lift + DOG + CSSA
# 教师: Pure RGB 56.71% (puremal_refine_giou35, 冻结在线蒸馏)
# 加速: /dev/shm 内存盘 (0延迟) + 16 Workers (208核) + 锁页内存 DMA
# =============================================================================
set -e
set -u

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}/vendor/DEIMv2"

ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/vendor/Depth-Anything-V2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 1. 自动确保 /dev/shm 内存盘全量高速缓存就绪 (耗时仅 ~8 秒)
if [ ! -d "/dev/shm/aic_cache/visible" ]; then
    echo "=========================================================="
    echo "Preloading dataset into /dev/shm RAM disk (Zero-latency bus)..."
    mkdir -p /dev/shm/aic_cache
    cp -r "${AIC_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/ &
    cp -r "${AIC_ROOT}/dataset/train/visible" /dev/shm/aic_cache/ &
    wait
    echo "Dataset preloaded to /dev/shm successfully!"
    df -h /dev/shm
fi

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/deim_hgnetv2_x_aic_sovereign_distill_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_x_aic_sovereign_distill_rtx6000"
TUN="./outputs/deim_hgnetv2_x_aic_dog_mm/best_stg1.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign Multimodal Distillation Training on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN}"
echo "Teacher Model: outputs/deim_hgnetv2_x_aic_puremal_refine_giou35/best_stg1.pth (56.71% mAP)"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_distill_train.log"

echo "Training finished: $(date)"
