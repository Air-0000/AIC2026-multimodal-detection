#!/usr/bin/env bash
# =============================================================================
# Sovereign V22 极值高分辨率 (1216x672) 知识蒸馏主战攻坚启动脚本
# 算力平台: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, 600W 满血, 208 CPU Cores)
# 用户身份: 刘正阳 (liuzhengyang)
# =============================================================================
set -e
set -u

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}/vendor/DEIMv2"

ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/vendor/Depth-Anything-V2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 1. 验证 /dev/shm 内存盘缓存就绪 (2000张无缝内存常驻)
echo "=========================================================="
echo "Verifying /dev/shm RAM disk status..."
df -h /dev/shm
for mod in visible train_aligned_ir train_lbg_ir depth_p0; do
    echo "  - /dev/shm/aic_cache/${mod}: $(ls -1 /dev/shm/aic_cache/${mod} | wc -l) images"
done
echo "=========================================================="

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/sovereign_v22_highres1216_distill_rtx6000.yml"
OUT="./outputs/sovereign_v22_highres1216_distill_rtx6000"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

# 2. 检查微调检查点 / 教师网络权重
TEACHER_WEIGHT="${AIC_ROOT}/weight/sovereign_v21_quality_master_apex.pth"
echo "Teacher Model Weight: ${TEACHER_WEIGHT} ($(ls -lh ${TEACHER_WEIGHT} | awk '{print $5}'))"

echo "=========================================================="
echo "Starting Sovereign V22 High-Res 1216 Distillation on RTX PRO 6000 Blackwell"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Batch Size: 8 (Native 1216x672 16:9)"
echo "Epochs: 36 (~2.1 hours total on Blackwell 96GB)"
echo "=========================================================="

nvidia-smi

# 3. 启动主训练流程 (支持可选热启微调权重 $1)
TUNING_ARG=""
if [ -n "${1:-}" ] && [ "$1" != "scratch" ]; then
    echo "Using hot-start tuning checkpoint: $1"
    TUNING_ARG="-t $1"
else
    echo "Training from scratch with ImageNet HGNetv2 backbone & Sovereign teacher guidance!"
fi

/root/miniconda3/bin/python train.py -c "${CFG}" ${TUNING_ARG} --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_sovereign_v22_highres1216_train.log"

echo "Training finished at: $(date)"
