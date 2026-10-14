#!/usr/bin/env bash
# =============================================================================
# Sovereign V21-Quality-Master 进修启动脚本 (RTX PRO 6000 Blackwell 96GB)
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

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/sovereign_v21_quality_finetune_rtx6000.yml"
OUT="./outputs/sovereign_v21_quality_finetune_rtx6000"
TUN="${AIC_ROOT}/weight/sovereign_v21_apex_soup_581.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign V21 Quality Fine-tuning on RTX PRO 6000 Blackwell"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN}"
echo "Memory Cache: /dev/shm/aic_cache (Active)"
echo "Batch Size: 16 (112 steps per epoch, ~45s per epoch)"
echo "=========================================================="

nvidia-smi

/root/miniconda3/bin/python train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/v21_quality_finetune_rtx6000.log"

echo "Training finished: $(date)"
