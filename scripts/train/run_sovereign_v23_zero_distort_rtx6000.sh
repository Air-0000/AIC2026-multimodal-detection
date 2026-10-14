#!/usr/bin/env bash
# =============================================================================
# Sovereign V23 理论零畸变超高清 (1152x2048) 54-Epoch 全满血攻坚启动脚本
# 算力平台: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, 600W 满血)
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
CFG="configs/deimv2/sovereign_v23_zero_distort_2048_rtx6000.yml"
OUT="./outputs/sovereign_v23_zero_distort_2048_rtx6000"
TUN="${AIC_ROOT}/weight/sovereign_v22_soup_c1_c2.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "============================================================================"
echo " Starting Sovereign V23 1152x2048 Zero-Distortion Full Run on RTX PRO 6000"
echo " Date: $(date)"
echo " Config: ${CFG}"
echo " Pretrained Hot-Start: ${TUN}"
echo " Resolution: 1152 x 2048 (Strict 16:9 Zero Distortion, 2^11 CUDA aligned)"
echo " Batch Size: 6 x 4 accum = 24 effective batch"
echo " Dataset: quality1998_train.json (1998 images, 17921 boxes, +18% completed)"
echo " Epochs: 54"
echo "============================================================================"

nvidia-smi

/root/miniconda3/bin/python train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/v23_zero_distort_2048_rtx6000.log"

echo "Sovereign V23 54-epoch training finished at: $(date)"
