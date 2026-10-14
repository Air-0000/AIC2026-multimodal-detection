#!/usr/bin/env bash
# =============================================================================
# Sovereign V22 极值高分辨率 (1216x672) 72-Epoch 满血挂载脚本
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
CFG="configs/deimv2/sovereign_v22_highres1216_rtx6000_full72e.yml"
OUT="./outputs/sovereign_v22_highres1216_rtx6000_full72e"
TUN="${AIC_ROOT}/weight/sovereign_v22_highres1216_probe_apex_513.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign V22 1216x672 72-Epoch Full Run on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN}"
echo "Batch Size: 24 (1216x672 16:9, ~94.8GB VRAM, 75 steps/epoch)"
echo "Augmentations: Multimodal Mosaic + MixUp + CopyBlend + Photometric + Zoom/Crop"
echo "Epochs: 72 (Estimated total time ~2.7 hours)"
echo "=========================================================="

nvidia-smi

/root/miniconda3/bin/python train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/v22_1216_full72e_rtx6000.log"

echo "Full 72-epoch training finished at: $(date)"
