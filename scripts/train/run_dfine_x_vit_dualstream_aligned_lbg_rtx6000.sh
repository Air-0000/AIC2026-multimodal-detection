#!/usr/bin/env bash
# =============================================================================
# DEIM-D-FINE-X 异构双流 (HGNetv2-B5 CNN + DAv2 ViT) -- 物理对齐+LBG强化版
# NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM)
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
CFG="configs/deimv2/deim_hgnetv2_vit_dualstream_aligned_lbg_rtx6000.yml"
OUT="./outputs/deim_hgnetv2_vit_dualstream_aligned_lbg_rtx6000"
TUN="ckpts/dfine_x_obj2coco.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting DEIM-D-FINE-X Dual-Stream (Aligned IR + LBG) on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained weights: ${TUN}"
echo "=========================================================="

nvidia-smi

python3 train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/rtx6000_dualstream_aligned_lbg_train.log"

echo "Training finished: $(date)"
