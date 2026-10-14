#!/usr/bin/env bash
# =============================================================================
# Sovereign V22 极值高分辨率 (1216x672) 2-Epoch 先锋探路启动脚本
# 算力平台: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
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
CFG="configs/deimv2/sovereign_v22_highres1216_probe_rtx6000.yml"
OUT="./outputs/sovereign_v22_highres1216_probe_rtx6000"
TUN="${AIC_ROOT}/weight/sovereign_v21_quality_master_apex.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "=========================================================="
echo "Starting Sovereign V22 1216x672 2-Epoch Probe on RTX PRO 6000"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Pretrained Hot-Start: ${TUN}"
echo "Batch Size: 24 (1216x672 16:9, ~90GB VRAM)"
echo "Epochs: 2 (Estimated time ~5 minutes total)"
echo "=========================================================="

nvidia-smi

/root/miniconda3/bin/python train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/v22_1216_probe_rtx6000.log"

echo "Probe finished at: $(date)"
