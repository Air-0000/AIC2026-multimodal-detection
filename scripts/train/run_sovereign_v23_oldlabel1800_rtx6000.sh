#!/usr/bin/env bash
# =============================================================================
# Sovereign V23 老标签 1800 全量重训 72-Epoch 全满血启动脚本
# 算力平台: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, 600W 满血)
# 用户身份: 刘正阳 (liuzhengyang)
# 战术目标: 老标签1800纯净标注 + 内生化翻转TTA + 1152x2048理论零畸变超高清
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
CFG="configs/deimv2/sovereign_v23_oldlabel1800_2048.yml"
OUT="./outputs/sovereign_v23_oldlabel1800_2048"
# 巅峰热启: 使用当前收敛最好的 V23 EMA 权重热启，实现最平滑的老标签域吸附
TUN="outputs/sovereign_v23_zero_distort_2048_fast32/best_stg2.pth"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

echo "============================================================================"
echo " Starting Sovereign V23 Old Label 1800 Full Run with Internal TTA on RTX PRO 6000"
echo " Date: $(date)"
echo " Config: ${CFG}"
echo " Pretrained Hot-Start: ${TUN}"
echo " Resolution: 1152 x 2048 (Strict 16:9 Zero Distortion)"
echo " Dataset: instances_train.json (1800 images, 13672 boxes, Pure Old Labels)"
echo " Validation: instances_val.json (200 images, 1523 boxes, Official Old Labels)"
echo " Internal TTA: ON (Horizontally Flipped Batch + Integrated WBF)"
echo " Epochs: 72"
echo "============================================================================"

nvidia-smi

python train.py -c "${CFG}" -t "${TUN}" --use-amp 2>&1 | tee "${LOG_DIR}/v23_oldlabel1800_rtx6000.log"

echo "Sovereign V23 Old Label 1800 training finished at: $(date)"
