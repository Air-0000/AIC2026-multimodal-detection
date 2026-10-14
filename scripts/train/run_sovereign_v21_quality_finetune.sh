#!/usr/bin/env bash
# =============================================================================
# Sovereign V21-Quality-Master 进修微调启动脚本
# 运行环境: 实验室科研服务器 ailab40 (Tesla V100 32GB @ GPU 1)
# 用户身份: 刘正阳 (liuzhengyang)
# -----------------------------------------------------------------------------
# 战术目标:
#   - 起步权重: sovereign_v21_apex_soup_581.pth (58.12% 现役巅峰)
#   - 极小学习率: lr = 1e-5 (平滑微调)
#   - 训练真值: instances_train_clean_quality.json (1,800 张, 16,170 框, +2,498 框)
#   - 验证真值: instances_val_clean_quality.json (200 张, 1,757 框, +234 框)
#   - 策略: 训练 6 个 Epoch，每轮评测记录，保存最高权重为 V21-Quality-Master
# =============================================================================
set -e
set -u

BASE_DIR="/home/ailab/ssd/bigdata/liuzhengyang"
DEIM_DIR="${BASE_DIR}/deimv2"
LOG_DIR="${BASE_DIR}/logs"
WEIGHT_DIR="${BASE_DIR}/weights"
OUT_DIR="${BASE_DIR}/outputs/sovereign_v21_quality_finetune_ailab40"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"

cd "${DEIM_DIR}"

ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export TORCH_CUDA_MATMUL_NO_CUBLASLT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

CFG="configs/deimv2/sovereign_v21_quality_finetune_ailab40.yml"
TUN="${WEIGHT_DIR}/sovereign_v21_apex_soup_581.pth"
LOG_FILE="${LOG_DIR}/v21_quality_finetune_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "Starting Sovereign V21 Quality Fine-tuning Training"
echo "Host: ailab40 (Tesla V100 32GB @ GPU 1)"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Hot-Start Checkpoint: ${TUN}"
echo "Logging to: ${LOG_FILE}"
echo "=========================================================="

nvidia-smi

/home/ailab40/anaconda3/envs/rfdetr/bin/python train.py \
    -c "${CFG}" \
    -t "${TUN}" \
    --use-amp 2>&1 | tee "${LOG_FILE}"

echo "=========================================================="
echo "Sovereign V21 Quality Fine-tuning Complete: $(date)"
echo "=========================================================="
