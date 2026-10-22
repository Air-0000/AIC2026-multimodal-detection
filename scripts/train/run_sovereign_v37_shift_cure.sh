#!/usr/bin/env bash
# =============================================================================
# Sovereign V37: Sub-grid Shift Invariance & Continuous Equivariant Alignment
# 核心使命: 根治微小位移下的置信度跳变/断崖下跌 (406 帧 0.668 -> 541 帧 0.216)
# 全套根治架构:
#   1. 方案 A: 长宽比多尺度分层路由 (P3 30%, P4 30%, Global 40%)
#   2. 方案 B: 抗混叠低通 BlurPool 下采样 (消除 Nyquist-Shannon 跨步混叠)
#   3. 方案 C: 自适应可学习空间偏置 (零初始化 gamma 自适应学习局部距离约束)
#   4. 方案 D: 平移等变一致性损失 (SECL, 将分类置信度平滑锚定在几何 IoU 上)
#   5. 全生命周期微平移数据增强: RandomMicroShift(max_shift_px=16, p=0.5)
# =============================================================================

set -e
set -u

echo "============================================================================"
echo " [Sovereign V37] Pre-flight Environment Inspection & Initialization"
echo " Time: $(date)"
echo "============================================================================"

AIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_EXEC="$(which python)"
TORCHRUN_EXEC="$(which torchrun)"

echo "[Init] Project Root: ${AIC_ROOT}"
echo "[Init] Python Exec:   ${PYTHON_EXEC}"
echo "[Init] Torchrun Exec: ${TORCHRUN_EXEC}"

# GPU 状态检测
DETECTED_COUNT=$(${PYTHON_EXEC} -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
echo "[Init] Detected CUDA devices count: ${DETECTED_COUNT}"

CONFIG_PATH="configs/deimv2/sovereign_v37_shift_cure_2560.yml"
BASE_CKPT="outputs/sovereign_v34_clean_from_scratch_2560/best_stg2.pth"

cd "${AIC_ROOT}/vendor/DEIMv2"

export PYTHONPATH="${AIC_ROOT}/scripts/lib:${AIC_ROOT}/vendor/DEIMv2:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${AIC_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sovereign_v37_shift_cure_$(date +%Y%m%d_%H%M%S).log"

echo "[Init] Configuration: ${CONFIG_PATH}"
echo "[Init] Tuning Checkpoint: ${BASE_CKPT}"
echo "[Init] Logging to: ${LOG_FILE}"

if [ "${DETECTED_COUNT}" -gt 1 ]; then
    echo "[Run] Launching Distributed Training with torchrun on ${DETECTED_COUNT} GPUs..."
    ${TORCHRUN_EXEC} --nproc_per_node="${DETECTED_COUNT}" train.py \
        -c "${CONFIG_PATH}" \
        -t "${BASE_CKPT}" \
        --use-amp 2>&1 | tee "${LOG_FILE}"
else
    echo "[Run] Launching Single-GPU Training on device ${CUDA_VISIBLE_DEVICES:-0}..."
    ${PYTHON_EXEC} train.py \
        -c "${CONFIG_PATH}" \
        -t "${BASE_CKPT}" \
        --use-amp 2>&1 | tee "${LOG_FILE}"
fi

echo "============================================================================"
echo " [Sovereign V37] Training Finished at $(date)"
echo "============================================================================"
