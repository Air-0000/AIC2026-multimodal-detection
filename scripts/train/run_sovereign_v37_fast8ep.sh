#!/usr/bin/env bash
# =============================================================================
# Sovereign V37: 8-Epoch Fast Verification Pipeline (All 4 Mechanisms Integrated)
# 机制矩阵:
#   1. 机制一: Smooth Depth Fallback Mask (DOG 远景盲区柔性透传)
#   2. 机制二: 分层配额路由 (P3 30%, P4 30%, Global 40%)
#   3. 机制三: L0/L1 采样偏移 tanh 软裁剪 (限制在 +/- 2.0 物理网格)
#   4. 机制四: 渐进式辅助损失衰减 ([0.4, 0.5, 0.6, 0.8, 0.9, 1.0])
#
# 运行环境自适应:
#   - 远端 8 卡 (8 x RTX PRO 6000 / Tesla): 8卡并发并行 (bs=16~32), 约 8~10 分钟全流程结束!
#   - 本地单卡 (RTX 5060 Ti): 自动单卡降配运行, 约 25~30 分钟。
# =============================================================================

set -e
set -u

# --- 环境自适应探测 ---
if [ -d "/root/autodl-tmp/AIC" ]; then
    AIC_ROOT="/root/autodl-tmp/AIC"
    PYTHON_EXEC="/root/miniconda3/bin/python"
    [ ! -f "${PYTHON_EXEC}" ] && PYTHON_EXEC="$(which python)"
elif [ -d "/home/ailab/ssd/bigdata/hejianzhong/AIC_V29" ]; then
    AIC_ROOT="/home/ailab/ssd/bigdata/hejianzhong/AIC_V29"
    PYTHON_EXEC="/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python"
    [ ! -f "${PYTHON_EXEC}" ] && PYTHON_EXEC="$(which python)"
else
    AIC_ROOT="/home/constant/CodeBuddy/Dev/AIC"
    PYTHON_EXEC="/home/constant/miniconda3/envs/deimv2/bin/python"
    [ ! -f "${PYTHON_EXEC}" ] && PYTHON_EXEC="$(which python)"
fi

cd "${AIC_ROOT}/vendor/DEIMv2"
ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BUFFSIZE=16777216
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/sovereign_v37_fast8ep_verify.yml"
OUT="./outputs/sovereign_v37_fast8ep_verify"

# 寻找 V34 基底权重
START_CKPT=""
CANDIDATES=(
    "${AIC_ROOT}/vendor/DEIMv2/outputs/sovereign_v34_clean_from_scratch_2560/best_stg2.pth"
    "${AIC_ROOT}/outputs/checkpoints/sovereign_v34_clean_from_scratch_2560/best_stg2.pth"
    "${AIC_ROOT}/weights/sovereign_v34_best_stg2.pth"
    "./outputs/sovereign_v34_clean_from_scratch_2560/best_stg2.pth"
)

for c in "${CANDIDATES[@]}"; do
    if [ -f "$c" ]; then
        START_CKPT="$c"
        break
    fi
done

if [ -z "$START_CKPT" ]; then
    echo "ERROR: Could not find V34 base checkpoint in candidate locations!"
    exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

# 探测 GPU 数量
NUM_GPUS=$(${PYTHON_EXEC} -c "import torch; print(torch.cuda.device_count())")
echo "============================================================================"
echo " Starting Sovereign V37 Fast Verification (8 Epochs)"
echo " Timestamp:         $(date)"
echo " Root Directory:    ${AIC_ROOT}"
echo " Python Executable: ${PYTHON_EXEC}"
echo " Base Checkpoint:   ${START_CKPT}"
echo " Config:            ${CFG}"
echo " Output Directory:  ${OUT}"
echo " Detected GPUs:     ${NUM_GPUS}"
echo "============================================================================"

# 选择 torchrun 或 单卡 python
TORCHRUN_EXEC="$(dirname ${PYTHON_EXEC})/torchrun"
[ ! -f "${TORCHRUN_EXEC}" ] && TORCHRUN_EXEC="$(which torchrun 2>/dev/null || true)"

if [ "${NUM_GPUS}" -ge 8 ] && [ -n "${TORCHRUN_EXEC}" ]; then
    echo ">>> Running in 8-GPU Full Distributed Parallel Mode (Target time: ~8-10 min) <<<"
    ${TORCHRUN_EXEC} --nproc_per_node=8 train.py \
        -c "${CFG}" \
        -t "${START_CKPT}" \
        --use-amp \
        --seed 42 \
        2>&1 | tee -a "${OUT}/train_console.log" | tee "${LOG_DIR}/v37_fast8ep_8gpu.log"
elif [ "${NUM_GPUS}" -gt 1 ] && [ -n "${TORCHRUN_EXEC}" ]; then
    echo ">>> Running in Multi-GPU Mode (${NUM_GPUS} GPUs) <<<"
    ${TORCHRUN_EXEC} --nproc_per_node=${NUM_GPUS} train.py \
        -c "${CFG}" \
        -t "${START_CKPT}" \
        --use-amp \
        --seed 42 \
        2>&1 | tee -a "${OUT}/train_console.log" | tee "${LOG_DIR}/v37_fast8ep_${NUM_GPUS}gpu.log"
else
    echo ">>> Running in Single-GPU Mode (Target time: ~25-30 min) <<<"
    ${PYTHON_EXEC} -u train.py \
        -c "${CFG}" \
        -t "${START_CKPT}" \
        --use-amp \
        --seed 42 \
        2>&1 | tee -a "${OUT}/train_console.log" | tee "${LOG_DIR}/v37_fast8ep_1gpu.log"
fi

echo "============================================================================"
echo " Sovereign V37 Fast Verification Completed! ($(date))"
echo " Final Best Checkpoint: ${OUT}/best_stg2.pth"
echo "============================================================================"

# --- 自动触发 200 张验证集打分闭环 ---
if [ -f "${OUT}/best_stg2.pth" ]; then
    echo ">>> Running Immediate Validation Scorecard Evaluation... <<<"
    ${PYTHON_EXEC} "${AIC_ROOT}/scripts/eval/eval_v37_fast8ep_val.py" \
        --ckpt "${OUT}/best_stg2.pth" \
        --cfg "${CFG}"
fi
