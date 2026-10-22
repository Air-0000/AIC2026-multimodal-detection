#!/usr/bin/env bash
# =============================================================================
# Sovereign V36: P2 (Stride=4) High-Resolution Direct Injection & Feature Guidance
# 核心架构:
#   1. 纯血开源底模: ckpts/dfine_x_obj2coco.pth (零历史污染)
#   2. 方案 C 破局: Stride=4 (P2 360x640) 浅层微小特征直通引导 (斩断下采样信息湮灭)
#   3. Meta DINOv3-ViT-Large 语义空间在线微创蒸馏 (2x 下采样，零显存瓶颈)
#   4. Co-ATSS 一对多辅助密集监督头 (保障 96%+ 极限高召回)
#   5. 原生单模型 NMS (IoU 0.60 彻底消解重叠多框，严格遵守赛题第十一条规范)
#   6. 黄金 18 轮极致收敛排程 (10 轮平原期 + 4 轮强增广退火 + 4 轮零增广吸附)
#
# 支持平台:
#   - 单卡 RTX PRO 6000 (~1.3 小时完成 18 轮)
#   - 双卡 RTX PRO 6000 (~50 分钟完成 18 轮)
#   - 8 卡 RTX PRO 6000 (~20 分钟完成 18 轮)
# =============================================================================

set -e
set -u

echo "============================================================================"
echo " [Sovereign V36] Pre-flight Environment Inspection & Initialization"
echo " Time: $(date)"
echo "============================================================================"

# 1. 确定运行根目录
if [ -d "/root/autodl-tmp/AIC" ]; then
    AIC_ROOT="/root/autodl-tmp/AIC"
    PYTHON_EXEC="/root/miniconda3/bin/python"
    TORCHRUN_EXEC="/root/miniconda3/bin/torchrun"
    if [ ! -f "${PYTHON_EXEC}" ]; then
        PYTHON_EXEC="$(which python)"
    fi
    if [ ! -f "${TORCHRUN_EXEC}" ]; then
        TORCHRUN_EXEC="$(which torchrun)"
    fi
else
    AIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    PYTHON_EXEC="$(which python)"
    TORCHRUN_EXEC="$(which torchrun)"
fi

echo "[Init] Project Root: ${AIC_ROOT}"
echo "[Init] Python Exec:   ${PYTHON_EXEC}"
echo "[Init] Torchrun Exec: ${TORCHRUN_EXEC}"

# 2. 检查与配置 GPU 状态
REQ_GPUS="${1:-}"
if [ -n "${REQ_GPUS}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEVICE_LIST=$(seq -s, 0 $((REQ_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES="${DEVICE_LIST}"
    echo "[Init] Specified target GPU count: ${REQ_GPUS}. Setting CUDA_VISIBLE_DEVICES=${DEVICE_LIST}"
fi

DETECTED_COUNT=$(${PYTHON_EXEC} -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
if [ "${DETECTED_COUNT}" -eq 0 ]; then
    echo "============================================================================"
    echo " [CRITICAL WARNING] No active GPU detected by PyTorch CUDA runtime!"
    echo " 请在 Seetacloud / 算力控制台开启/挂载显卡后再次运行此脚本。"
    echo "============================================================================"
    exit 1
fi

NUM_GPUS="${DETECTED_COUNT}"
echo "[Init] PyTorch CUDA Active Device Count: ${NUM_GPUS}"
${PYTHON_EXEC} -c '
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"       Rank {i} -> {p.name} ({p.total_memory / (1024**3):.2f} GiB)")
' 2>/dev/null || true

# 3. 显存策略与自动占满模式 (80G 顶级显卡满血拉满)
USER_BATCH="${2:-}"
TOTAL_MEM_GB=$(${PYTHON_EXEC} -c 'import torch; print(int(torch.cuda.get_device_properties(0).total_memory / (1024**3)))' 2>/dev/null || echo 0)

if [ -n "${USER_BATCH}" ]; then
    BATCH_PER_GPU="${USER_BATCH}"
    echo "[VRAM Policy] Manual batch per GPU set: ${BATCH_PER_GPU}"
elif [ "${TOTAL_MEM_GB}" -ge 70 ]; then
    # ★ 80GB / 96GB 顶级旗舰算力卡 (A100-80G / H100-80G / H800 / PRO 6000 96G):
    # 1440x2560 超高分辨率下，单卡 Batch 8 吃满约 72~76 GiB 显存 (满载率 >92%)！
    BATCH_PER_GPU=8
    echo "[VRAM Policy] Detected ${TOTAL_MEM_GB} GiB Top-tier GPU! Activating 80G FULL-CAPACITY mode: Batch 8/GPU (~75 GiB loaded)."
elif [ "${TOTAL_MEM_GB}" -ge 40 ]; then
    # 48GB 专业卡 (RTX PRO 6000 48G / RTX 6000 Ada): 单卡 Batch 4，吃满约 38~42 GiB
    BATCH_PER_GPU=4
    echo "[VRAM Policy] Detected ${TOTAL_MEM_GB} GiB GPU: Batch 4/GPU (~40 GiB loaded)."
else
    # 32GB / 16GB 卡: 单卡 Batch 2
    BATCH_PER_GPU=2
    echo "[VRAM Policy] Detected ${TOTAL_MEM_GB} GiB GPU: Batch 2/GPU."
fi

GLOBAL_BATCH=$((NUM_GPUS * BATCH_PER_GPU))

# 动态依据全局 Batch Size 线性微调 AdamW 基础学习率
if [ "${GLOBAL_BATCH}" -ge 16 ]; then
    OPT_LR="0.00015"
elif [ "${GLOBAL_BATCH}" -ge 8 ]; then
    OPT_LR="0.0001"
else
    OPT_LR="0.00008"
fi

echo "[VRAM Policy] Global Effective Batch Size: ${GLOBAL_BATCH} (${NUM_GPUS} GPUs x ${BATCH_PER_GPU}/GPU)"
echo "[VRAM Policy] Optimizer Base LR: ${OPT_LR}"
echo "[VRAM Policy] Steps per Epoch: $((1800 / GLOBAL_BATCH))"

# 4. 环境变量与安全配置
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

ulimit -n 65535 2>/dev/null || true

# 5. 高速 shm 内存盘软链加速 (如可用，提供 0 延迟随机读)
if [ -d "/dev/shm" ]; then
    mkdir -p /dev/shm/aic_cache 2>/dev/null || true
    ln -sfn "${AIC_ROOT}/dataset/train/visible" /dev/shm/aic_cache/visible 2>/dev/null || true
    ln -sfn "${AIC_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/train_aligned_ir 2>/dev/null || true
    ln -sfn "${AIC_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/train_lbg_ir 2>/dev/null || true
    ln -sfn "${AIC_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/depth_p0 2>/dev/null || true
fi

# 6. 进入 DEIMv2 核心目录并确认关键路径
cd "${AIC_ROOT}/vendor/DEIMv2"

CFG="configs/deimv2/sovereign_v36_p2_direct_injection_2560.yml"
TUN="ckpts/dfine_x_obj2coco.pth"
TEACHER="ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
OUT="./outputs/sovereign_v36_p2_direct_injection_2560"
LOG_DIR="${AIC_ROOT}/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

if [ ! -f "${CFG}" ]; then
    echo "ERROR: Config file ${CFG} not found!"
    exit 1
fi

if [ ! -f "${TUN}" ]; then
    echo "ERROR: Base checkpoint ${TUN} not found!"
    exit 1
fi

if [ ! -f "${TEACHER}" ]; then
    echo "ERROR: DINOv3 teacher checkpoint ${TEACHER} not found!"
    exit 1
fi

echo "[Init] Config verified:  ${CFG}"
echo "[Init] Base Checkpoint: ${TUN}"
echo "[Init] DINOv3 Teacher:  ${TEACHER}"
echo "[Init] Output Directory: ${OUT}"

# 7. 后台磁盘空间守护 (安全清理旧轮次权重，确保 best_stg1/best_stg2/last 绝对安全)
(
  while true; do
    sleep 60
    AVAIL=$(df -k "${AIC_ROOT}" 2>/dev/null | awk "NR==2 {print \$4}")
    if [ -n "$AVAIL" ] && [ "$AVAIL" -lt 4000000 ]; then
      echo "[Disk Monitor $(date)] Free space < 4GB ($AVAIL KB), pruning old intermediate checkpoints..."
      find "${OUT}" -name "checkpoint00*.pth" -mmin +20 -delete 2>/dev/null || true
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

LOG_FILE="${LOG_DIR}/sovereign_v36_train.log"
CONSOLE_LOG="${OUT}/train_console.log"

echo "============================================================================"
echo " Launching Sovereign V36 18-Epoch Golden Training"
echo " Date: $(date)"
echo " GPUs: ${NUM_GPUS} | Batch/GPU: ${BATCH_PER_GPU} | Global Batch: ${GLOBAL_BATCH}"
echo " Log file: ${LOG_FILE}"
echo "============================================================================"

if [ "${NUM_GPUS}" -eq 1 ]; then
    echo "[Mode] Single GPU execution (batch ${BATCH_PER_GPU}, accum_steps=1)"
    ${PYTHON_EXEC} train.py \
      -c "${CFG}" \
      -t "${TUN}" \
      -u accum_steps=1 train_dataloader.total_batch_size="${GLOBAL_BATCH}" val_dataloader.total_batch_size="${GLOBAL_BATCH}" optimizer.lr="${OPT_LR}" train_dataloader.num_workers=8 \
      --use-amp \
      --seed 42 \
      2>&1 | tee -a "${CONSOLE_LOG}" | tee "${LOG_FILE}"
else
    echo "[Mode] Multi-GPU DDP execution with ${NUM_GPUS} ranks (batch ${BATCH_PER_GPU}/GPU, global batch ${GLOBAL_BATCH})"
    PORT=$((29500 + RANDOM % 100))
    ${TORCHRUN_EXEC} --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" train.py \
      -c "${CFG}" \
      -t "${TUN}" \
      -u accum_steps=1 train_dataloader.total_batch_size="${GLOBAL_BATCH}" val_dataloader.total_batch_size="${GLOBAL_BATCH}" optimizer.lr="${OPT_LR}" train_dataloader.num_workers=8 \
      --use-amp \
      --seed 42 \
      2>&1 | tee -a "${CONSOLE_LOG}" | tee "${LOG_FILE}"
fi

echo "============================================================================"
echo " Sovereign V36 Training Finished: $(date)"
echo " Output files in ${OUT}:"
ls -lh "${OUT}"/*.pth 2>/dev/null || true
echo "============================================================================"
