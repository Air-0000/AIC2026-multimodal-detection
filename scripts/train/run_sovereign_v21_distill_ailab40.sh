#!/usr/bin/env bash
# =============================================================================
# Sovereign V21 端到端单阶段知识蒸馏主战启动脚本
# 运行环境: 实验室科研服务器 ailab40 (双卡 Tesla V100 32GB)
# 用户身份: 刘正阳 (liuzhengyang)
# -----------------------------------------------------------------------------
# 架构配置:
#   - 教师网络: Sovereign V20 58.5% 历史巅峰模型汤 (teacher_sovereign_v20_585.pth)
#   - 学生网络: Sovereign V21 终极微小目标攻坚架构 (StarConv + Spatial-PBD + LMM + P2)
#   - 热启基座: sovereign_v14_soup.pth
#   - 数据总线: /dev/shm/aic_cache 48GB 内存盘 (0延迟物理读取)
#   - 分布式: 双卡 DDP (Backend: Gloo, 规避 NVML 拓扑死锁)
# =============================================================================
set -e
set -u

BASE_DIR="/home/ailab/ssd/bigdata/liuzhengyang"
DEIM_DIR="${BASE_DIR}/deimv2"
LOG_DIR="${BASE_DIR}/logs"
DATA_DIR="${BASE_DIR}/data"
WEIGHT_DIR="${BASE_DIR}/weights"
OUT_DIR="${BASE_DIR}/outputs/sovereign_v21_distill_ailab40"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"

cd "${DEIM_DIR}"

ulimit -n 65535 2>/dev/null || true

# 1. 自动预热装载 /dev/shm 内存盘 (48GB 内存盘常驻，0 延迟 I/O)
SHM_DIR="/dev/shm/aic_cache"
if [ ! -d "${SHM_DIR}/visible" ]; then
    echo "=========================================================="
    echo ">>> Preloading 4-modality dataset into /dev/shm RAM disk..."
    mkdir -p "${SHM_DIR}"
    
    # 异步多进程极速拷贝 (只需 ~5 秒)
    cp -r "${DATA_DIR}/local_official_full_20260915/visible" "${SHM_DIR}/" &
    cp -r "${DATA_DIR}/preprocessed/train_aligned_ir" "${SHM_DIR}/" &
    cp -r "${DATA_DIR}/preprocessed/train_lbg_ir" "${SHM_DIR}/" &
    cp -r "${DATA_DIR}/preprocessed/depth_p0" "${SHM_DIR}/" &
    wait
    
    echo ">>> All channels preloaded into /dev/shm successfully!"
    df -h /dev/shm
fi

# 2. 环境变量设定 (双卡 V100 优化)
export PYTHONPATH="${DEIM_DIR}:${BASE_DIR}/scripts/lib:${PYTHONPATH:-}"
export TORCH_CUDA_MATMUL_NO_CUBLASLT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

CFG="configs/deimv2/sovereign_v21_distill_ailab40.yml"
TUN="${WEIGHT_DIR}/sovereign_v14_soup.pth"
TEACHER="${WEIGHT_DIR}/teacher_sovereign_v20_585.pth"
LOG_FILE="${LOG_DIR}/sovereign_v21_distill_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "Starting Sovereign V21 End-to-End Distillation Training"
echo "Host: ailab40 (Tesla V100 32GB @ GPU 1)"
echo "Date: $(date)"
echo "Config: ${CFG}"
echo "Student Hot-Start: ${TUN}"
echo "Teacher Model: ${TEACHER} (58.5% Sovereign Apex)"
echo "Memory Cache: ${SHM_DIR} (Active)"
echo "Logging to: ${LOG_FILE}"
echo "=========================================================="

nvidia-smi

/home/ailab40/anaconda3/envs/rfdetr/bin/python train.py \
    -c "${CFG}" \
    -t "${TUN}" \
    --use-amp 2>&1 | tee "${LOG_FILE}"

echo "=========================================================="
echo "Sovereign V21 Distillation Training Complete: $(date)"
echo "=========================================================="
