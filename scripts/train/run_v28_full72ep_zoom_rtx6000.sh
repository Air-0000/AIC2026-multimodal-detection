#!/usr/bin/env bash
# =============================================================================
# Sovereign V28 区域放大觉醒 72-Epoch 全满血淬炼启动脚本
# 算力平台: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, 600W 满血环境)
# 基础起点: sovereign_v23_best_stg2_ep71.pth (71 轮成熟小目标高置信度模型)
# 核心架构: P2 RoI Zoom 区域放大检测 + PRCG 刚体门控 + Soft-Containment 软消解 + DCDA 视差觉醒
# =============================================================================
set -e
set -u

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -d "/root/autodl-tmp/AIC" ]; then
    AIC_ROOT="/root/autodl-tmp/AIC"
    PYTHON_EXEC="/root/miniconda3/bin/python"
    if [ ! -f "${PYTHON_EXEC}" ]; then
        PYTHON_EXEC="$(which python)"
    fi
else
    AIC_ROOT="/home/constant/CodeBuddy/Dev/AIC"
    PYTHON_EXEC="/home/constant/miniconda3/envs/deimv2/bin/python"
    if [ ! -f "${PYTHON_EXEC}" ]; then
        PYTHON_EXEC="$(which python)"
    fi
fi

cd "${AIC_ROOT}/vendor/DEIMv2"
ulimit -n 65535 2>/dev/null || true

export PYTHONPATH="${AIC_ROOT}/vendor/DEIMv2:${AIC_ROOT}/vendor/Depth-Anything-V2:${AIC_ROOT}/scripts/lib:${PYTHONPATH:-}"

# 1. 确保高速 shm 内存缓存链路极速直读
mkdir -p /dev/shm/aic_cache
ln -sfn "${AIC_ROOT}/dataset/train/visible" /dev/shm/aic_cache/visible
ln -sfn "${AIC_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/train_aligned_ir
ln -sfn "${AIC_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/train_lbg_ir
ln -sfn "${AIC_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/depth_p0
echo "[$(date)] /dev/shm/aic_cache memory disk links are ready!"

LOG_DIR="${AIC_ROOT}/logs"
CFG="configs/deimv2/sovereign_v28_full72ep_zoom_awakening_rtx6000.yml"
OUT="./outputs/sovereign_v28_full72ep_zoom_awakening_rtx6000"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}"

# 定位 V23 71-Epoch 基础成熟权重
if [ -f "${AIC_ROOT}/vendor/DEIMv2/outputs/sovereign_v23_tadl_oldlabel1800_2048/best_stg2.pth" ]; then
    TUN="${AIC_ROOT}/vendor/DEIMv2/outputs/sovereign_v23_tadl_oldlabel1800_2048/best_stg2.pth"
elif [ -f "${AIC_ROOT}/weights/sovereign_v23_best_stg2_ep71.pth" ]; then
    TUN="${AIC_ROOT}/weights/sovereign_v23_best_stg2_ep71.pth"
elif [ -f "${AIC_ROOT}/weight/sovereign_v23_best_stg2_ep71.pth" ]; then
    TUN="${AIC_ROOT}/weight/sovereign_v23_best_stg2_ep71.pth"
else
    echo "ERROR: Cannot find sovereign_v23_best_stg2_ep71.pth!"
    exit 1
fi

echo "============================================================================"
echo " Starting Sovereign V28 72-Epoch Zoom Awakening Full Run on RTX PRO 6000"
echo " Date: $(date)"
echo " Config: ${CFG}"
echo " Pretrained Tuning Hot-Start: ${TUN}"
echo " Resolution: 1152 x 2048 (Strict 16:9 Zero Distortion, 2^11 CUDA aligned)"
echo " Batch Size: 8 per GPU x 2 accum = 16 effective batch"
echo " Dataset: instances_train_quality1998.json (1998 images, 17921 boxes)"
echo " Epochs: 72 (Flat: 48, Clean No-Aug Annealing: 6)"
echo " Enabled Features:"
echo "   - use_p2_cascade_zoom: True (Regional Zoom / P2 Saliency Booster)"
echo "   - use_planar_rigidity_gate: True (PRCG Rigid 2D Plane vs Bio Prior)"
echo "   - use_containment_fusion: True (Soft-Containment IoU>=0.85, Cont>=0.88)"
echo "   - use_dcda: True (loss_cmal: 0.5, ir_align lr: 1e-4)"
echo "   - use_midboost: True (Adaptive illumination gating)"
echo "   - use_internal_tta: True (Native Horizontal Flip TTA Engine)"
echo "============================================================================"

nvidia-smi

# 守护磁盘空间，防盘满 (自动剪枝旧的中间轮次 checkpoint，安全保留 best_stg1/best_stg2/last)
(
  while true; do
    sleep 60
    AVAIL=$(df "${AIC_ROOT}" | awk "NR==2 {print \$4}")
    if [ "$AVAIL" -lt 3000000 ]; then
      echo "[Disk Monitor $(date)] Free space < 3GB ($AVAIL KB), pruning old intermediate checkpoints..."
      find "${OUT}" -name "checkpoint00*.pth" -mmin +20 -delete 2>/dev/null || true
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

${PYTHON_EXEC} train.py \
  -c "${CFG}" \
  -t "${TUN}" \
  --use-amp \
  --seed 42 \
  2>&1 | tee -a "${OUT}/train_console.log" | tee "${LOG_DIR}/v28_full72ep_zoom_awakening_rtx6000.log"

echo "============================================================================"
echo "Sovereign V28 72-epoch full training completed successfully at: $(date)"
echo "============================================================================"
