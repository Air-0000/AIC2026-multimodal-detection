#!/bin/bash
# =============================================================================
# Sovereign V29: Meta DINOv3-ViT-Large 微创在线特征蒸馏启动脚本
# 基底: Sovereign V23 Gold Checkpoint (88.83% mAP, 严禁使用任何退化权重)
# =============================================================================
set -e

# 1. 动态自适应根路径识别 (支持本地与云端两用)
if [ -d "/root/autodl-tmp/AIC" ]; then
    PROJECT_ROOT="/root/autodl-tmp/AIC"
    PYTHON_EXEC="/root/miniconda3/bin/python"
    DEVICE_STR="8-GPU / Single GPU Cloud Node"
else
    PROJECT_ROOT="/home/constant/CodeBuddy/Dev/AIC"
    PYTHON_EXEC="/home/constant/miniconda3/envs/deimv2/bin/python"
    DEVICE_STR="Local RTX 5060 Ti Node"
fi

echo "=========================================================================="
echo "=== 启动 Sovereign V29: DINOv3-ViT-Large 语义特征在线微创手术蒸馏 ==="
echo "=========================================================================="
echo "执行平台: $DEVICE_STR"
echo "项目根目录: $PROJECT_ROOT"
echo "Python解释器: $PYTHON_EXEC"

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_ROOT}/scripts/lib:${PROJECT_ROOT}/vendor/DEIMv2:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}/vendor/DEIMv2"
mkdir -p outputs/sovereign_v29_dinov3_distill_v23_base

# 2. 内存高速缓存链路保障
mkdir -p /dev/shm/aic_cache 2>/dev/null || true
if [ -d "/root/autodl-tmp/AIC" ]; then
    ln -sfn "${PROJECT_ROOT}/dataset/train/visible" /dev/shm/aic_cache/visible 2>/dev/null || true
    ln -sfn "${PROJECT_ROOT}/work_dir/train_aligned_ir" /dev/shm/aic_cache/train_aligned_ir 2>/dev/null || true
    ln -sfn "${PROJECT_ROOT}/work_dir/train_lbg_ir" /dev/shm/aic_cache/train_lbg_ir 2>/dev/null || true
    ln -sfn "${PROJECT_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/depth_p0 2>/dev/null || true
else
    ln -sfn "${PROJECT_ROOT}/dataset/train/visible" /dev/shm/aic_cache/visible 2>/dev/null || true
    ln -sfn "${PROJECT_ROOT}/dataset/train/infrared_aligned" /dev/shm/aic_cache/train_aligned_ir 2>/dev/null || true
    ln -sfn "${PROJECT_ROOT}/dataset/train/infrared" /dev/shm/aic_cache/train_lbg_ir 2>/dev/null || true
    ln -sfn "${PROJECT_ROOT}/dataset/train/depth_p0" /dev/shm/aic_cache/depth_p0 2>/dev/null || true
fi

# 3. 锁定 Sovereign V23 纯净基底权重
START_CKPT="${PROJECT_ROOT}/weights/sovereign_v23_best_stg2_ep71.pth"
if [ ! -f "$START_CKPT" ]; then
    START_CKPT="${PROJECT_ROOT}/vendor/DEIMv2/outputs/sovereign_v23_tadl_oldlabel1800_2048/best_stg2.pth"
fi

if [ ! -f "$START_CKPT" ]; then
    echo "❌ 严重错误: 找不到 Sovereign V23 纯净基底权重: $START_CKPT"
    exit 1
fi
echo "黄金学生基底: $START_CKPT (大小: $(ls -lh "$START_CKPT" | awk '{print $5}'))"

# 4. 检查 DINOv3 教师模型权重
TEACHER_DIR="${PROJECT_ROOT}/weights/dinov3_large"
if [ ! -f "${TEACHER_DIR}/model.safetensors" ]; then
    echo "⚠️ 提示: ${TEACHER_DIR}/model.safetensors 尚未就绪，请等待下载完成或完成同步。"
else
    echo "冻结教师权重: ${TEACHER_DIR}/model.safetensors (大小: $(ls -lh "${TEACHER_DIR}/model.safetensors" | awk '{print $5}'))"
fi

# 5. 云端磁盘空间保护守护进程
if [ -d "/root/autodl-tmp/AIC" ]; then
    (
        while true; do
            sleep 60
            AVAIL=$(df /root/autodl-tmp | awk "NR==2 {print \$4}")
            if [ "$AVAIL" -lt 2500000 ]; then
                echo "[Disk Monitor $(date)] 磁盘可用空间 < 2.5GB ($AVAIL KB), 正在清理中间 checkpoints..."
                find outputs/sovereign_v29_dinov3_distill_v23_base/ -name "checkpoint000*.pth" -mmin +20 -delete 2>/dev/null || true
            fi
        done
    ) &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null || true" EXIT
fi

# 6. 正式驱动训练
echo "=== 开始训练: $(date) ==="
$PYTHON_EXEC train.py \
    -c configs/deimv2/sovereign_v29_dinov3_distill_v23_base.yml \
    -t "$START_CKPT" \
    --use-amp \
    --seed 42 \
    2>&1 | tee -a outputs/sovereign_v29_dinov3_distill_v23_base/train_console.log

echo "=== Sovereign V29 训练完成: $(date) ==="
