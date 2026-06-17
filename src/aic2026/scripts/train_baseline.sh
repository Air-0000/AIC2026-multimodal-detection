#!/usr/bin/env bash
# =============================================================
# AIC 2026 — Track 1: M2I2HA Baseline 训练脚本
# 在实验室服务器上运行
# 使用方式:
#   bash src/aic2026/scripts/train_baseline.sh
#   bash src/aic2026/scripts/train_baseline.sh --device 0,1  # 多卡
#   bash src/aic2026/scripts/train_baseline.sh --batch 8     # 低显存
# =============================================================
set -e

# ---------- 配置 ----------
PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

DATASET_CFG="${PROJECT_DIR}/configs/aic_baseline.yaml"
ALGO_CFG="${PROJECT_DIR}/src/aic2026/configs/baseline.yaml"
OUTPUT_DIR="runs/aic2026_baseline"
DEVICE="0"                 # GPU ID 或 "0,1" (多卡)
BATCH_SIZE=16
EPOCHS=300
IMG_SIZE=640
NET_SCALE="s"              # "n" | "s" | "m"
USE_AMP=""                 # 默认不开 AMP，--amp 启用

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)  DEVICE="$2"; shift 2 ;;
        --batch)   BATCH_SIZE="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --scale)   NET_SCALE="$2"; shift 2 ;;
        --amp)     USE_AMP="--amp"; shift ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "  --device ID    GPU ID (默认: 0, 多卡用逗号分隔: 0,1)"
            echo "  --batch N      Batch size (默认: 16)"
            echo "  --epochs N     Epochs (默认: 300)"
            echo "  --scale NAME   模型规模 n/s/m (默认: s)"
            echo "  --amp          启用混合精度训练"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---------- 环境检查 ----------
if ! python -c "import machine_learning" 2>/dev/null; then
    echo "[ERROR] M2I2HA (machine_learning) 未安装，请先运行 setup_linux.sh"
    exit 1
fi

if [[ ! -f "$DATASET_CFG" ]]; then
    echo "[ERROR] 数据集配置未找到: $DATASET_CFG"
    exit 1
fi

# ---------- 训练 ----------
echo "========== AIC 2026 — M2I2HA Baseline 训练 =========="
echo "  设备:    $DEVICE"
echo "  Batch:   $BATCH_SIZE"
echo "  Epochs:  $EPOCHS"
echo "  模型:    ${NET_SCALE}"
echo "  AMP:     $([ -n "$USE_AMP" ] && echo '开启' || echo '关闭')"
echo "  数据:    $DATASET_CFG"
echo "===================================================="

cd "$PROJECT_DIR"

python -m machine_learning.train \
    --name "aic2026_m2i2ha_baseline" \
    --cfg "${ALGO_CFG}" \
    --dataset "${DATASET_CFG}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMG_SIZE}" \
    --net_scale "${NET_SCALE}" \
    --augment \
    --save_best \
    ${USE_AMP}
