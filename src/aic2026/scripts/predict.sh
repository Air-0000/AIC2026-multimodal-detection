#!/usr/bin/env bash
# =============================================================
# AIC 2026 — Track 1: M2I2HA Baseline 推理 & 提交脚本
# =============================================================
set -e

# ---------- 默认配置 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT="${PROJECT_DIR}/runs/aic2026_baseline/weights/best.pt"
DATA_DIR=""  # 必须由用户指定
SPLIT="test"
OUTPUT="${PROJECT_DIR}/submissions/m2i2ha_baseline"
CONF=0.25
IOU=0.7
DEVICE="auto"

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)      CKPT="$2"; shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --split)     SPLIT="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --conf)      CONF="$2"; shift 2 ;;
        --iou)       IOU="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --help|-h)
            echo "用法: bash predict.sh [选项]"
            echo "  --ckpt PATH     模型权重 (默认: runs/aic2026_baseline/weights/best.pt)"
            echo "  --data-dir PATH 竞赛数据根目录 (必填)"
            echo "  --split NAME    数据集划分: test|val (默认: test)"
            echo "  --output DIR    输出目录 (默认: submissions/m2i2ha_baseline)"
            echo "  --conf NUM      置信度阈值 (默认: 0.25)"
            echo "  --iou NUM       NMS IoU 阈值 (默认: 0.7)"
            echo "  --device DEVICE 设备 (auto/cuda/cpu)"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---------- 检查 ----------
if [[ -z "$DATA_DIR" ]]; then
    echo "[ERROR] 请指定 --data-dir"
    echo "  用法: bash predict.sh --data-dir /path/to/aic2026/data"
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] 模型权重未找到: $CKPT"
    echo "  请先训练 Baseline，或指定 --ckpt"
    exit 1
fi

# ---------- 运行 ----------
echo "========== AIC 2026 — M2I2HA Baseline 推理 =========="
echo "  权重:    $CKPT"
echo "  数据:    $DATA_DIR"
echo "  划分:    $SPLIT"
echo "  输出:    $OUTPUT"
echo "  置信度:  $CONF"
echo "  IoU:     $IOU"
echo "  设备:    $DEVICE"
echo "===================================================="

cd "$PROJECT_DIR"
python "$SCRIPT_DIR/predict.py" \
    --ckpt "$CKPT" \
    --data-dir "$DATA_DIR" \
    --split "$SPLIT" \
    --output "$OUTPUT" \
    --conf "$CONF" \
    --iou "$IOU" \
    --device "$DEVICE"

echo ""
echo "✅ 推理完成！提交文件: ${OUTPUT}/submission.zip"
echo "   将 submission.zip 上传至竞赛官网即可。"
