#!/usr/bin/env bash
# =============================================================
# AIC 2026 — Track 1: M2I2HA Baseline 训练脚本
# 在实验室服务器上运行
# =============================================================
set -e

# ---------- 配置（部署时修改） ----------
DATASET_CFG="/path/to/aic2026/configs/aic_baseline.yaml"   # ← TODO
ALGO_CFG="src/aic2026/configs/baseline.yaml"
OUTPUT_DIR="runs/aic2026_baseline"
DEVICE="0"                 # GPU ID 或 "0,1" (多卡)
BATCH_SIZE=16
EPOCHS=300
IMG_SIZE=640
NET_SCALE="s"              # "n" | "s"

# ---------- 安装依赖 ----------
# pip install -e .
# pip install git+https://github.com/WSYANGSX/machine_learning.git

# ---------- 训练 ----------
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
    --amp
