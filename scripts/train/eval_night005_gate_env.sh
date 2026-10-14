#!/usr/bin/env bash
# night005 极端暗光复评 (ENV-inspired 门控模型) -- 安全版
# 训练进程占卡时自动跳过，避免与训练抢显存导致 OOM。
set -e

TRAIN_OUT=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_geo_gate
LOG=/home/constant/CodeBuddy/Dev/AIC/logs/eval_night005_gate_env.log
SMOKE=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_geo_gate_val_night005_smoke.yml

# 训练进程占卡 -> 跳过
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "[$(date)] GPU busy (training running), skip night005 eval" | tee -a "$LOG"
  exit 0
fi

# 找 best ckpt (优先 checkpoint_best* / best_stg*，否则最新 checkpoint*)
CKPT=$(ls -t "$TRAIN_OUT"/checkpoint_best*.pth "$TRAIN_OUT"/best_stg*.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t "$TRAIN_OUT"/checkpoint*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[$(date)] no checkpoint yet, skip" | tee -a "$LOG"
  exit 0
fi

echo "[$(date)] night005 eval with $CKPT" | tee -a "$LOG"
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
/home/constant/miniconda3/envs/deimv2/bin/python train.py -c "$SMOKE" --test-only -r "$CKPT" --use-amp 2>&1 | tee -a "$LOG"
