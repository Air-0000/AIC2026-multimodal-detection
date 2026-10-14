#!/bin/bash
# L4+L3 (不含 L2 采样) 单变量对照: 训练 + 10 档评测
# 目的: 验证删掉共线最严重的 L2 后, night01 的 -0.011 缺口能否补回 (~L3 的 +0.049 水平)
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
SRC=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
CFG=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3.yml
OUT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START L4U3 $(date) =====" >> "$W/l4u3_run.log"

# 排队: 等待 GPU 空闲(可用 >= 12GB)再启动, 避免与分割线 mask2former 抢卡。
# micro 6 峰值约 11GB, 低于 12GB 阈值必定 OOM。
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/l4u3_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/l4u3_run.log"

$PY train.py -c "$CFG" -t "$SRC" --use-amp > "$W/midboost_l4u3_train.log" 2>&1
echo "TRAIN DONE exit=$? $(date)" >> "$W/l4u3_run.log"

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "L4U3 CKPT=$CKPT" >> "$W/l4u3_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== l4u3 eval $lv $(date) ===" >> "$W/l4u3_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/midboost_l4u3_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/l4u3_run.log"

cd /home/constant/CodeBuddy/Dev/AIC && python3 scripts/analysis/plot_additivity.py >> "$W/l4u3_run.log" 2>&1
echo "ALL DONE L4U3 $(date)" >> "$W/l4u3_run.log"
