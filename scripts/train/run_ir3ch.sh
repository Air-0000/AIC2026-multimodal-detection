#!/bin/bash
# 3ch-IR 单变量对照: 相对 midboost_l4 (6ch, best mAP 0.5390) 只加第三路 IR (de-leak)
# 判据: 与 0.5390 比较。涨 = 第三路有净增益; 持平/跌 = de-leak 与 raw 冗余。
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
SRC=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4_ir3ch.yml
CACHE=/home/constant/CodeBuddy/Dev/AIC/work_dir/deepir_cache3/train

echo "===== START IR3CH $(date) =====" >> "$W/ir3ch_run.log"

# 0) 等 7ch 缓存建完 (1800 帧)
while true; do
  N=$(ls "$CACHE" 2>/dev/null | wc -l)
  if [ "$N" -ge 1800 ]; then break; fi
  echo "waiting cache: $N/1800 $(date)" >> "$W/ir3ch_run.log"
  sleep 20
done
echo "cache ready $(date)" >> "$W/ir3ch_run.log"

# 1) 排队: 等 GPU 空闲 (可用 >= 12GB)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/ir3ch_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/ir3ch_run.log"

$PY train.py -c "$CFG" -t "$SRC" --use-amp > "$W/ir3ch_train.log" 2>&1
echo "===== DONE IR3CH rc=$? $(date) =====" >> "$W/ir3ch_run.log"
