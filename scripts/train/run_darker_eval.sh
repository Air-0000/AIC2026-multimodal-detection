#!/bin/bash
# 杠杆④ 拓展: ufall 5ch 裸深度底座在更黑档(night002/night001/night0005)评测。
# 提取 bbox AP@0.50:0.95 汇总到 darker_eval_summary.txt
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OD=/home/constant/CodeBuddy/Dev/AIC/work_dir
CFGD=configs/deimv2
CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/best_stg2.pth
SUMMARY=$OD/logs_text/darker_eval_summary.txt
: > "$SUMMARY"

run() {
  local tag=$1 cfg=$2
  local log=$OD/ufall_${tag}_eval.log
  echo "===== $tag =====" | tee -a "$SUMMARY"
  $PY train.py -c "$CFGD/$cfg" --test-only -r "$CKPT" > "$log" 2>&1
  local ap
  ap=$(grep -E "IoU=0.50:0.95 \| area=   all \| maxDets=100" "$log" | head -1 | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  echo "$tag  AP=$ap" | tee -a "$SUMMARY"
}

run night002 deimv2_dinov3_l_aic_val_ufall_night002.yml
run night001 deimv2_dinov3_l_aic_val_ufall_night001.yml
run night0005 deimv2_dinov3_l_aic_val_ufall_night0005.yml

echo "DONE" | tee -a "$SUMMARY"
