#!/bin/bash
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
SRC=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
MB=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost
CFG=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost.yml
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
echo "START MIDBOOST $(date)" >> "$W/midboost_run.log"
$PY train.py -c $CFG -t $SRC --use-amp > "$W/midboost_train.log" 2>&1
echo "TRAIN DONE exit=$? $(date)" >> "$W/midboost_run.log"
MB_CKPT=$(ls -t "$MB"/*.pth 2>/dev/null | head -1)
echo "MB_CKPT=$MB_CKPT" >> "$W/midboost_run.log"
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"
for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== eval $lv $(date) ===" >> "$W/midboost_run.log"
  $PY train.py -c "$cfg" -r "$MB_CKPT" --use-amp --test-only > "$W/midboost_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/midboost_run.log"
cd "$W" && python3 /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_midboost.py >> "$W/midboost_run.log" 2>&1
echo "FIGURE DONE $(date)" >> "$W/midboost_run.log"
