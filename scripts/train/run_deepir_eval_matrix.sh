#!/bin/bash
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CK=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
CFGD=configs/deimv2
OD=/home/constant/CodeBuddy/Dev/AIC/work_dir
SUMMARY=$OD/logs_text/deepir_eval_matrix_summary.txt
: > "$SUMMARY"
LEVELS="orig night0005 night001 night002 night005 night01 night05 night12 night25 night50 nightify"
cd "$DEIM" || exit 1
for lv in $LEVELS; do
  cfg=${CFGD}/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_${lv}.yml
  log=$OD/deepir_eval_${lv}.log
  echo "===== $lv =====" | tee -a "$SUMMARY"
  $PY train.py -c "$cfg" -r "$CK" --use-amp --test-only > "$log" 2>&1
  ap=$(grep -aE "IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = " "$log" | head -1 | sed -E 's/.*\] = //' | tr -d '\r')
  ap50=$(grep -aE "IoU=0.50 \| area=   all \| maxDets=100 \] = " "$log" | head -1 | sed -E 's/.*\] = //' | tr -d '\r')
  echo "$lv  AP=$ap  AP50=$ap50" | tee -a "$SUMMARY"
done
echo "DONE" | tee -a "$SUMMARY"
