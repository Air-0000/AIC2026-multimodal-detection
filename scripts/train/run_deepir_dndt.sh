#!/bin/bash
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_dndt_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="night50 night25 night12 night05 night01 night005 night002 night001 night0005"
echo "START ALL $(date)" >> "$W/deepir_dndt_run.log"
for lv in $LV; do
  echo "=== START $lv $(date) ===" >> "$W/deepir_dndt_run.log"
  $PY train.py -c "${PREF}${lv}.yml" -r "$CKPT" --use-amp --test-only > "$W/deepir_dndt_eval_${lv}.log" 2>&1
  echo "=== DONE $lv exit=$? $(date) ===" >> "$W/deepir_dndt_run.log"
done
echo "=== ALL DONE, regenerating figure $(date) ===" >> "$W/deepir_dndt_run.log"
cd "$W" && python3 /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_night_full.py >> "$W/deepir_dndt_run.log" 2>&1
echo "=== FIGURE REGENERATED $(date) ===" >> "$W/deepir_dndt_run.log"
