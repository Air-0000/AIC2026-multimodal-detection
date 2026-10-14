#!/bin/bash
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"
# keep_<k>: ablate everything EXCEPT k
declare -A M=(
  [heat]="rawir,depth"
  [rawir]="heat,depth"
  [depth]="heat,rawir"
  [rgb]="heat,rawir,depth"
)
echo "START ALL $(date)" >> "$W/deepir_combo_run.log"
for k in heat rawir depth rgb; do
  envv=${M[$k]}
  for lv in $LV; do
    cfg=${PREF}${lv}.yml
    [ "$lv" = "day" ] && cfg=${PREF}orig.yml
    echo "=== START keep_$k $lv $(date) ===" >> "$W/deepir_combo_run.log"
    ABLATE=$envv $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/deepir_keep_${k}_${lv}.log" 2>&1
    echo "=== DONE keep_$k $lv exit=$? $(date) ===" >> "$W/deepir_combo_run.log"
  done
done
echo "=== ALL DONE $(date) ===" >> "$W/deepir_combo_run.log"
cd "$W" && python3 /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_deepir_combo.py >> "$W/deepir_combo_run.log" 2>&1
echo "=== FIGURE REGENERATED $(date) ===" >> "$W/deepir_combo_run.log"
