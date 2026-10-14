#!/bin/bash
# Channel (modality) ablation on the FULLY-INTEGRATED model (midboost_combo = L2+L3+L4).
# Mirrors run_deepir_ablate.sh: same deepir val configs (midboost reuses them),
# same ABLATE=heat|rawir|depth occlusion env var, only the checkpoint differs.
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo/checkpoint0049.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"
ABL="rawir heat depth"
echo "START ALL $(date)" >> "$W/midboost_ablate_run.log"
for ab in $ABL; do
  for lv in $LV; do
    cfg=${PREF}${lv}.yml
    [ "$lv" = "day" ] && cfg=${PREF}orig.yml
    echo "=== START $ab $lv $(date) ===" >> "$W/midboost_ablate_run.log"
    ABLATE=$ab $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/midboost_combo_ablate_${ab}_${lv}.log" 2>&1
    echo "=== DONE $ab $lv exit=$? $(date) ===" >> "$W/midboost_ablate_run.log"
  done
done
echo "=== ALL DONE $(date) ===" >> "$W/midboost_ablate_run.log"
cd "$W" && $PY /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_midboost_ablation.py >> "$W/midboost_ablate_run.log" 2>&1
echo "=== FIGURE REGENERATED $(date) ===" >> "$W/midboost_ablate_run.log"
