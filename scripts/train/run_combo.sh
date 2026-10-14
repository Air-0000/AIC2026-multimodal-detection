#!/bin/bash
# 组合轮: L2+L3+L4 叠加, 从 deepir base 出发, 训练 + 10档评测 + 出图
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
SRC=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START COMBO $(date) =====" >> "$W/combo_run.log"
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo.yml -t $SRC --use-amp > "$W/midboost_combo_train.log" 2>&1
echo "TRAIN DONE COMBO exit=$? $(date)" >> "$W/combo_run.log"
CKPT=$(ls -t outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo/*.pth 2>/dev/null | head -1)
echo "COMBO CKPT=$CKPT" >> "$W/combo_run.log"
for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== combo eval $lv $(date) ===" >> "$W/combo_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/midboost_combo_${lv}.log" 2>&1
done
echo "EVAL DONE COMBO $(date)" >> "$W/combo_run.log"
cd "$W" && python3 /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_combo.py >> "$W/combo_run.log" 2>&1
echo "ALL DONE COMBO $(date)" >> "$W/combo_run.log"
