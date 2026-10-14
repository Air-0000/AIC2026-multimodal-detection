#!/bin/bash
# 一键串行: L2 -> L3 -> L4 (每个都从 deepir base best_stg2 出发, 隔离单变量)
# 每个实验: 训练 + 10档夜间评测; 全部结束后画总对比图 (base/v1/L2/L3/L4/ufall)
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
SRC=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

run_exp () {
  local NAME=$1 CFG=$2 OUT=$3 EVALP=$4
  echo "===== START $NAME $(date) =====" >> "$W/all_run.log"
  $PY train.py -c "$CFG" -t "$SRC" --use-amp > "$W/${EVALP}_train.log" 2>&1
  echo "TRAIN DONE $NAME exit=$? $(date)" >> "$W/all_run.log"
  local CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
  echo "$NAME CKPT=$CKPT" >> "$W/all_run.log"
  for lv in $LV; do
    local cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
    echo "=== $NAME eval $lv $(date) ===" >> "$W/all_run.log"
    $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/${EVALP}_${lv}.log" 2>&1
  done
  echo "===== EVAL DONE $NAME $(date) =====" >> "$W/all_run.log"
}

run_exp L2 configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l2.yml outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l2 midboost_l2_eval
run_exp L3 configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l3.yml outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l3 midboost_l3_eval
run_exp L4 configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4.yml outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4 midboost_l4_eval

cd "$W" && python3 /home/constant/CodeBuddy/Dev/AIC/scripts/analysis/plot_all.py >> "$W/all_run.log" 2>&1
echo "ALL DONE $(date)" >> "$W/all_run.log"
