#!/usr/bin/env bash
# Rerun 10-level night eval for vitl16_l4like_640 (fixed configs: vitl16 backbone)
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG_DIR=configs/deimv2
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl16_l4like_640_val_
CKPT=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/vitl16_l4like_640_scratch/last.pth

for lv in day night50 night25 night12 night05 night01 night005 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== vitl16l4like eval $lv $(date) ===" >> "$W/vitl16_l4like_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitl16_l4like_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/vitl16_l4like_run.log"
done
echo "ALL DONE VITL16_L4LIKE_EVAL_FIXED $(date)" >> "$W/vitl16_l4like_run.log"
