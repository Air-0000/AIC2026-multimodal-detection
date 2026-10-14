#!/usr/bin/env bash
# mal2 重评测 (ops 修复后的 val 配置)
set -u
ulimit -n 65535
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_mal2

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "RE-EVAL MAL2 CKPT=$CKPT $(date)" >> "$W/dfine_x_mal2_reeval.log"
[ -z "$CKPT" ] && { echo "NO CKPT"; exit 1; }

PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_mal2_val_
for lv in orig night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml
  echo "=== re-eval $lv $(date) ===" >> "$W/dfine_x_mal2_reeval.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_mal2_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_mal2_reeval.log"
done
echo "ALL DONE MAL2 RE-EVAL $(date)" >> "$W/dfine_x_mal2_reeval.log"
