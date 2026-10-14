#!/usr/bin/env bash
# dfine_x_obj2coco 断点续训 (cudaErrorLaunchTimeout 被杀后, 从 last.pth 继续)
# 注意: -r 续训 (载入 model/ema/optimizer/scaler), 评测段与主脚本一致
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft.yml
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft

echo "===== RESUME DFINE_X_FT $(date) =====" >> "$W/dfine_x_ft_run.log"
$PY train.py -c "$CFG" -r "$OUT/last.pth" --use-amp > "$W/dfine_x_ft_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_ft_run.log"
if [ "$TRC" -ne 0 ]; then
  echo "TRAIN FAILED (rc=$TRC), skip evals" >> "$W/dfine_x_ft_run.log"
  exit 1
fi

CKPT="$OUT/last.pth"
PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_val_
for lv in day night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== dfine_x eval $lv $(date) ===" >> "$W/dfine_x_ft_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_ft_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_ft_run.log"
done
echo "ALL DONE DFINE_X_FT $(date)" >> "$W/dfine_x_ft_run.log"
