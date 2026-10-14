#!/usr/bin/env bash
# DEIM-D-FINE-X loss_mal=2 单变量实验 (对照 ft_dcn)
set -u
ulimit -n 65535
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_mal2.yml
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_mal2

echo "===== START DFINE_X_MAL2 $(date) =====" >> "$W/dfine_x_mal2_run.log"
$PY train.py -c "$CFG" -t ckpts/dfine_x_obj2coco.pth --use-amp > "$W/dfine_x_mal2_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_mal2_run.log"
if [ "$TRC" -ne 0 ]; then
  echo "TRAIN FAILED (rc=$TRC), skip evals" >> "$W/dfine_x_mal2_run.log"
  exit 1
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "DFINE_X_MAL2 CKPT=$CKPT" >> "$W/dfine_x_mal2_run.log"
[ -z "$CKPT" ] && { echo "NO CKPT, abort eval"; exit 1; }

PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_mal2_val_
for lv in day night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== dfine_x_mal2 eval $lv $(date) ===" >> "$W/dfine_x_mal2_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_mal2_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_mal2_run.log"
done
echo "ALL DONE DFINE_X_MAL2 $(date)" >> "$W/dfine_x_mal2_run.log"
