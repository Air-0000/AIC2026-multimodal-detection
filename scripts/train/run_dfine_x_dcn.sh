#!/usr/bin/env bash
# DEIM-D-FINE-X (HGNetv2-B5) AIC 单阶段微调 + day/9档night 评测
# 起点: ckpts/dfine_x_obj2coco.pth (官方 O365→COCO 微调成品, COCO AP 59.3)
# 配置: deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn.yml (官方 obj2coco_24e 配方, AIC 化)
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn.yml
OUT=./outputs/deim_hgnetv2_x_aic_obj2coco_ft_dcn

echo "===== START DFINE_X_FT $(date) =====" >> "$W/dfine_x_dcn_run.log"
$PY train.py -c "$CFG" -t ckpts/dfine_x_obj2coco.pth --use-amp > "$W/dfine_x_dcn_train.log" 2>&1
TRC=$?
echo "train rc=$TRC" >> "$W/dfine_x_dcn_run.log"
if [ "$TRC" -ne 0 ]; then
  echo "TRAIN FAILED (rc=$TRC), skip evals" >> "$W/dfine_x_dcn_run.log"
  exit 1
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "DFINE_X CKPT=$CKPT" >> "$W/dfine_x_dcn_run.log"
[ -z "$CKPT" ] && { echo "NO CKPT, abort eval"; exit 1; }

PREF=configs/deimv2/deim_hgnetv2_x_aic_obj2coco_ft_dcn_val_
for lv in day night50 night25 night12 night05 night005 night01 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== dfine_x eval $lv $(date) ===" >> "$W/dfine_x_dcn_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/dfine_x_dcn_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/dfine_x_dcn_run.log"
done
echo "ALL DONE DFINE_X_FT $(date)" >> "$W/dfine_x_dcn_run.log"
