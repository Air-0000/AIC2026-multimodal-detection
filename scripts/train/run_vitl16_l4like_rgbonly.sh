#!/usr/bin/env bash
# vitl16 l4like RGB-only: 单段训练 + 10 档 night 评测
# 配置: deimv2_dinov3_l_aic_vitl16_l4like_640_rgbonly.yml (唯一改动 ir/depth drop=1.0)
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl16_l4like_640_rgbonly.yml
OUT=./outputs/vitl16_l4like_640_rgbonly

echo "===== START RGBONLY $(date) =====" >> "$W/vitl16_rgbonly_run.log"
$PY train.py -c "$CFG" -t ckpts/dinov3/vitl16_rdttrack_scratch_full.pth --use-amp > "$W/vitl16_rgbonly_train.log" 2>&1
echo "train rc=$?" >> "$W/vitl16_rgbonly_run.log"

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "RGBONLY CKPT=$CKPT" >> "$W/vitl16_rgbonly_run.log"
[ -z "$CKPT" ] && { echo "NO CKPT, abort eval"; exit 1; }

PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl16_l4like_640_rgbonly_val_
for lv in day night50 night25 night12 night05 night01 night005 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== rgbonly eval $lv $(date) ===" >> "$W/vitl16_rgbonly_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitl16_rgbonly_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/vitl16_rgbonly_run.log"
done
echo "ALL DONE RGBONLY $(date)" >> "$W/vitl16_rgbonly_run.log"
