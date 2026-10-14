#!/usr/bin/env bash
# vitl16 RGB-only · 官方 DEIMv2-X 配方移植版: 单段训练 + 10 档 night 评测
# 配置: deimv2_dinov3_l_aic_vitl16_640_rgbonly_xstyle.yml
# 要点: 全解冻 ViT-L @平1e-5 (官方X), 头 5e-4, 58ep/flat29/no_aug8
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2:/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl16_640_rgbonly_xstyle.yml
OUT=./outputs/vitl16_640_rgbonly_xstyle

echo "===== START XSTYLE $(date) =====" >> "$W/vitl16_xstyle_run.log"
$PY train.py -c "$CFG" -t ckpts/dinov3/vitl16_rdttrack_scratch_full.pth --use-amp > "$W/vitl16_xstyle_train.log" 2>&1
echo "train rc=$?" >> "$W/vitl16_xstyle_run.log"

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "XSTYLE CKPT=$CKPT" >> "$W/vitl16_xstyle_run.log"
[ -z "$CKPT" ] && { echo "NO CKPT, abort eval"; exit 1; }

PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl16_640_rgbonly_xstyle_val_
for lv in day night50 night25 night12 night05 night01 night005 night002 night001 night0005; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== xstyle eval $lv $(date) ===" >> "$W/vitl16_xstyle_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitl16_xstyle_eval_${lv}.log" 2>&1
  echo "rc=$? $lv" >> "$W/vitl16_xstyle_run.log"
done
echo "ALL DONE XSTYLE $(date)" >> "$W/vitl16_xstyle_run.log"
