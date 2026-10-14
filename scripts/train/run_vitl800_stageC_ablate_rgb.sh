#!/bin/bash
# 诊断: Stage C 极暗端崩溃是 "IR 通路死" 还是 "黑 RGB 垃圾特征拖死"?
# ABLATE=rgb -> RGB 三通道填 Normalize 均值 0.5 (归一化后=0, 中性关闭), IR/heat/depth 完好.
# 对照: 已有全模态 night 表 (checkpoint0039 同 ckpt 同路径).
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/vitl800_scratch_stageC_fuse/last.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="night005 night01 night002 night0005"
echo "=== START ablate=rgb $(date) ===" >> "$W/stageC_ablate_rgb_run.log"
for lv in $LV; do
  cfg=${PREF}${lv}.yml
  echo "=== START $lv $(date) ===" >> "$W/stageC_ablate_rgb_run.log"
  ABLATE=rgb $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/stageC_ablate_rgb_${lv}.log" 2>&1
  echo "=== DONE $lv exit=$? $(date) ===" >> "$W/stageC_ablate_rgb_run.log"
done
echo "=== ALL DONE $(date) ===" >> "$W/stageC_ablate_rgb_run.log"
