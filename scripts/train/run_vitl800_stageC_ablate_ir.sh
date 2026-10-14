#!/bin/bash
# 补充诊断: 反向消融 ABLATE=rawir,heat,depth (IR/heat/depth 中性关闭, 只留 RGB)
# 目的: 量化 IR 流在 RGB 活着时的真实贡献 (full - IRoff), 并核对 n05 的 +14pt 来自 IR 还是我们 RGB 更强.
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/vitl800_scratch_stageC_fuse/last.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="orig night50 night05 night01"
echo "=== START ablate=rawir,heat,depth $(date) ===" >> "$W/stageC_ablate_ir_run.log"
for lv in $LV; do
  if [ "$lv" = "orig" ]; then cfg=${PREF}orig.yml; tag=day; else cfg=${PREF}${lv}.yml; tag=$lv; fi
  echo "=== START $tag $(date) ===" >> "$W/stageC_ablate_ir_run.log"
  ABLATE=rawir,heat,depth $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/stageC_ablate_ir_${tag}.log" 2>&1
  echo "=== DONE $tag exit=$? $(date) ===" >> "$W/stageC_ablate_ir_run.log"
done
echo "=== ALL DONE $(date) ===" >> "$W/stageC_ablate_ir_run.log"
