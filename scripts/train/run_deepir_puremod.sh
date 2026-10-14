#!/bin/bash
# 「纯单模态」消融：检验模型是否只是随亮度从 RGB 硬切换到 IR，而非真正融合。
#
# 同一权重 (_x_deepir/best_stg2.pth) + 同一套 val 配置，只改推理输入通道：
#   full       无遮挡                                  -> 已有 deepir_eval_{orig,night*}
#   rgb        ABLATE=heat,rawir,depth   只留 RGB       -> 已有 deepir_keep_rgb_*
#   ir         ABLATE=rgb,depth          只留 heat+rawIR (本脚本新增)
#   heat       ABLATE=rgb,rawir,depth    只留 heat      (本脚本新增)
#   rawir      ABLATE=rgb,heat,depth     只留 rawIR     (本脚本新增)
#   depth      ABLATE=rgb,heat,rawir     只留 depth     (本脚本新增)
#
# ABLATE=rgb 会把通道 0..2 一起置为 Normalize 均值(0.5) -> 归一化后恒为常量,
# RGB 分支拿不到任何信息(见 engine/data/dataset/coco_dataset_mm.py 的 ABLATE 段)。
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir/best_stg2.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
W=/home/constant/CodeBuddy/Dev/AIC/logs
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

# nightify_aic 在 scripts/lib, engine.data.transforms 会 import 它
export PYTHONPATH=/home/constant/CodeBuddy/Dev/AIC/scripts/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 等等 GPU 空出来 (至少 5GB)
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE:-0}" -ge 5000 ]; then break; fi
  echo "waiting for GPU, free=${FREE}MiB" >> "$W/deepir_only_run.log"
  sleep 30
done

declare -A M=(
  [ir]="rgb,depth"
  [heat]="rgb,rawir,depth"
  [rawir]="rgb,heat,depth"
  [depth]="rgb,heat,rawir"
)

echo "=== START ALL $(date) ===" >> "$W/deepir_only_run.log"
for k in ir heat rawir depth; do
  envv=${M[$k]}
  for lv in $LV; do
    cfg=${PREF}${lv}.yml
    [ "$lv" = "day" ] && cfg=${PREF}orig.yml
    echo "=== START only_$k $lv  ABLATE=$envv  $(date) ===" >> "$W/deepir_only_run.log"
    ABLATE=$envv $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only \
      > "$W/deepir_only_${k}_${lv}.log" 2>&1
    echo "=== DONE only_$k $lv exit=$? $(date) ===" >> "$W/deepir_only_run.log"
  done
done
echo "=== ALL DONE $(date) ===" >> "$W/deepir_only_run.log"
