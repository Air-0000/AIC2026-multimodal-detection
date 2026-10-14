#!/bin/bash
# =============================================================================
# Stage C / l4like 变体 —— 对齐 midboost_l4 口径 (unfreeze 等比 4 + RGB lr 3e-5 + night 均匀采样)
#
# 相对 vitl800_scratch_stageC_fuse 的两个变量:
#   1. llrd_top_lr 2e-5 -> 3e-5, llrd_decay 0.85 -> 1.0 (unfreeze 6->4)
#      (RGB 实际 lr 3e-5 = l4 口径; 旧 run 实际只有 8.87e-6)
#   2. RandomNight5ch: L2 bands -> level [0.005, 0.45] 均匀 (= l4 原样)
# 起点: Stage B last.pth (与旧 C 完全同起点 => 干净配对对照)
# =============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageC_l4like.yml
OUT=outputs/vitl800_scratch_stageC_l4like
STAGE_B_OUT=outputs/vitl800_scratch_stageB_mod
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START STAGE_C_L4LIKE $(date) =====" >> "$W/stageC_l4like_run.log"

# 排队: 等 GPU 可用 >= 12GB
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/stageC_l4like_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/stageC_l4like_run.log"

if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/stageC_l4like_run.log"
else
  SRC_B=$(ls -t "$STAGE_B_OUT"/last.pth "$STAGE_B_OUT"/best_stg1.pth 2>/dev/null | head -1)
  if [ -z "$SRC_B" ]; then
    echo "NO STAGE_B CKPT, abort" | tee -a "$W/stageC_l4like_run.log"
    exit 1
  fi
  RESUME_ARG="-t $SRC_B"
  echo "FRESH from Stage B: $SRC_B ($(date))" >> "$W/stageC_l4like_run.log"
fi

$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/stageC_l4like_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/stageC_l4like_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC (重跑本脚本会从 last.pth 续训)" >> "$W/stageC_l4like_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "STAGE_C_L4LIKE CKPT=$CKPT" >> "$W/stageC_l4like_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== l4like eval $lv $(date) ===" >> "$W/stageC_l4like_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/stageC_l4like_eval_${lv}.log" 2>&1
done
echo "ALL DONE STAGE_C_L4LIKE $(date)" >> "$W/stageC_l4like_run.log"
