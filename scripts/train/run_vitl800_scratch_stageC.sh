#!/bin/bash
# =============================================================================
# Stage C / 联合微调 —— 三阶段流程第三步 (出最终成品)
#
# 起点: Stage B 的 last.pth (完整走完 lr 调度的最终权重, 而非 val 峰值 best)
#   unfreeze_rgb_stages: 6 -> 放开 RGB 末 6 个 block (18..23)
#   llrd_top_lr: 2.0e-5  -> 主干只做小幅调整, 避免把 Stage A/B 的成果洗掉
#   ir/depth_drop_prob: 0.15 -> 保持缺模态鲁棒性
# =============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_scratch_stageC_fuse.yml
OUT=outputs/vitl800_scratch_stageC_fuse
STAGE_B_OUT=outputs/vitl800_scratch_stageB_mod
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START STAGE_C $(date) =====" >> "$W/stageC_fuse_run.log"

# 排队: 等 GPU 可用 >= 12GB
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/stageC_fuse_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/stageC_fuse_run.log"

if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/stageC_fuse_run.log"
else
  SRC_B=$(ls -t "$STAGE_B_OUT"/last.pth "$STAGE_B_OUT"/best_stg1.pth 2>/dev/null | head -1)
  if [ -z "$SRC_B" ]; then
    echo "NO STAGE_B CKPT, abort" | tee -a "$W/stageC_fuse_run.log"
    exit 1
  fi
  RESUME_ARG="-t $SRC_B"
  echo "FRESH from Stage B: $SRC_B ($(date))" >> "$W/stageC_fuse_run.log"
fi

$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/stageC_fuse_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/stageC_fuse_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC (重跑本脚本会从 last.pth 续训)" >> "$W/stageC_fuse_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "STAGE_C CKPT=$CKPT" >> "$W/stageC_fuse_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== stageC eval $lv $(date) ===" >> "$W/stageC_fuse_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/stageC_fuse_eval_${lv}.log" 2>&1
done
echo "ALL DONE STAGE_C $(date)" >> "$W/stageC_fuse_run.log"
