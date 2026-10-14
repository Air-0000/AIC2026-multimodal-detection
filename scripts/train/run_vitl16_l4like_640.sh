#!/bin/bash
# =============================================================================
# vitl16 l4like @640 -- 单段 from-scratch (无三阶段)
# 模板 = midboost_l4; 只改 backbone (vits16plus -> vitl16, interaction [11,17,23],
# aux 显式钉死 vits16plus)。起点 = scratch_full (官方三主干 + 随机头/融合)。
# 训完自动跑 10 档 night 评测 (640 口径, 与 l4 的评测协议一致)。
# =============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl16_l4like_640.yml
OUT=outputs/vitl16_l4like_640_scratch
INIT=ckpts/dinov3/vitl16_rdttrack_scratch_full.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START VITL16_L4LIKE_640 $(date) =====" >> "$W/vitl16_l4like_run.log"

[ -f "$INIT" ] || { echo "NO INIT CKPT $INIT, abort" | tee -a "$W/vitl16_l4like_run.log"; exit 1; }

# 排队: 等 GPU 可用 >= 14GB (ViT-L @640 micro5, 峰值未知, 留余量)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 14000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vitl16_l4like_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/vitl16_l4like_run.log"

if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/vitl16_l4like_run.log"
else
  RESUME_ARG="-t $INIT"
  echo "FRESH from scratch_full: $INIT ($(date))" >> "$W/vitl16_l4like_run.log"
fi

$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/vitl16_l4like_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/vitl16_l4like_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC (重跑本脚本会从 last.pth 续训)" >> "$W/vitl16_l4like_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "VITL16_L4LIKE CKPT=$CKPT" >> "$W/vitl16_l4like_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== l4like640 eval $lv $(date) ===" >> "$W/vitl16_l4like_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitl16_l4like_eval_${lv}.log" 2>&1
done
echo "ALL DONE VITL16_L4LIKE_640 $(date)" >> "$W/vitl16_l4like_run.log"
