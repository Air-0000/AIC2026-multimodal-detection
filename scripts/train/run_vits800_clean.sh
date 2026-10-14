#!/bin/bash
# vits800 CLEAN: vitl800_clean 的受控 ViT-S 对照 (单变量 = backbone 大小 @800)
# 与 vitl800_clean 唯一变量 = backbone 容量 (ViT-S 12层 vs ViT-L 24层), 其余严格一致.
# 增强/lr 同 midboost_combo; warm-start 继承 midboost_combo best_stg1.pth (非 l4u3 微调成品).
# 对比含义:
#   vits800_clean vs midboost_combo = 纯分辨率效应 (640->800)
#   vits800_clean vs vitl800_clean  = 纯 backbone 效应 (ViT-S->ViT-L @800)
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vits800_rdttrack_deepir_aug_val_
CFG=configs/deimv2/deimv2_dinov3_l_aic_vits800_clean.yml
SRC=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo/best_stg1.pth
OUT=outputs/deimv2_dinov3_l_aic_vits800_clean
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START VITS800_CLEAN $(date) =====" >> "$W/vits800_clean_run.log"

# 排队: 严格等 vitl800_clean 完全释放(含 11 档 eval)后再接力, 避免争抢 GPU.
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 14000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vits800_clean_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/vits800_clean_run.log"

# 干净目录无 last.pth -> 必走 -t 热启动 (继承 midboost_combo 已训 ViT-S 检测器)
if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/vits800_clean_run.log"
else
  RESUME_ARG="-t $SRC"
  echo "FRESH tuning from $SRC ($(date))" >> "$W/vits800_clean_run.log"
fi
$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/vits800_clean_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/vits800_clean_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC —— 不跑 eval (重跑本脚本会从 last.pth 续训)" >> "$W/vits800_clean_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "VITS800_CLEAN CKPT=$CKPT" >> "$W/vits800_clean_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== vits800_clean eval $lv $(date) ===" >> "$W/vits800_clean_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vits800_clean_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/vits800_clean_run.log"
echo "ALL DONE VITS800_CLEAN $(date)" >> "$W/vits800_clean_run.log"
