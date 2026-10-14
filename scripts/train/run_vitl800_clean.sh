#!/bin/bash
# vitl800 CLEAN: 干净单变量测试 (ViT-L @800, 标准增强, RGB lr 1e-4)
# 基准 = midboost_combo (ViT-S/640, mAP 0.5222). 本配置见对应 *_vitl800_clean.yml 注释.
# 与 vits800_clean 的唯一变量 = backbone 大小 (ViT-L 24层 vs ViT-S 12层), 其余严格一致.
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
CFG=configs/deimv2/deimv2_dinov3_l_aic_vitl800_clean.yml
SRC=ckpts/dinov3/vitl16_rdttrack_init.pth
OUT=outputs/deimv2_dinov3_l_aic_vitl800_clean
LV="day night50 night25 night12 night05 night01 night005 night002 night001 night0005"

echo "===== START VITL800_CLEAN $(date) =====" >> "$W/vitl800_clean_run.log"

# 排队: 等 GPU 空闲(可用 >= 12GB)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vitl800_clean_run.log"
  sleep 60
done
echo "GPU free, launching train $(date)" >> "$W/vitl800_clean_run.log"

# 干净目录无 last.pth -> 必走 -t 热启动 (官方 ViT-L + 继承 enc/dec, 融合按 shape 重初始化)
if [ -f "$OUT/last.pth" ]; then
  RESUME_ARG="-r $OUT/last.pth"
  echo "RESUMING from $OUT/last.pth ($(date))" >> "$W/vitl800_clean_run.log"
else
  RESUME_ARG="-t $SRC"
  echo "FRESH tuning from $SRC ($(date))" >> "$W/vitl800_clean_run.log"
fi
$PY train.py -c "$CFG" $RESUME_ARG --use-amp > "$W/vitl800_clean_train.log" 2>&1
TRAIN_RC=$?
echo "TRAIN DONE exit=$TRAIN_RC $(date)" >> "$W/vitl800_clean_run.log"
if [ "$TRAIN_RC" -ne 0 ]; then
  echo "TRAIN FAILED rc=$TRAIN_RC —— 不跑 eval (重跑本脚本会从 last.pth 续训)" >> "$W/vitl800_clean_run.log"
  exit "$TRAIN_RC"
fi

CKPT=$(ls -t "$OUT"/*.pth 2>/dev/null | head -1)
echo "VITL800_CLEAN CKPT=$CKPT" >> "$W/vitl800_clean_run.log"
if [ -z "$CKPT" ]; then echo "NO CKPT, abort eval"; exit 1; fi

for lv in $LV; do
  cfg=${PREF}${lv}.yml; [ "$lv" = "day" ] && cfg=${PREF}orig.yml
  echo "=== vitl800_clean eval $lv $(date) ===" >> "$W/vitl800_clean_run.log"
  $PY train.py -c "$cfg" -r "$CKPT" --use-amp --test-only > "$W/vitl800_clean_eval_${lv}.log" 2>&1
done
echo "EVAL DONE $(date)" >> "$W/vitl800_clean_run.log"
echo "ALL DONE VITL800_CLEAN $(date)" >> "$W/vitl800_clean_run.log"
