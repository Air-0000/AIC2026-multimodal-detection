#!/bin/bash
# l4u3 (ViT-S/640) NIGHT INFERENCE + TTA — the TTA counterpart of the no-TTA
# table produced by run_l4u3.sh (train.py --test-only on the same _x_deepir_val_*
# configs), so vitl800-TTA vs l4u3-TTA is an apples-to-apples comparison.
#
# TTA = multi-scale(512,640) x h-flip, merged with the project's WBF recipe
# (scripts/eval/predict_mm_tta.py). 512/640 mirrors the 0.8x/1.0x ratio used for
# the 800-native vitl800 run (640,800).
# Must run on the HOST (needs CUDA). Relaunches vitl800_clean when finished.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
CKPT=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3/best_stg1.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_
LV="orig night0005 night001 night002 night005 night01 night05 night12 night25 night50"
SCALES="512,640"
SUMMARY=/home/constant/CodeBuddy/Dev/AIC/outputs/l4u3_best_night_infer_tta.txt

echo "===== L4U3 NIGHT TTA ON BEST ($CKPT) scales=$SCALES flip=on $(date) =====" >> "$W/l4u3_night_tta.log"

# wait for GPU free (>= 6GB)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 6000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/l4u3_night_tta.log"
  sleep 30
done

echo "l4u3 best_stg1.pth — night inference WITH TTA (scales=$SCALES + hflip, WBF-merged)" > "$SUMMARY"
echo "" >> "$SUMMARY"
printf '%-10s %12s %10s\n' cond 'AP@.5:.95' 'AP50' >> "$SUMMARY"

for lv in $LV; do
  cfg=${PREF}${lv}.yml
  echo "=== l4u3 TTA infer $lv $(date) ===" >> "$W/l4u3_night_tta.log"
  OUT=$($PY /home/constant/CodeBuddy/Dev/AIC/scripts/tta_eval.py -c "$cfg" -r "$CKPT" \
        --use-amp --scales "$SCALES" --flip 2>&1 | tee "$W/l4u3_night_tta_${lv}.log")
  AP=$(echo "$OUT" | grep -oE 'RESULT_AP [0-9.]+' | awk '{print $2}')
  AP50=$(echo "$OUT" | grep -oE 'RESULT_AP50 [0-9.]+' | awk '{print $2}')
  printf '%-10s %12s %10s\n' "$lv" "${AP:-ERR}" "${AP50:-ERR}" >> "$SUMMARY"
  echo "l4u3 TTA $lv AP=$AP AP50=$AP50" >> "$W/l4u3_night_tta.log"
done

echo "L4U3 TTA NIGHT INFER DONE $(date)" >> "$W/l4u3_night_tta.log"
cat "$SUMMARY"

# resume the main clean-training experiment
echo "resuming vitl800_clean training $(date)" >> "$W/l4u3_night_tta.log"
nohup bash /home/constant/CodeBuddy/Dev/AIC/scripts/train/run_vitl800_clean.sh >/dev/null 2>&1 &
echo "relaunched vitl800_clean (pid $!) $(date)" >> "$W/l4u3_night_tta.log"
