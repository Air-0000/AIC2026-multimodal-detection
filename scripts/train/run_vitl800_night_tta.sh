#!/bin/bash
# vitl800 NIGHT INFERENCE + TTA on the POLLUTED best_stg1.pth (same ckpt as the
# no-TTA table in outputs/metrics/vitl800_best_night_infer.txt) so it's a clean
# TTA-vs-noTTA comparison. TTA = multi-scale(640,800) x h-flip, merged via NMS.
# Must run on the HOST (needs CUDA). Pauses the clean training first (the driver
# is launched after vitl800_clean is killed), and relaunches vitl800_clean at the end.
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2

PY=/home/constant/miniconda3/envs/deimv2/bin/python
W=/home/constant/CodeBuddy/Dev/AIC/logs
BEST=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug/best_stg1.pth
PREF=configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug_val_
LV="orig night0005 night001 night002 night005 night01 night05 night12 night25 night50"
SCALES="640,800"
SUMMARY=/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/vitl800_best_night_infer_tta.txt

echo "===== VITL800 NIGHT TTA ON BEST ($BEST) scales=$SCALES flip=on $(date) =====" >> "$W/vitl800_best_night_tta.log"

# wait for GPU free (>= 6GB)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 6000 ]; then break; fi
  echo "queued: free=${FREE}MiB $(date)" >> "$W/vitl800_best_night_tta.log"
  sleep 30
done

echo "vitl800 (polluted) best_stg1.pth — night inference WITH TTA (scales=$SCALES + hflip, NMS-merged)" > "$SUMMARY"
echo "" >> "$SUMMARY"
printf '%-10s %12s %10s\n' cond 'AP@.5:.95' 'AP50' >> "$SUMMARY"

for lv in $LV; do
  cfg=${PREF}${lv}.yml
  echo "=== TTA infer $lv $(date) ===" >> "$W/vitl800_best_night_tta.log"
  OUT=$($PY /home/constant/CodeBuddy/Dev/AIC/scripts/tta_eval.py -c "$cfg" -r "$BEST" --use-amp --scales "$SCALES" --flip 2>&1 | tee "$W/vitl800_best_night_tta_${lv}.log")
  AP=$(echo "$OUT" | grep -oE 'RESULT_AP [0-9.]+' | awk '{print $2}')
  AP50=$(echo "$OUT" | grep -oE 'RESULT_AP50 [0-9.]+' | awk '{print $2}')
  printf '%-10s %12s %10s\n' "$lv" "${AP:-ERR}" "${AP50:-ERR}" >> "$SUMMARY"
  echo "TTA $lv AP=$AP AP50=$AP50" >> "$W/vitl800_best_night_tta.log"
done

echo "TTA NIGHT INFER DONE $(date)" >> "$W/vitl800_best_night_tta.log"
cat "$SUMMARY"

# resume the main clean-training experiment (resumes from last.pth)
echo "resuming vitl800_clean training $(date)" >> "$W/vitl800_best_night_tta.log"
nohup bash /home/constant/CodeBuddy/Dev/AIC/scripts/train/run_vitl800_clean.sh >/dev/null 2>&1 &
echo "relaunched vitl800_clean (pid $!) $(date)" >> "$W/vitl800_best_night_tta.log"
