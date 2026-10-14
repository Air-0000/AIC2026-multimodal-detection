#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Fine-tune the nightcut_inorm (AIC) model on BDD daytime, then eval on real BDD night.
# Fixes the step-1 AssertionError crash: tuning from best_stg1.pth (valid init) +
# robust box_ops (NaN/degenerate boxes no longer abort training).
set -e
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime.yml
TUNE=./outputs/deimv2_dinov3_l_aic_nightcut_inorm/best_stg1.pth
OUT=./outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime
LOG=/home/constant/CodeBuddy/Dev/AIC/ft_dndt_day4.log

# wait until GPU is free (<3000 MiB) so we don't OOM like before
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ -n "$used" ] && [ "$used" -lt 3000 ]; then break; fi
  sleep 60
done

echo "[launcher] GPU free (${used} MiB), start tuning train $(date)" | tee -a "$LOG"
$PY train.py -c $CFG -t $TUNE 2>&1 | tee -a "$LOG"

echo "[launcher] train done $(date), start real BDD-night eval" | tee -a "$LOG"
$PY train.py -c $CFG -r $OUT/best_stg1.pth --test-only 2>&1 | tee -a "$LOG"
echo "[launcher] all done $(date)" | tee -a "$LOG"
