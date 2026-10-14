#!/usr/bin/env bash
# Auto-queue for Task 4: waits for the heat-IR 5ch A/B training to free the GPU,
# then trains a 4ch (RGB+IR) model on AIC and evaluates it on LLVIP (OOD test).
set -u
LOG=/home/constant/CodeBuddy/Dev/AIC/logs/task4_queue.log
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG_4CH=configs/deimv2/deimv2_dinov3_l_aic_diffuse_4ch.yml
CFG_LLVIP=configs/deimv2/deimv2_dinov3_l_aic_diffuse_4ch_llvip.yml
CKPT_DIR=$DEIM/outputs/deimv2_dinov3_l_aic_diffuse_4ch
HEAT_MATCH="deimv2_dinov3_l_aic_rdttrack_heat.yml"

ts(){ date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] queue started; waiting for heat training ($HEAT_MATCH) to finish..."

# Wait until the heat training process is gone (poll every 60s).
while pgrep -f "$HEAT_MATCH" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(ts)] heat training process gone. Proceeding to 4ch training."

# ---- Step 1: train 4ch (RGB+IR) on AIC ----
echo "[$(ts)] === START 4ch training ($CFG_4CH) ==="
cd "$DEIM"
$PY train.py -c "$CFG_4CH" --use-amp
echo "[$(ts)] 4ch training exited with code $?"

# ---- Step 2: pick the best available checkpoint ----
if [ -f "$CKPT_DIR/best_stg2.pth" ]; then CKPT="$CKPT_DIR/best_stg2.pth"
elif [ -f "$CKPT_DIR/best_stg1.pth" ]; then CKPT="$CKPT_DIR/best_stg1.pth"
elif [ -f "$CKPT_DIR/last.pth" ]; then CKPT="$CKPT_DIR/last.pth"
else CKPT=""; fi

if [ -z "$CKPT" ]; then
  echo "[$(ts)] NO checkpoint found in $CKPT_DIR -> cannot eval LLVIP. ABORT."
  exit 2
fi
echo "[$(ts)] using checkpoint: $CKPT"

# ---- Step 3: evaluate the 4ch model on LLVIP (whole set, OOD) ----
echo "[$(ts)] === START LLVIP eval ($CFG_LLVIP) ==="
cd "$DEIM"
$PY train.py -c "$CFG_LLVIP" -r "$CKPT" --use-amp --test-only
echo "[$(ts)] === LLVIP eval finished (exit $?) ==="
echo "[$(ts)] TASK4 QUEUE COMPLETE"
