#!/usr/bin/env bash
# Evaluate the BiDirPaper 4ch (RGB+IR-heat) night model on 9 darkness levels.
# Run AFTER training finishes (needs outputs/.../bidir_paper_4ch/best_stg2.pth).
set -u
ROOT=/home/constant/CodeBuddy/Dev/AIC
DEIM=$ROOT/vendor/DEIMv2
CKPT=$DEIM/outputs/deimv2_dinov3_l_aic_bidir_paper_4ch/best_stg2.pth
PYTHON=/home/constant/miniconda3/envs/deimv2/bin/python
LOGDIR=/tmp/night_full
mkdir -p "$LOGDIR"

LEVELS=(night50 night25 night12 night05 night01 night005 night002 night001 night0005)

if [ ! -f "$CKPT" ]; then
  echo "CHECKPOINT MISSING: $CKPT (training not finished?)"
  exit 1
fi

cd "$DEIM" || exit 1
for lv in "${LEVELS[@]}"; do
  cfg="configs/deimv2/deimv2_dinov3_l_aic_bidir_paper_4ch_val_${lv}.yml"
  out="$LOGDIR/bidir_paper_${lv}.log"
  # idempotent: skip if a complete eval log already exists
  if [ -f "$out" ] && grep -q "DONE (t=" "$out"; then
    echo "===== skip $lv (already done) ====="
    continue
  fi
  echo "===== eval $lv ====="
  "$PYTHON" train.py -c "$cfg" -r "$CKPT" --use-amp --test-only 2>&1 | tee "$out"
done
echo "ALL DONE -> logs in $LOGDIR/bidir_paper_*.log"
