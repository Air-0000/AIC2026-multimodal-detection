#!/usr/bin/env bash
# Probe: does best_stg1.pth really reproduce the logged best val mAP?
# Run on the GPU host.
set -u
ROOT=/home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
export PYTHONPATH="$ROOT/scripts/lib:${PYTHONPATH:-}"
cd "$ROOT/vendor/DEIMv2"

CFG=configs/deimv2/deimv2_dinov3_l_aic_nightsim_inorm_rnight.yml
DIR=outputs/deimv2_dinov3_l_aic_nightsim_inorm_rnight

for ck in best_stg1 checkpoint0039 checkpoint0034 checkpoint0029 last; do
  f="$DIR/$ck.pth"
  [ -f "$f" ] || { echo "$ck  (missing)"; continue; }
  ap=$($PY train.py -c "$CFG" -r "$f" --use-amp --test-only \
        --output-dir "$ROOT/work_dir/_probe_rgb" 2>&1 \
        | grep -m1 'IoU=0.50:0.95' | awk '{print $NF}')
  echo "$ck  AP=$ap"
done
echo PROBE_DONE
