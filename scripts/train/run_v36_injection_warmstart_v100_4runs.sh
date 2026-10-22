#!/bin/bash
set -euo pipefail
R=/home/ailab/ssd/bigdata/hejianzhong/AIC_V29
NR=/home/ailab/ssd/bigdata/hejianzhong/AIC_V36_INJ_20261019
PY=/home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python
cd "$NR/vendor_DEIMv2"
export PYTHONPATH="$NR/vendor_DEIMv2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
CFG=configs/deimv2/v36_injection_warmstart_20261019
echo "runner start $(date -Is) on $(hostname) GPU=$CUDA_VISIBLE_DEVICES torch=$($PY -c 'import torch;print(torch.__version__)')"
for run in seed42_control seed42_p3p4 seed43_control seed43_p3p4; do
  SEED=$(printf '%s' "$run" | sed -n 's/^seed\([0-9][0-9]\)_.*/\1/p')
  OUT=$NR/training/$run
  if [ -f "$OUT/checkpoint0007.pth" ]; then echo "SKIP $run: checkpoint0007 exists"; continue; fi
  TS=$(date +%Y%m%d_%H%M)
  echo "=== START $run seed=$SEED $TS ==="
  "$PY" train.py -c "$CFG/$run.yml" --seed "$SEED" > "$NR/logs/v36inj_${run}_${TS}.log" 2>&1
  echo "=== END $run $(date -Is) ==="
done
echo "ALL_FOUR_DONE $(date -Is)"
