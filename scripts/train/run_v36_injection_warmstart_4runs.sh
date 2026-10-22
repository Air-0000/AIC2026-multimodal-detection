#!/bin/bash
set -euo pipefail
P=/home/adlab4/ssd3/addata/AIC_V36_ABLATION_20261018
PY=/home/adlab4/ssd3/addata/aic-jiao/DFINE-Reference-AIC/env/bin/python
cd "$P/vendor/DEIMv2"
export PYTHONPATH="$P/vendor/DEIMv2:$P/scripts/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
CFG=configs/deimv2/v36_injection_warmstart_20261019
LOG=$P/logs
mkdir -p "$LOG"
echo "runner start $(date -Is) on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
# train.py merges every non-None CLI default into the config, so --seed MUST be passed
# explicitly: its default 0 would silently override the config's seed 42/43.
for run in seed42_control seed42_p3p4 seed43_control seed43_p3p4; do
  SEED=$(printf '%s' "$run" | sed -n 's/^seed\([0-9][0-9]\)_.*/\1/p')
  OUT=$P/work_dir/audits/v36_injection_warmstart_20261019/training/$run
  if [ -f "$OUT/checkpoint0007.pth" ]; then echo "SKIP $run: checkpoint0007 already exists"; continue; fi
  TS=$(date +%Y%m%d_%H%M)
  echo "=== START $run seed=$SEED $TS ==="
  "$PY" train.py -c "$CFG/$run.yml" --seed "$SEED" > "$LOG/v36_inj_${run}_${TS}.log" 2>&1
  echo "=== END $run $(date -Is) ==="
done
echo "ALL_FOUR_DONE $(date -Is)"
