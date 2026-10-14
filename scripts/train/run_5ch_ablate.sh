#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# 5ch modality ablation on AIC val (real depth available here).
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth

for n in "" "_val_noir" "_val_nodepth" "_val_noaux"; do
  echo "[$(date +%H:%M:%S)] start ${n:-baseline}" >> /tmp/5ch_ablate_progress.log
  $PY train.py -c "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat${n}.yml" \
      -r "$CKPT" --use-amp --test-only > "/tmp/5ch_val${n}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done ${n:-baseline}" >> /tmp/5ch_ablate_progress.log
done

# LLVIP: IR blacked out (depth is null there anyway)
echo "[$(date +%H:%M:%S)] start llvip_noir" >> /tmp/5ch_ablate_progress.log
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_llvip_noir.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/5ch_llvip_noir.log 2>&1
echo "[$(date +%H:%M:%S)] ALL DONE" >> /tmp/5ch_ablate_progress.log
