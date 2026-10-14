#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# 4ch (DiffFuse) IR ablation on AIC val -- same protocol as the 5ch ablation.
set -u
cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CKPT=outputs/deimv2_dinov3_l_aic_diffuse_4ch/best_stg2.pth

echo "[$(date +%H:%M:%S)] start 4ch baseline" >> /tmp/4ch_ablate_progress.log
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_diffuse_4ch.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/4ch_val_base.log 2>&1

echo "[$(date +%H:%M:%S)] start 4ch no-IR" >> /tmp/4ch_ablate_progress.log
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_diffuse_4ch_val_noir.yml \
    -r "$CKPT" --use-amp --test-only > /tmp/4ch_val_noir.log 2>&1

echo "[$(date +%H:%M:%S)] 4ch ALL DONE" >> /tmp/4ch_ablate_progress.log
