#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Launch pure-IR 2ch diagnostic directly (queue was deadlocked on its own pgrep match).
VENDOR=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
LOG=/home/constant/CodeBuddy/Dev/AIC/pureir2ch.log
cd "$VENDOR"
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.launch --nproc_per_node=1 --master_port=29590 \
  train.py -c configs/deimv2/deimv2_dinov3_l_aic_pureir2ch.yml --use-amp >> "$LOG" 2>&1
echo "[$(date)] pure-ir2ch exit=$?"
