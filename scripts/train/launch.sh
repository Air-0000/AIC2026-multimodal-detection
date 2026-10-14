#!/usr/bin/env bash
set -e

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}"
mkdir -p "${AIC_ROOT}/logs"

pkill -f 'train.py' || true
sleep 1

nohup bash "${AIC_ROOT}/scripts/train/run_sovereign_distill_rtx6000.sh" > "${AIC_ROOT}/logs/run_sovereign_distill_rtx6000.log" 2>&1 &
echo "Training launched with PID $!"
sleep 2
ps aux | grep train.py
