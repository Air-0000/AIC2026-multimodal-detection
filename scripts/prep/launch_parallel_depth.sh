#!/usr/bin/env bash
set -e

AIC_ROOT="/root/autodl-tmp/AIC"
cd "${AIC_ROOT}"
mkdir -p "${AIC_ROOT}/logs"

echo "Launching 4 parallel depth generation workers on RTX PRO 6000..."
for i in 0 1 2 3; do
    nohup python3 scripts/prep/prep_p0_depth_unified.py --limit 0 --stride 4 --offset $i > logs/prep_p0_depth_$i.log 2>&1 &
    echo "Worker $i launched (PID: $!)."
done

sleep 2
ps aux | grep prep_p0
