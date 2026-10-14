#!/usr/bin/env bash
# 排队器: 等待 GPU 空闲(<2000 MiB)后启动 RS 训练, 最长等 10h。
# 用途: mal2 训练+评测结束后自动接力, 避免显存撞车。
set -u
W=/home/constant/CodeBuddy/Dev/AIC/logs
DEADLINE=$(( $(date +%s) + 36000 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$(date '+%m-%d %H:%M:%S') gpu_used=${USED}MiB" >> "$W/rs_queue.log"
  if [ "$USED" -lt 2000 ]; then
    echo "$(date '+%m-%d %H:%M:%S') GPU free, launching RS" >> "$W/rs_queue.log"
    exec bash /home/constant/CodeBuddy/Dev/AIC/scripts/train/run_dfine_x_rs.sh
  fi
  sleep 180
done
echo "$(date '+%m-%d %H:%M:%S') TIMEOUT 10h, abort" >> "$W/rs_queue.log"
