#!/usr/bin/env bash
# dfine_x_obj2coco 微调的 GPU 排队启动器:
# 等待空闲显存 >= 12 GiB 后执行 run_dfine_x_obj2coco_ft.sh (训练+10档评测)
set -u
NEED_MB=12000
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "$(date) free=${FREE}MiB need=${NEED_MB}MiB" >> /home/constant/CodeBuddy/Dev/AIC/logs/dfine_x_ft_queue.log
  if [ "$FREE" -ge "$NEED_MB" ]; then
    echo "$(date) GPU ready, launching" >> /home/constant/CodeBuddy/Dev/AIC/logs/dfine_x_ft_queue.log
    break
  fi
  sleep 300
done
bash /home/constant/CodeBuddy/Dev/AIC/scripts/train/run_dfine_x_obj2coco_ft.sh
