#!/bin/bash
# 纯 IR 3ch 对照实验 (from scratch, 与 pureir2ch 同配方)
#   run 1: pureir3ch       vs pureir2ch      (0.2449)
#   run 2: pureir3ch_deep  vs pureir2ch_deep (0.2983)
# 唯一改动: IR 分支 2ch -> 3ch [heat, raw, deleak]
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENDOR=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
cd "$VENDOR"

# 排队: 等 GPU 空闲 (可用 >= 12GB)
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  FREE=$((TOTAL - USED))
  if [ "$FREE" -ge 12000 ]; then break; fi
  sleep 60
done

echo "[$(date)] START pureir3ch"
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.launch --nproc_per_node=1 --master_port=29591 \
  train.py -c configs/deimv2/deimv2_dinov3_l_aic_pureir3ch.yml --use-amp \
  >> /home/constant/CodeBuddy/Dev/AIC/pureir3ch.log 2>&1
echo "[$(date)] pureir3ch exit=$?"

echo "[$(date)] START pureir3ch_deep"
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.launch --nproc_per_node=1 --master_port=29592 \
  train.py -c configs/deimv2/deimv2_dinov3_l_aic_pureir3ch_deep.yml --use-amp \
  >> /home/constant/CodeBuddy/Dev/AIC/pureir3ch_deep.log 2>&1
echo "[$(date)] pureir3ch_deep exit=$?"
