#!/bin/bash
# 严格单变量对照: pureir2ch + train bs 12  (基线 pureir2ch bs8 = 0.2449)
# 目的: 把 pureir3ch (bs12, 0.3369) 的增益拆成「通道数 2->3」与「bs 8->12」两部分。
# 与基线的唯一差别: train_dataloader.total_batch_size 8 -> 12
# (epoches / lr / warmup_iter / val bs / 数据 / 增强 全部不动)
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=deimv2_dinov3_l_aic_pureir2ch_bs12.yml
LOG=$AIC/pureir2ch_bs12.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] START pureir2ch_bs12"
rm -f /dev/shm/torch_* 2>/dev/null
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.launch --nproc_per_node=1 --master_port=29593 \
  train.py -c configs/deimv2/$CFG --use-amp >> $LOG 2>&1
echo "[$(ts)] pureir2ch_bs12 exit=$?"
echo "[$(ts)] DONE (see $LOG / outputs/deimv2_dinov3_l_aic_pureir2ch_bs12)"
