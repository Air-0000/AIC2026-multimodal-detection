#!/bin/bash
# pureir3ch_na : raw_na + heat_na + localbg_na   (符合预期栈 raw+heat+deleak)
# 与 pureir2ch_na 的唯一差别: 第三路 localbg + ir_in_chans 2->3 + Normalize 2->3 值
#   localbg_na = clip(IR - median(IR,31x31), 0, 255)  中值 top-hat
#   框内外对比度 4.17 (raw_na 1.37 / heat_na 1.50)，与 raw/heat corr 仅 0.483/0.424
#   -> 语义上就是 deleak 想要的「压掉背景只留热异常」，但用 IR 自身局部背景作参考
#      (原 deleak 定义 IR=热+k*VIS 已被证伪: 分层 slope -0.28~-0.53, 边缘 corr ~0)
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=deimv2_dinov3_l_aic_pureir3ch_na.yml
LOG=$AIC/pureir3ch_na.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] START pureir3ch_na"
rm -f /dev/shm/torch_* 2>/dev/null
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.launch --nproc_per_node=1 --master_port=29596 \
  train.py -c configs/deimv2/$CFG --use-amp >> $LOG 2>&1
echo "[$(ts)] pureir3ch_na exit=$?"
echo "[$(ts)] DONE (see $LOG / outputs/deimv2_dinov3_l_aic_pureir3ch_na)"
