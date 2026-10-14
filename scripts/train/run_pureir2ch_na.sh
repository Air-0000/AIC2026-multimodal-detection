#!/bin/bash
# pureir2ch_na : raw_na + heat_na   (去 mis-align 污染后的干净 2ch 基线)
# 与 pureir2ch_bs12 的唯一差别: 数据目录 work_dir/deimv2_heat|deimv2_ir -> *_na
#   raw_na  = IR 原图 (不再被 prep_ir_heat.py 平移 46~86 px)
#   heat_na = clip(IR - RGB_gray, 0, 255)  (基于未平移的 IR)
# bs 保持 12, epoches/lr/warmup/增强 全不动 -> 单变量 = 修 bug 的收益。
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=deimv2_dinov3_l_aic_pureir2ch_na.yml
LOG=$AIC/pureir2ch_na.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] START pureir2ch_na"
rm -f /dev/shm/torch_* 2>/dev/null
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.launch --nproc_per_node=1 --master_port=29595 \
  train.py -c configs/deimv2/$CFG --use-amp >> $LOG 2>&1
echo "[$(ts)] pureir2ch_na exit=$?"
echo "[$(ts)] DONE (see $LOG / outputs/deimv2_dinov3_l_aic_pureir2ch_na)"
