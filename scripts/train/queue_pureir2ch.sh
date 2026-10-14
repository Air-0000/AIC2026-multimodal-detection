#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Queue: wait for the BDD-DNDT-UDA run (GPU busy) to fully finish, then train
# the pure-IR 2ch diagnostic (heat residual + aligned raw IR, NO RGB).
# Read-out: standalone IR mAP on AIC val vs the 0.029 prior (1ch heat pseudo-RGB).
set -u
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=deimv2_dinov3_l_aic_pureir2ch.yml
LOG=$AIC/pureir2ch.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] queue started; waiting for BDD-DNDT-UDA (train+eval) to finish..."
while pgrep -f "nightsim_inorm_dndt_uda" > /dev/null; do sleep 60; done
# grace period: catch the gap between train.py exiting and its eval starting
sleep 180
if pgrep -f "nightsim_inorm_dndt_uda" > /dev/null; then
  echo "[$(ts)] eval still running; keep waiting..."
  while pgrep -f "nightsim_inorm_dndt_uda" > /dev/null; do sleep 60; done
fi
echo "[$(ts)] GPU free; cleaning shm; starting pure-IR-2ch diagnostic"
rm -f /dev/shm/torch_* 2>/dev/null
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.launch --nproc_per_node=1 --master_port=29590 \
  train.py -c configs/deimv2/$CFG --use-amp >> $LOG 2>&1
echo "[$(ts)] pure-ir2ch exit=$?"
echo "[$(ts)] PURE-IR-2CH DONE  (see $LOG / outputs/deimv2_dinov3_l_aic_pureir2ch)"
