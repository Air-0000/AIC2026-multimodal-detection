#!/bin/bash
# ============================================================================
# Re-run of the three runs that were OOM-killed at 04:47 (D, E, F).
# Root cause was NOT the scratch configs: the previous run had already left the
# process table but still held ~1.6 GiB of CUDA memory, so this one started with
# only 60 MiB free.  wait_free() now also polls nvidia-smi before launching.
#
#   D: 3ch clean bs12 68ep SCRATCH
#   E: 2ch clean bs12 68ep SCRATCH
#   F: 3ch clean bs12 96ep SCRATCH
#
# Already measured and reusable:
#   A = 0.2947 (2ch clean, pt)      B = 0.3124 (3ch clean, pt)
#   2ch dirty bs12 = 0.3188         3ch dirty bs12 = 0.3369
# ============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
QUEUE_LOG=$AIC/queue_scratch3.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

wait_free(){
  local used guard=0
  while :; do
    if ! pgrep -f "train.py" > /dev/null 2>&1; then
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      used=${used:-0}
      if [ "$used" -lt 2500 ]; then break; fi
    fi
    sleep 20
    guard=$((guard + 1))
    [ "$guard" -gt 180 ] && break
  done
  sleep 20
}

run_one(){
  local cfg=$1 log=$2 port=$3
  echo "[$(ts)] ===== START $cfg (port $port) =====" | tee -a $QUEUE_LOG
  rm -f /dev/shm/torch_* 2>/dev/null
  cd $VENDOR
  CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=$port \
      train.py -c configs/deimv2/$cfg --use-amp >> $log 2>&1
  echo "[$(ts)] ===== DONE  $cfg exit=$? =====" | tee -a $QUEUE_LOG
}

echo "[$(ts)] queue_scratch3 launched" | tee -a $QUEUE_LOG

# scratch logs are tainted by the aborted attempts -> start each one fresh
rm -f $AIC/pureir3ch_na_scratch.log $AIC/pureir2ch_na_scratch.log \
      $AIC/pureir3ch_na_scratch_e96.log

run_one deimv2_dinov3_l_aic_pureir3ch_na_scratch.yml      $AIC/pureir3ch_na_scratch.log      29597
wait_free
run_one deimv2_dinov3_l_aic_pureir2ch_na_scratch.yml      $AIC/pureir2ch_na_scratch.log      29598
wait_free
run_one deimv2_dinov3_l_aic_pureir3ch_na_scratch_e96.yml  $AIC/pureir3ch_na_scratch_e96.log  29599
wait_free

echo "[$(ts)] ===== QUEUE SCRATCH3 FINISHED =====" | tee -a $QUEUE_LOG
/home/constant/.workbuddy/binaries/python/envs/default/bin/python \
    $AIC/work/summarize_all.py >> $QUEUE_LOG 2>&1
echo "[$(ts)] summary written" | tee -a $QUEUE_LOG
