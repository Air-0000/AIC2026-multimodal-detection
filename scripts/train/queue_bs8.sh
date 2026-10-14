#!/bin/bash
# ============================================================================
# bs8 family: the three from-scratch / pretrained runs that could not fit at
# bs12 (see work/mk_bs8_configs.py).  All three share bs8 so that contrasts
# inside this family remain single-factor:
#
#   D8: 3ch clean 68ep SCRATCH     ----.
#   E8: 2ch clean 68ep SCRATCH     ----'--> D8 - E8 = channel gain (scratch)
#   B8: 3ch clean 68ep pretrained  ------> B8 - D8 = pretraining worth
#
# Measured already, kept as separate bs12 evidence:
#   A = 0.2947 (2ch clean bs12 pt)   B = 0.3124 (3ch clean bs12 pt)
# ============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
QUEUE_LOG=$AIC/queue_bs8.log
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

echo "[$(ts)] queue_bs8 launched" | tee -a $QUEUE_LOG

run_one deimv2_dinov3_l_aic_pureir3ch_na_scratch_bs8.yml $AIC/pureir3ch_na_scratch_bs8.log 29601
wait_free
run_one deimv2_dinov3_l_aic_pureir2ch_na_scratch_bs8.yml $AIC/pureir2ch_na_scratch_bs8.log 29602
wait_free
run_one deimv2_dinov3_l_aic_pureir3ch_na_bs8.yml          $AIC/pureir3ch_na_bs8.log         29603
wait_free

echo "[$(ts)] ===== QUEUE BS8 FINISHED =====" | tee -a $QUEUE_LOG
/home/constant/.workbuddy/binaries/python/envs/default/bin/python \
    $AIC/work/summarize_all.py >> $QUEUE_LOG 2>&1
echo "[$(ts)] summary written" | tee -a $QUEUE_LOG
