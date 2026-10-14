#!/bin/bash
# ============================================================================
# Full na-queue.  Waits for the run already on the GPU (A: pureir2ch_na),
# then executes B -> D -> E -> F strictly serially (single 16GB GPU).
#
#   A: 2ch  bs12 clean 68ep pretrained   (already running, started 01:12)
#   B: 3ch  bs12 clean 68ep pretrained
#   D: 3ch  bs12 clean 68ep SCRATCH
#   E: 2ch  bs12 clean 68ep SCRATCH
#   F: 3ch  bs12 clean 96ep SCRATCH
#
# Every answer afterwards is one subtraction (pairs differ in ONE factor):
#   bug cost           = A - 0.3188        (0.3188 = 2ch bs12 on dirty data)
#   channel gain (pt)  = B - A
#   pretraining worth  = B - D
#   channel gain (sc)  = D - E
#   under-training     = F - D
# ============================================================================
set -u
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
QUEUE_LOG=$AIC/queue_na_all.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

wait_free(){
  # Wait until no training process remains AND the GPU memory is actually back
  # down.  Checking pgrep alone is NOT enough: the previous run can be gone from
  # the process table while its CUDA context / dataloader workers still hold
  # ~1.6 GiB, which is exactly what OOM-killed the three scratch runs.
  local used guard=0
  while :; do
    if ! pgrep -f "train.py" > /dev/null 2>&1; then
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      used=${used:-0}
      if [ "$used" -lt 2500 ]; then
        break
      fi
    fi
    sleep 20
    guard=$((guard + 1))
    [ "$guard" -gt 180 ] && break   # give up after 1h instead of hanging forever
  done
  sleep 20
}

run_one(){
  local cfg=$1 log=$2 port=$3
  echo "[$(ts)] ===== QUEUE START $cfg (port $port) =====" | tee -a $QUEUE_LOG
  rm -f /dev/shm/torch_* 2>/dev/null
  cd $VENDOR
  CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=$port \
      train.py -c configs/deimv2/$cfg --use-amp >> $log 2>&1
  local rc=$?
  echo "[$(ts)] ===== QUEUE DONE  $cfg exit=$rc =====" | tee -a $QUEUE_LOG
}

echo "[$(ts)] queue_na_all launched; waiting for the current run to finish" | tee -a $QUEUE_LOG

wait_free
run_one deimv2_dinov3_l_aic_pureir3ch_na.yml              $AIC/pureir3ch_na.log              29596
wait_free
run_one deimv2_dinov3_l_aic_pureir3ch_na_scratch.yml      $AIC/pureir3ch_na_scratch.log      29597
wait_free
run_one deimv2_dinov3_l_aic_pureir2ch_na_scratch.yml      $AIC/pureir2ch_na_scratch.log      29598
wait_free
run_one deimv2_dinov3_l_aic_pureir3ch_na_scratch_e96.yml  $AIC/pureir3ch_na_scratch_e96.log  29599
wait_free

echo "[$(ts)] ===== QUEUE ALL FINISHED =====" | tee -a $QUEUE_LOG
/home/constant/.workbuddy/binaries/python/envs/default/bin/python \
    $AIC/work/summarize_all.py >> $QUEUE_LOG 2>&1
echo "[$(ts)] summary written to $QUEUE_LOG" | tee -a $QUEUE_LOG
