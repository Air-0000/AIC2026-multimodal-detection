#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# DNDT (Debiased Teacher) UDA fine-tune: full self-training loop
#   source = BDD daytime + RandomDNDTNightify (supervised, night-like)
#   target = BDD REAL night (unlabeled) -> teacher pseudo-labels, ConCal (student-teacher agreement)
#   CDRC   = paste source night-like objects into real-night scenes (ICFB balanced)
#   val    = BDD REAL night (the metric)
set +e
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
TPY=/home/constant/miniconda3/envs/deimv2/bin/python
TRAINPY=$VENDOR/train.py
CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightsim_inorm_dndt_uda.yml
INIT=$VENDOR/outputs/deimv2_dinov3_l_aic_nightsim_inorm/best_stg1.pth
OUTDIR=$VENDOR/outputs/deimv2_dinov3_l_aic_nightsim_inorm_dndt_uda
LOG=$AIC/dndt_uda.log
EVALLOG=$AIC/dndt_uda_eval.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] gen config" | tee -a $LOG
cd $AIC
$TPY gen_dndt_uda_ft.py | tee -a $LOG

echo "[$(ts)] TRAIN (torch env=$TPY)" | tee -a $LOG
# clean stale torch shared-memory segments left by previously killed runs
rm -f /dev/shm/torch_* 2>/dev/null
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29582 $TRAINPY -c $CFG -t $INIT --use-amp >> $LOG 2>&1
echo "[$(ts)] training exit=$?" | tee -a $LOG

CKPT=$(ls -t $OUTDIR/best_stg*.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t $OUTDIR/*.pth 2>/dev/null | head -1)
echo "[$(ts)] eval ckpt=$CKPT" | tee -a $LOG
[ -z "$CKPT" ] && { echo "[$(ts)] NO CKPT -> abort" | tee -a $LOG; exit 1; }

echo "===== [DNDT-UDA] REAL BDD-NIGHT ($CKPT) =====" | tee -a $EVALLOG
CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29583 $TRAINPY --test-only -c $CFG -t $CKPT -d 0 >> $EVALLOG 2>&1

for lvl in nightify dndt_night01 dndt_night05 dndt_night005; do
  vc=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightsim_inorm_val_${lvl}.yml
  [ -f "$vc" ] || continue
  echo "===== [DNDT-UDA] synthetic $lvl =====" | tee -a $EVALLOG
  CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29584 $TRAINPY --test-only -c $vc -t $CKPT -d 0 >> $EVALLOG 2>&1 || true
done

echo "[$(ts)] DNDT-UDA PIPELINE DONE" | tee -a $LOG
