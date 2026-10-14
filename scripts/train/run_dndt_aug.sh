#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# DNDT-online-aug fine-tune: let the detector learn the inverse of the DNDT night
# degradation from the training data (no hand-coded test-time inverse).
# Train on BDD daytime + RandomDNDTNightify; validate on REAL BDD night.
set +e
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
TPY=/home/constant/miniconda3/envs/deimv2/bin/python
TRAINPY=$VENDOR/train.py
CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightsim_inorm_dndtaug.yml
INIT=$VENDOR/outputs/deimv2_dinov3_l_aic_nightsim_inorm/best_stg1.pth
OUTDIR=$VENDOR/outputs/deimv2_dinov3_l_aic_nightsim_inorm_dndtaug
LOG=$AIC/dndtaug.log
EVALLOG=$AIC/dndtaug_eval.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] gen config" | tee -a $LOG
cd $AIC
$TPY gen_dndt_aug_ft.py | tee -a $LOG

echo "[$(ts)] TRAIN (torch env=$TPY)" | tee -a $LOG
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29579 $TRAINPY -c $CFG -t $INIT --use-amp >> $LOG 2>&1
echo "[$(ts)] training exit=$?" | tee -a $LOG

CKPT=$(ls -t $OUTDIR/best_stg*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then CKPT=$(ls -t $OUTDIR/*.pth 2>/dev/null | head -1); fi
echo "[$(ts)] eval ckpt=$CKPT" | tee -a $LOG
if [ -z "$CKPT" ]; then echo "[$(ts)] NO CKPT -> abort" | tee -a $LOG; exit 1; fi

echo "===== [DNDT-aug] REAL BDD-NIGHT ($CKPT) =====" | tee -a $EVALLOG
CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29580 $TRAINPY --test-only -c $CFG -t $CKPT -d 0 >> $EVALLOG 2>&1

# synthetic DNDT brackets: did it learn the inverse? (reuse nightsim_inorm val cfgs)
for lvl in nightify dndt_night01 dndt_night05 dndt_night005; do
  vc=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightsim_inorm_val_${lvl}.yml
  [ -f "$vc" ] || continue
  echo "===== [DNDT-aug] synthetic $lvl =====" | tee -a $EVALLOG
  CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29581 $TRAINPY --test-only -c $vc -t $CKPT -d 0 >> $EVALLOG 2>&1 || true
done

echo "[$(ts)] DNDT-AUG PIPELINE DONE" | tee -a $LOG
