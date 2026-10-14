#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# 修正版：用带 torch 的 deimv2 环境跑训练+评测（base 环境无 torch，上一版在此崩）
set +e
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
TPY=/home/constant/miniconda3/envs/deimv2/bin/python
MPY=/home/constant/miniconda3/bin/python
TRAINPY=$VENDOR/train.py
CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime_nightify.yml
EVALCFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_night.yml
INIT=$VENDOR/outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime/best_stg2.pth
OUTDIR=$VENDOR/outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime_nightify
ZSLOG=$AIC/bdd_night_zs_eval.log
LOG=$AIC/nightify_pipeline.log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] TRAIN+EVAL restart (torch env=$TPY)" | tee -a $LOG
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29577 $TRAINPY -c $CFG -t $INIT --use-amp >> $LOG 2>&1
echo "[$(ts)] training exit=$?" | tee -a $LOG

CKPT=$(ls -t $OUTDIR/best_stg*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then CKPT=$(ls -t $OUTDIR/*.pth 2>/dev/null | head -1); fi
echo "[$(ts)] eval ckpt=$CKPT" | tee -a $LOG
if [ -z "$CKPT" ]; then echo "[$(ts)] NO CKPT -> abort" | tee -a $LOG; exit 1; fi

echo "===== [NIGHTIFY(bdd_day_ft+nightify)] BDD-NIGHT ZERO-SHOT ($CKPT) =====" | tee -a $ZSLOG
CUDA_VISIBLE_DEVICES=0 $TPY -m torch.distributed.launch --nproc_per_node=1 --master_port=29578 $TRAINPY --test-only -c $EVALCFG -t $CKPT -d 0 >> $ZSLOG 2>&1
echo "[$(ts)] eval done" | tee -a $LOG

echo "[$(ts)] updating archive/plots ..." | tee -a $LOG
$MPY $AIC/work_dir/zs_results_20260915/make_plots.py >> $LOG 2>&1
echo "[$(ts)] PIPELINE DONE" | tee -a $LOG
