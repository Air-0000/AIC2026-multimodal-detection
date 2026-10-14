#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# 全自动流水线：等夜化生成 -> 建混合集 -> 起训 nightify 模型 -> 训后评测 -> 更新归档图
# 日志：/home/constant/CodeBuddy/Dev/AIC/nightify_pipeline.log  （评测块另追加到 bdd_night_zs_eval.log）
set +e
AIC=/home/constant/CodeBuddy/Dev/AIC
VENDOR=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/bin/python
TRAINPY=$VENDOR/train.py
CFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime_nightify.yml
EVALCFG=$VENDOR/configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_night.yml
INIT=$VENDOR/outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime/best_stg2.pth
OUTDIR=$VENDOR/outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime_nightify
ZSLOG=$AIC/bdd_night_zs_eval.log
LOG=$AIC/nightify_pipeline.log
GENPID=1647021
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] PIPELINE start" | tee -a $LOG

# 1) 等夜化生成进程结束
echo "[$(ts)] waiting for nightify gen pid $GENPID ..." | tee -a $LOG
while kill -0 $GENPID 2>/dev/null; do sleep 30; done
echo "[$(ts)] gen finished (nightify pngs: $(ls $AIC/data_aux/BDD100K/rgb_nightify 2>/dev/null | wc -l))" | tee -a $LOG

# 2) 确保混合数据集已建
if [ ! -f $AIC/data_aux/BDD100K_nightify/annotations/instances_train.json ]; then
  echo "[$(ts)] building dataset ..." | tee -a $LOG
  $PY $AIC/scripts/prep/build_bdd_nightify.py >> $LOG 2>&1
fi
echo "[$(ts)] dataset images: $(ls $AIC/data_aux/BDD100K_nightify/images 2>/dev/null | wc -l)" | tee -a $LOG

# 3) 起训（从 BDD-FT init）
echo "[$(ts)] training start (init $INIT)" | tee -a $LOG
cd $VENDOR
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29577 $TRAINPY -c $CFG -t $INIT --use-amp >> $LOG 2>&1
echo "[$(ts)] training exit=$?" | tee -a $LOG

# 4) 选 best 权重
CKPT=$(ls -t $OUTDIR/best_stg*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then CKPT=$(ls -t $OUTDIR/*.pth 2>/dev/null | head -1); fi
echo "[$(ts)] eval ckpt=$CKPT" | tee -a $LOG
if [ -z "$CKPT" ]; then echo "[$(ts)] NO CKPT -> abort" | tee -a $LOG; exit 1; fi

# 5) BDD-night zero-shot 评测（块追加到 ZSLOG，供 make_plots 解析）
echo "===== [NIGHTIFY(bdd_day_ft+nightify)] BDD-NIGHT ZERO-SHOT ($CKPT) =====" | tee -a $ZSLOG
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29578 $TRAINPY --test-only -c $EVALCFG -t $CKPT -d 0 >> $ZSLOG 2>&1
echo "[$(ts)] eval done" | tee -a $LOG

# 6) 更新归档 + 图
echo "[$(ts)] updating archive/plots ..." | tee -a $LOG
$PY $AIC/work_dir/zs_results_20260915/make_plots.py >> $LOG 2>&1
echo "[$(ts)] PIPELINE DONE" | tee -a $LOG
