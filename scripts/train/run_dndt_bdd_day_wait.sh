#!/bin/bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# 等 GPU 空闲(白天FT结束)后: 1) 训练 DNDT BDD daytime(nightcut_inorm) 2) 自动评测真实 BDD 夜
# 注意: 必须用 deimv2 conda 环境的 python (含 torch), 裸 python 会 ModuleNotFoundError
PY=/home/constant/miniconda3/envs/deimv2/bin/python
THRESH=3000
LOG=/home/constant/CodeBuddy/Dev/AIC/ft_dndt_day2.log
cd /home/constant/CodeBuddy/Dev/AIC
echo "[waiter] start $(date); waiting GPU mem < ${THRESH}MiB" >> "$LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
  echo "[waiter] $(date) GPU used=${used}MiB" >> "$LOG"
  if [ -n "$used" ] && [ "$used" -lt "$THRESH" ]; then
    echo "[waiter] GPU free, launching DNDT BDD daytime ..." >> "$LOG"
    break
  fi
  sleep 120
done

cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime.yml >> "$LOG" 2>&1
echo "[waiter] DNDT train finished exit=$? at $(date)" >> "$LOG"

# ---- 链式: 真实 BDD 夜评测 (bdd_night config, test-only) ----
# 只选训练产出的权重, 排除遗留的 eval.pth
CKPT=$(ls -t outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime/best_stg*.pth 2>/dev/null | head -n1)
[ -z "$CKPT" ] && CKPT=$(ls -t outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime/checkpoint*.pth 2>/dev/null | head -n1)
echo "[waiter] eval checkpoint=$CKPT" >> "$LOG"
if [ -n "$CKPT" ]; then
  $PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_nightcut_inorm_bdd_night.yml -r "$CKPT" --use-amp --test-only >> "$LOG" 2>&1
  echo "[waiter] BDD-night eval finished exit=$? at $(date)" >> "$LOG"
else
  echo "[waiter] NO checkpoint found, skip eval" >> "$LOG"
fi
