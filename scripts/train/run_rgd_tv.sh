#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Lightweight RGB-D ablation: torchvision RetinaNet, 4 variants, single-variable.
# Run from repo root:  bash $AIC/scripts/train/run_rgd_tv.sh
cd /home/constant/CodeBuddy/Dev/AIC
PY=/home/constant/miniconda3/envs/deimv2/bin/python
EPOCHS=${EPOCHS:-50}
IMSZ=${IMSZ:-512}
BATCH=${BATCH:-16}
LR=${LR:-2e-4}

for V in baseline inv_eq naive_clip8 zero; do
  echo "================ $V (epochs=$EPOCHS batch=$BATCH lr=$LR) ================"
  $PY scripts/train/train_rgd_tv.py --variant $V --epochs $EPOCHS --imgsz $IMSZ --batch $BATCH --lr $LR
done

echo "================ SUMMARY ================"
for V in baseline inv_eq naive_clip8 zero; do
  grep -h "DONE variant" work_dir/rgd_tv/$V/train.log | tail -1
done
