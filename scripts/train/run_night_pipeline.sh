#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Day -> night pipeline (single 15.5GiB GPU, serialised).
#
#   AIC is a daytime-only training set but the test condition is night, so the
#   night-robustness curve has to become the primary metric. This driver:
#     1) waits for the running stage-2 unfreeze training to finish
#     2) evaluates the unfreeze model on the night curve
#     3) trains the day->night model (RandomRGBDegrade5ch, resume from prompt)
#     4) evaluates the day->night model on the night curve
#
#   Night curve = AP vs visible-light level on AIC val, IR/Depth untouched:
#       dark50 (dusk) / dark25 (twilight) / dark12 (night) / dark05 (deep night)
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT"
ts() { date '+%F %T'; }

cd "$DEIM" || exit 1

eval_model() {   # eval_model <stem> <ckpt> <tag>
    local stem=$1 ckpt=$2 tag=$3 lv ap ap50
    for lv in dark50 dark25 dark12 dark05; do
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$ckpt" --use-amp --test-only \
            > "$OUT/${tag}_${lv}.log" 2>&1
        ap=$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/${tag}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')
        ap50=$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/${tag}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')
        printf '  %-16s %-8s AP=%s  AP50=%s\n' "$tag" "$lv" "$ap" "$ap50"
    done
}

echo "[$(ts)] waiting for rdtdiffuse_unfreeze training to finish..."
while pgrep -f "deimv2_dinov3_l_aic_rdtdiffuse_unfreeze.yml" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] unfreeze training done"

if [ -f outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze/best_stg2.pth ]; then
    echo "[$(ts)] === night curve: unfreeze (diffuse + stage-2 unfreeze) ==="
    eval_model deimv2_dinov3_l_aic_rdtdiffuse_unfreeze \
        outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze/best_stg2.pth unfrz
else
    echo "[$(ts)] WARN: no unfreeze best_stg2.pth, skipping"
fi

echo "[$(ts)] === training day->night model (RandomRGBDegrade5ch) ==="
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night.yml \
    -t outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth --use-amp \
    >> /tmp/train_night_aug.log 2>&1
echo "[$(ts)] night-aug training done"

if [ -f outputs/deimv2_dinov3_l_aic_rdttrack_heat_night/best_stg2.pth ]; then
    echo "[$(ts)] === night curve: nightaug (day->night trained) ==="
    eval_model deimv2_dinov3_l_aic_rdttrack_heat_night \
        outputs/deimv2_dinov3_l_aic_rdttrack_heat_night/best_stg2.pth nightaug
else
    echo "[$(ts)] WARN: no nightaug best_stg2.pth, skipping"
fi

echo "[$(ts)] NIGHT PIPELINE COMPLETE"
