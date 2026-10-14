#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E2 / E3 -- night-training variants that target the dark05/12/25/50 curve.
#
#   E2 = prompt  + unfreeze + degrade(LOG-uniform light)  resume from prompt ckpt
#   E3 = diffuse + unfreeze + degrade(LOG-uniform light)  resume from unfreeze ckpt
#        (E3 also exercises the fixed OrthogonalProjectionFusion -- true
#         Gram-Schmidt, no quadratic magnitude blow-up)
#
# Log-uniform light sampling gives the four (~x2-spaced) evaluation levels equal
# mass; the E1 linear draw over-sampled the bright end.
#
# Serialised behind run_llvip640.sh on the single 15.5GiB GPU.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT"
ts() { date '+%F %T'; }

cd "$DEIM" || exit 1

eval_model() {   # eval_model <stem> <ckpt> <tag>
    local stem=$1 ckpt=$2 tag=$3 lv ap
    for lv in dark50 dark25 dark12 dark05; do
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$ckpt" --use-amp --test-only \
            > "$OUT/${tag}_${lv}.log" 2>&1
        ap=$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/${tag}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')
        printf '  %-10s %-8s AP=%s\n' "$tag" "$lv" "$ap"
    done
}

train_one() {    # train_one <config> <resume_ckpt> <log> <tag> <stem> <out_ckpt>
    local cfg=$1 ck=$2 log=$3 tag=$4 stem=$5 outp=$6
    echo "[$(ts)] === training $tag ==="
    $PY train.py -c "configs/deimv2/$cfg" -t "$ck" --use-amp >> "$log" 2>&1
    echo "[$(ts)] $tag training done"
    if [ -f "$outp" ]; then
        eval_model "$stem" "$outp" "$tag"
    else
        echo "[$(ts)] WARN: $outp missing"
    fi
}

while pgrep -f "run_llvip640.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] LLVIP640 drained -> start E2/E3"

train_one deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml \
    outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth \
    /tmp/train_night_log.log nightlog \
    deimv2_dinov3_l_aic_rdttrack_heat_night_log \
    outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log/best_stg2.pth

train_one deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night.yml \
    outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze/best_stg2.pth \
    /tmp/train_difnight.log difnight \
    deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night \
    outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night/best_stg2.pth

echo "[$(ts)] E2/E3 COMPLETE"
