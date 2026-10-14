#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# BDD100K night -- test-time IlluminationNorm sweep (zero retrain).
#
# Question: how much of the RGB baseline's night drop is pure photometric drift
# (night input stats vs the daytime ImageNet Normalize it was trained with)?
# If rescaling/gamma-lifting the input recovers a lot of AP, the bottleneck is
# input statistics and night-sim augmentation should pay off; if it recovers
# nothing, the information itself is gone and only enhancement/adaptation helps.
#
# Serialised behind run_bdd_night.sh (single GPU, and the training queue is
# running concurrently with ~6GiB spare).
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/bdd_night
mkdir -p "$OUT"
ts() { date '+%F %T'; }

while pgrep -f "run_bdd_night\.sh" >/dev/null 2>&1; do sleep 30; done
echo "[$(ts)] baseline BDD run drained -> illumination sweep"
cd "$DEIM" || exit 1

run() {   # run <stem> <ckpt> <tag>
    local stem=$1 ck=$2 tag=$3
    $PY train.py -c "configs/deimv2/$stem.yml" -r "$ck" --use-amp --test-only > "$OUT/$tag.log" 2>&1
    printf '  %-22s AP=%s  AP50=%s\n' "$tag" \
        "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$tag.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
        "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$tag.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
}

CK=outputs/deimv2_dinov3_l_aic/best_stg2.pth
echo "=== BDD100K night -- IlluminationNorm sweep (rgb3ch) ==="
for tag in illum_g10_m45 illum_g07_m45 illum_g05_m45 illum_g07_m45_std; do
    run "deimv2_dinov3_l_aic_rgb3ch_bdd_night_${tag}" "$CK" "rgb3ch_${tag}"
done
echo "[$(ts)] ILLUM SWEEP DONE"
