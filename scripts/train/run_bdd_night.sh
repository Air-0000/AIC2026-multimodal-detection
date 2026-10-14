#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# BDD100K Nighttime detection -- ZERO-SHOT from AIC (RGB-only 3ch model).
#
# BDD100K has no infrared and no depth, so the RGB-only model is the only
# meaningful comparison; the 5ch models can additionally be probed in a
# "missing aux" configuration. Published bounds to compare against (mAP@0.5,
# 10-class protocol; we cover the 5 overlapping AIC classes):
#     Lower-Bound (day-trained):  Ped 50.0 Car 66.6 Bicycle 39.5 TL 41.0 TS 56.5
#     Upper-Bound (night-trained):Ped 52.1 Car 73.6 Bicycle 41.8 TL 52.2 TS 63.3
#
# Waits for prepare_bdd100k.py to finish, then evaluates. Runs alongside the
# night-training queue (RGB-only eval peak ~2-3GiB, card has ~6GiB spare).
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/bdd_night
mkdir -p "$OUT"
ts() { date '+%F %T'; }

J=/home/constant/CodeBuddy/Dev/AIC/data_aux/BDD100K/instances_bdd_night.json
while [ ! -f "$J" ]; do echo "[$(ts)] waiting for BDD100K prep..."; sleep 60; done
echo "[$(ts)] BDD100K night data ready"

cd "$DEIM" || exit 1

run() {   # run <stem> <ckpt> <tag>
    local stem=$1 ck=$2 tag=$3
    [ -f "$ck" ] || { echo "  skip $tag (no ckpt $ck)"; return; }
    echo "[$(ts)] eval $tag"
    $PY train.py -c "configs/deimv2/$stem.yml" -r "$ck" --use-amp --test-only > "$OUT/$tag.log" 2>&1
    echo "  --- $tag ---"
    grep -a "Average Precision" "$OUT/$tag.log" | head -3 | sed 's/^/    /'
    grep -a "per-category" -A 12 "$OUT/$tag.log" | sed 's/^/    /'
}

O=outputs
echo "=== BDD100K Nighttime (zero-shot from AIC) ==="
run deimv2_dinov3_l_aic_rgb3ch_bdd_night   $O/deimv2_dinov3_l_aic/best_stg2.pth              rgb3ch

echo "=== BDD100K Nighttime -- 5ch prompt with MISSING aux (diagnostic) ==="
run deimv2_dinov3_l_aic_rdttrack_heat_bdd_night \
    $O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth                                  prompt_noaux

echo "[$(ts)] BDD NIGHT DONE"
