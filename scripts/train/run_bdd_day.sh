#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# BDD100K DAYTIME reference for the RGB baseline.
#
# Separates the two effects hiding inside our BDD-night zero-shot AP50 0.284:
#   BDD-day AP50           -> pure AIC->BDD cross-dataset transfer (daytime)
#   BDD-night / BDD-day    -> our day->night drop on the SAME dataset/classes
# (Published Lower-Bound 41.1 is BDD-day-trained -> BDD-night, so it only
#  contains the day->night part -- this is what makes the two comparable.)
#
# Waits for the daytime image download to finish, builds the COCO json, then
# evaluates. The eval is tiny (~0.5GiB) so it runs alongside the training queue.
# =============================================================================
set -u
AIC=/home/constant/CodeBuddy/Dev/AIC
DEIM=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/bdd_night
mkdir -p "$OUT"
ts() { date '+%F %T'; }

while ! grep -q "^done:" /tmp/bdd_dl_day.log 2>/dev/null; do
    echo "[$(ts)] waiting for BDD daytime images..."; sleep 60
done
echo "[$(ts)] daytime images ready"

if [ ! -f "$AIC/data_aux/BDD100K/instances_bdd_daytime.json" ]; then
    $PY "$AIC/scripts/prep/prepare_bdd100k.py" --split daytime --no-download 2>&1 | tail -5
fi

cd "$DEIM" || exit 1
echo "=== BDD100K DAYTIME (zero-shot from AIC) ==="
$PY train.py -c configs/deimv2/deimv2_dinov3_l_aic_rgb3ch_bdd_daytime.yml \
    -r outputs/deimv2_dinov3_l_aic/best_stg2.pth --use-amp --test-only > "$OUT/rgb3ch_day.log" 2>&1
grep -a "Average Precision" "$OUT/rgb3ch_day.log" | head -3 | sed 's/^/  /'
grep -a "per-category" -A 12 "$OUT/rgb3ch_day.log" | sed 's/^/  /'
echo "[$(ts)] BDD DAY DONE"
