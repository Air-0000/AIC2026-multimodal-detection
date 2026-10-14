#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Fill in the dark-curve points that were skipped because nightsim/
# nightsim_inorm were never registered in prep_night_robust.py (their
# *_val_darkXX.yml configs did not exist, so the eval died with FileNotFoundError
# and printed an empty AP). Tiny evals (~0.5GiB) -> safe alongside training.
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT"
ts() { date '+%F %T'; }
cd "$DEIM" || exit 1

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }

for pair in "deimv2_dinov3_l_aic_nightsim:nightsim" "deimv2_dinov3_l_aic_nightsim_inorm:inorm"; do
    stem="${pair%%:*}"; tag="${pair##*:}"
    C=$(ck "outputs/$stem")
    echo "[$(ts)] $tag ck=${C:-NONE}"
    [ -z "$C" ] && continue
    for lv in dark50 dark25 dark12 dark05; do
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$C" --use-amp --test-only \
            > "$OUT/${tag}_${lv}.log" 2>&1
        printf '  %-10s %-8s AP=%s\n' "$tag" "$lv" \
            "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/${tag}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
    done
done
echo "[$(ts)] EXTRA EVALS DONE"
