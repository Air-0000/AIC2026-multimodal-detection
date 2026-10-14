#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Does the multimodal model actually READ IR/Depth?
#
# "multimodal loses to RGB-only" has two very different explanations, and the
# fix is different for each:
#   (a) the fusion learned nothing usable from aux  -> the aux branches are dead
#       weight and the model is RGB-only with a handicap;
#   (b) it learned aux features but they do not pay off at night -> the fusion
#       design, not the data, is wrong.
#
# Replacing the aux channels with all-black images at inference separates them:
#   AP unchanged  -> the model never used them  -> (a)
#   AP collapses  -> the model depends on them  -> (b)
#
# Two models (rnight = best 5ch recipe, ufall = E11 full-unfreeze) and three
# variants: IR blacked, Depth blacked, both blacked. Baselines with the real aux
# come from the existing /tmp/night_{robust,real}/<tag>_<lv>.log.
#
# Note bothblk is also exactly what the BDD100K runs see, since BDD has neither
# sensor -- so this doubles as a check on those numbers being meaningful.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

while pgrep -f "run_cut_nightcut\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for the CUT/nightcut chain..."; sleep 120
done
echo "[$(ts)] GPU free -> aux ablation"
cd "$DEIM" || exit 1

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null \
       | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

STEMS=("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight:rnight"
       "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall:ufall")
LEVELS="dark50 dark05 night50 night05"
VARIANTS="irblk depblk bothblk"

for pair in "${STEMS[@]}"; do
    stem="${pair%%:*}"; tag="${pair##*:}"
    C=$(ck "outputs/$stem")
    [ -z "$C" ] && { echo "$tag: NO CKPT"; continue; }
    echo "[$(ts)] ---- $tag ($C)"
    for vname in $VARIANTS; do
        for lv in $LEVELS; do
            d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
            $PY train.py -c "configs/deimv2/${stem}_val_${lv}_ab${vname}.yml" -r "$C" \
                --use-amp --test-only > "$d/ab_${tag}_${lv}_${vname}.log" 2>&1
        done
    done
done

# ---------------------------------------------------------------- table
echo
echo "===== aux ablation (AIC val, AP 0.5:0.95) ====="
printf '%-8s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n' \
    model variant dark50 dark05 night50 night05 MEANdark MEANnight
for pair in "${STEMS[@]}"; do
    tag="${pair##*:}"
    for vname in real irblk depblk bothblk; do
        row=""; sd=0; sn=0; ok=1
        for lv in dark50 dark05; do
            if [ "$vname" = real ]; then f=/tmp/night_robust/${tag}_${lv}.log
            else f=/tmp/night_robust/ab_${tag}_${lv}_${vname}.log; fi
            v=$(ap "$f"); [ -z "$v" ] && { v="-"; ok=0; }
            row="$row $(printf '%-9s' "$v")"; [ "$v" != "-" ] && sd=$(echo "$sd+$v" | bc -l)
        done
        for lv in night50 night05; do
            if [ "$vname" = real ]; then f=/tmp/night_real/${tag}_${lv}.log
            else f=/tmp/night_real/ab_${tag}_${lv}_${vname}.log; fi
            v=$(ap "$f"); [ -z "$v" ] && { v="-"; ok=0; }
            row="$row $(printf '%-9s' "$v")"; [ "$v" != "-" ] && sn=$(echo "$sn+$v" | bc -l)
        done
        md=$( [ $ok -eq 1 ] && printf '%.4f' "$(echo "$sd/2" | bc -l)" || echo "-")
        mn=$( [ $ok -eq 1 ] && printf '%.4f' "$(echo "$sn/2" | bc -l)" || echo "-")
        printf '%-8s %-9s %s %-9s %-9s\n' "$tag" "$vname" "$row" "$md" "$mn"
    done
done
echo "[$(ts)] AUX ABLATION DONE"
