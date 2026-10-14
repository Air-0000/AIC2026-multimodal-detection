#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# The REAL-night column for every model we might submit.
#
# Why this exists: the darkXX protocol is "multiply the visible by a global
# constant" -- precisely the transform a per-image instance normalisation is
# invariant to -- so inorm's darkXX 0.5130 is a tautology rather than a night
# skill, and on real night (BDD100K) inorm is *worse* than the plain rgb3ch
# baseline (AP50 0.274 vs 0.284, see /tmp/bdd_night/inorm_night.log).
#
# BDD100K night = 3929 imgs / 63458 boxes / the 5 AIC-overlapping classes. It is
# the only REAL night data available, so no model gets picked without a number
# on it. BDD has no IR and no depth, so the 5ch checkpoints run with all-black
# aux -- the honest "sensors absent" deployment number.
#
# Waits for run_e11_e12.sh (E11 + E12 training) so nothing competes for the
# 16GiB card, then evaluates night + daytime for every night-trained model.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/bdd_night
mkdir -p "$OUT"
ts() { date '+%F %T'; }

# Reordered: the CUT/nightcut chain (E13) is the only experiment that can still
# raise the metric, so it now runs first and this diagnostic set follows it.
while pgrep -f "run_aux_ablation\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for the aux ablation to free the GPU..."; sleep 120
done
echo "[$(ts)] aux ablation drained -> BDD100K evals"
cd "$DEIM" || exit 1

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null \
       | head -1 | sed 's/.*\] = //' | tr -d '\r'; }
ap50() { grep -a 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null \
       | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

# stem -> tag.  (families that already have a number are printed at the end)
MODELS=(
  "deimv2_dinov3_l_aic_rdttrack_heat_night_log:nightlog"
  "deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm:nli"
  "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight:rnight"
  "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall:ufall"
  "deimv2_dinov3_l_aic_nightsim_inorm_rnight:irn"
)

for pair in "${MODELS[@]}"; do
    stem="${pair%%:*}"; tag="${pair##*:}"
    C=$(ck "outputs/$stem")
    if [ -z "$C" ]; then echo "[$(ts)] $tag: NO CKPT, skip"; continue; fi
    echo "[$(ts)] ---- $tag  ($C)"
    for split in night daytime; do
        cfg="configs/deimv2/${stem}_bdd_${split}.yml"
        if [ ! -f "$cfg" ]; then echo "    missing $cfg"; continue; fi
        $PY train.py -c "$cfg" -r "$C" --use-amp --test-only > "$OUT/${tag}_${split}.log" 2>&1
        printf '    %-8s AP=%s  AP50=%s\n' "$split" \
            "$(ap "$OUT/${tag}_${split}.log")" "$(ap50 "$OUT/${tag}_${split}.log")"
    done
done

# ---------------------------------------------------------------- summary
echo
echo "=============== BDD100K (real images, zero-shot from AIC) ==============="
printf '%-12s %-10s %-10s %-10s %-10s\n' model bdd_day bdd_night ratio night/day
findlog() {   # findlog <tag> <split>
    for c in "$OUT/$1_$2.log" "$OUT/$1.log" "$OUT/$1_$2.log"; do
        [ -f "$c" ] && { echo "$c"; return; }
    done
    echo ""
}
for pair in \
    "rgb3ch:rgb3ch" "nightsim:nightsim" "inorm:inorm" \
    "prompt(noaux):prompt_noaux" \
    "nightlog:nightlog" "nli:nli" "rnight:rnight" "ufall:ufall" "irn:irn"; do
    tag="${pair%%:*}"; f="${pair##*:}"
    dn=$(findlog "$f" daytime); nn=$(findlog "$f" night)
    [ -z "$dn" ] && dn=$(findlog "$f" day)
    d=$(ap "$dn"); n=$(ap "$nn")
    r="-"; [ -n "$d" ] && [ -n "$n" ] && r=$(printf '%.3f' "$(echo "scale=6; $n/$d" | bc -l)")
    printf '%-12s %-10s %-10s %-10s\n' "$tag" "${d:-N/A}" "${n:-N/A}" "$r"
done
echo "[$(ts)] BDD EVALS DONE"
