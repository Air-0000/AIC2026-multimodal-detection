#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Recovery run -- fix three queue bugs and finish the night programme.
#
# Bugs found after the first chain drained:
#   1) best_stg2.pth is only written once epoch >= collate_fn.stop_epoch (60).
#      E3 (ep80 but best at ep0), E4/E5 (ep40 < 60) therefore only produced
#      best_stg1.pth, and the eval scripts hard-coded best_stg2.pth ->
#      three fully-trained models were never evaluated.
#      Fix: ck() below resolves best_stg2 > best_stg1 > last.
#   2) gen_nightlog_unfreeze_configs wrote the LR via f"{lr}", producing the
#      *string* "5e-05" (PyYAML needs a '.' in the mantissa). LinearWarmup then
#      died on "can't multiply sequence by non-int of type 'float'" and E6/E7
#      aborted in ~14 s. Fix: format as a plain decimal + assert float.
#   3) CUT's argparse sees --norm as an ambiguous prefix of --normG/--normD.
#      Fix: pass --normG/--normD explicitly (in run_cut_day2night.sh).
#
# This script: 1) re-evaluates E3/E4/E5 on the dark curve (+BDD for the RGB
# ones), 2) re-runs E6/E7 deeper unfreeze. CUT/nightcut are chained after it.
# =============================================================================
set -u
AIC=/home/constant/CodeBuddy/Dev/AIC
DEIM=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT" /tmp/bdd_night
ts() { date '+%F %T'; }
cd "$DEIM" || exit 1

ck() {   # best available checkpoint: stg2 > stg1 > last
    local d=$1 f
    for f in best_stg2.pth best_stg1.pth last.pth; do
        if [ -f "$d/$f" ]; then echo "$d/$f"; return; fi
    done
    echo ""
}

ap_of() { grep -aF "IoU=0.50:0.95 | area=   all | maxDets=100 ] = " "$1" | head -1 | sed 's/.*\] = //' | tr -d '\r'; }
ap50_of() { grep -aF "IoU=0.50      | area=   all | maxDets=100 ] = " "$1" | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

curve() {   # curve <stem> <ckpt> <tag>
    local stem=$1 ckpt=$2 tag=$3 lv f
    for lv in dark50 dark25 dark12 dark05; do
        f="$OUT/${tag}_${lv}.log"
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$ckpt" --use-amp --test-only > "$f" 2>&1
        printf '  %-14s %-8s AP=%s\n' "$tag" "$lv" "$(ap_of "$f")"
    done
}

bdd() {   # bdd <stem> <ckpt> <tag>
    local stem=$1 ckpt=$2 tag=$3 sp f
    for sp in night daytime; do
        f="/tmp/bdd_night/${tag}_${sp}.log"
        $PY train.py -c "configs/deimv2/${stem}_bdd_${sp}.yml" -r "$ckpt" --use-amp --test-only > "$f" 2>&1
        printf '  %-14s bdd_%-8s AP=%s AP50=%s\n' "$tag" "$sp" "$(ap_of "$f")" "$(ap50_of "$f")"
    done
}

echo "[$(ts)] ===== 1) missed evaluations ====="
for pair in \
  "deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night:difnight:mm" \
  "deimv2_dinov3_l_aic_nightsim:nightsim:rgb" \
  "deimv2_dinov3_l_aic_nightsim_inorm:inorm:rgb" ; do
    stem="${pair%%:*}"; rest="${pair#*:}"; tag="${rest%%:*}"; kind="${rest##*:}"
    C=$(ck "outputs/$stem")
    echo "[$(ts)] $tag -> ${C:-NO CKPT}"
    if [ -n "$C" ]; then
        curve "$stem" "$C" "$tag"
        [ "$kind" = "rgb" ] && bdd "$stem" "$C" "$tag"
    fi
done

echo "[$(ts)] ===== 2) E6/E7 deeper unfreeze (LR bug fixed) ====="
for tag in uf6 uf12; do
    stem="deimv2_dinov3_l_aic_rdttrack_heat_night_log_${tag}"
    echo "[$(ts)] training $tag"
    : > "/tmp/train_${tag}.log"
    $PY train.py -c "configs/deimv2/${stem}.yml" \
        -t outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth --use-amp \
        >> "/tmp/train_${tag}.log" 2>&1
    grep -a "best_stat" "/tmp/train_${tag}.log" | tail -1 | sed 's/^/  in-domain /'
    C=$(ck "outputs/$stem")
    if [ -n "$C" ]; then curve "$stem" "$C" "$tag"; else echo "  WARN: no ckpt for $tag"; fi
done

echo "[$(ts)] RECOVER DONE"
