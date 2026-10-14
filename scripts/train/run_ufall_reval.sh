#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Correct AIC 8-level evaluation for E11 (ufall).
#
# run_e11_e12.sh evaluates E11 using rnight's *_val_*.yml, which lack
# `input_norm: instance` -- but E11 is trained WITH it. DINOv3STAsRDTTrack only
# calls _normalize_rgb when that flag is set, so the in-script numbers are a
# train/test mismatch and must not be used. This re-runs the same 8 levels with
# gen_ufall_val_configs.py's configs (input_norm present, everything else
# identical) and overwrites the logs.
#
# Queued behind run_bdd_all.sh so it never competes for the 16GiB card.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

while pgrep -f "run_bdd_all\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for BDD evals to finish..."; sleep 60
done
echo "[$(ts)] BDD drained -> E11 (ufall) correct re-eval"
cd "$DEIM" || exit 1

STEM=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall
ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null \
       | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

C=$(ck "outputs/$STEM")
if [ -z "$C" ]; then echo "  WARN: no ckpt for $STEM"; exit 0; fi
echo "[$(ts)] ufall ckpt=$C"

sd=0; cd_=0; sn=0; cn=0
for lv in dark50 dark25 dark12 dark05 night50 night25 night12 night05; do
    d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
    $PY train.py -c "configs/deimv2/${STEM}_val_${lv}.yml" -r "$C" \
        --use-amp --test-only > "$d/ufall_${lv}.log" 2>&1
    v=$(ap "$d/ufall_${lv}.log")
    printf '  %-8s AP=%s\n' "$lv" "$v"
    case "$lv" in
        dark*) [ -n "$v" ] && { sd=$(echo "$sd+$v" | bc -l); cd_=$((cd_+1)); } ;;
        night*) [ -n "$v" ] && { sn=$(echo "$sn+$v" | bc -l); cn=$((cn+1)); } ;;
    esac
done
md="-"; [ "$cd_" -gt 0 ] && md=$(printf '%.4f' "$(echo "$sd/$cd_" | bc -l)")
mn="-"; [ "$cn" -gt 0 ] && mn=$(printf '%.4f' "$(echo "$sn/$cn" | bc -l)")
echo "  ufall darkXX MEAN=$md  nightXX MEAN=$mn"
echo "[$(ts)] UFALL RE-EVAL DONE"
