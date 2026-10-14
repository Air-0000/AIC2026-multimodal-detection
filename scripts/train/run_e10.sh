#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E10 = nightlog recipe with the REALISTIC night augmentation.
#
# Motivation: the nightXX protocol (non-uniform lamp lighting + Poisson-Gaussian
# noise + colour shift) dropped every model hard, and nightlog -- trained only
# with a GLOBAL attenuation -- still lost 0.023 going from darkXX to nightXX.
# It never saw lamp lighting or noise-limited inputs.
#
# RandomNight5ch reproduces the nightXX degradation at train time (randomised,
# log-uniform exposure), with IR/Depth untouched. Same base as nightlog
# (RDTTrack prompt + unfreeze 2), so the augmentation is the only variable.
#
# Evaluated on BOTH protocols so the darkXX/nightXX gap stays visible.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

while pgrep -f "run_e9\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for E9 (nightlog + input_norm)..."; sleep 120
done
echo "[$(ts)] E9 drained -> E10"
cd "$DEIM" || exit 1

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

E10=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight
echo "[$(ts)] === training E10 ($E10) ==="
: > /tmp/train_rnight.log
$PY train.py -c "configs/deimv2/${E10}.yml" \
    -t outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth --use-amp \
    >> /tmp/train_rnight.log 2>&1
echo "[$(ts)] E10 training done"
grep -a "best_stat" /tmp/train_rnight.log | tail -1 | sed 's/^/  in-domain /'

C=$(ck "outputs/$E10")
if [ -z "$C" ]; then echo "  WARN: no E10 ckpt"; exit 0; fi
mkdir -p /tmp/night_robust /tmp/night_real
for lv in dark50 dark25 dark12 dark05 night50 night25 night12 night05; do
    d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
    $PY train.py -c "configs/deimv2/${E10}_val_${lv}.yml" -r "$C" --use-amp --test-only \
        > "$d/rnight_${lv}.log" 2>&1
    printf '  %-8s AP=%s\n' "$lv" "$(ap "$d/rnight_${lv}.log")"
done
echo "[$(ts)] E10 DONE"
