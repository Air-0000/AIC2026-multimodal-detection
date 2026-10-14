#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E15: DepthPrior quick confirm.
#   train  = rnight_ufall_depthrior (depth prior: DLS+DLW train, DCT infer)
#            warm-started from ufall best_stg2, 24-epoch quick run
#   eval    = all night-robust levels + both/ir/dep-blk ablations, compared to
#            the existing ufall reference logs in /tmp/night_robust|night_real.
# The decisive read: does aux (IR+Depth) contribution GROW at near-black, and
# does E15 beat ufall overall on nightXX?
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

while pgrep -f "train\.py" >/dev/null 2>&1; do
    echo "[$(ts)] GPU busy (train.py running), waiting..."; sleep 120
done
echo "[$(ts)] GPU free -> E15"
cd "$DEIM" || exit 1

E15=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_depthrior
UFALL=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall
INIT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/best_stg2.pth
[ -f "$INIT" ] || { echo "FATAL: missing warm start $INIT"; exit 2; }

mkdir -p /tmp/night_robust /tmp/night_real

echo "[$(ts)] ===== E15: train $E15 (depth prior, warm from ufall, 24ep) ====="
: > /tmp/train_e15.log
$PY train.py -c "configs/deimv2/${E15}.yml" -t "$INIT" --use-amp >> /tmp/train_e15.log 2>&1
echo "[$(ts)] E15 training done"
grep -a "best_stat" /tmp/train_e15.log | tail -1 | sed 's/^/  in-domain /'
grep -aE "trainable params" /tmp/train_e15.log | head -1 | sed 's/^/  /'

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

C=$(ck "outputs/$E15")
[ -z "$C" ] && { echo "  WARN: no E15 ckpt"; exit 0; }
echo "[$(ts)] E15 ckpt=$C"

sd=0; sn=0
for lv in dark50 dark25 dark12 dark05 night50 night25 night12 night05; do
    d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
    $PY train.py -c "configs/deimv2/${E15}_val_${lv}.yml" -r "$C" --use-amp --test-only > "$d/nd15_${lv}.log" 2>&1
    v=$(ap "$d/nd15_${lv}.log"); uv=$(ap "$d/ufall_${lv}.log")
    printf '  %-8s E15=%-8s ufall=%-8s\n' "$lv" "${v:-NA}" "${uv:-NA}"
    case "$lv" in
        dark*)  [ -n "$v" ] && sd=$(echo "$sd+$v" | bc -l);;
        night*) [ -n "$v" ] && sn=$(echo "$sn+$v" | bc -l);;
    esac
done
printf '  E15 darkXX MEAN=%.4f  nightXX MEAN=%.4f\n' "$(echo "$sd/4"|bc -l)" "$(echo "$sn/4"|bc -l)"

echo "[$(ts)] ===== aux ablation (both/ir/dep blk) @ dark05/night05 ====="
for lv in dark05 night05; do
    for ab in abbothblk abirblk abdepblk; do
        d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
        $PY train.py -c "configs/deimv2/${E15}_val_${lv}_${ab}.yml" -r "$C" --use-amp --test-only > "$d/nd15_${lv}_${ab}.log" 2>&1
        v=$(ap "$d/nd15_${lv}_${ab}.log"); uv=$(ap "$d/ufall_${lv}_${ab}.log")
        printf '  %-8s %-10s E15=%-8s ufall=%-8s\n' "$lv" "$ab" "${v:-NA}" "${uv:-NA}"
    done
done
echo "[$(ts)] E15 DONE"
