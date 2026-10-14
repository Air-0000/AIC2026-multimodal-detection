#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E14: 5ch with modality dropout removed -- the last fair test of the fusion.
#
# See gen_e14_nodrop.py for the full diagnosis. Short version: the heat channel
# carries a real person/animal signal (in/out 1.65 / 1.57, measured on the GT
# boxes), yet every well-sampled class gets WORSE when aux is added. The prime
# suspect is ir_drop_prob/depth_drop_prob = 0.15, which teaches a 1800-image
# model that ignoring aux is the safe optimum. E14 sets both to 0 and warm-starts
# from E11 (full fine-tune + input_norm), so the fusion has no choice but to use
# the aux channels.
#
# Queued behind the aux-ablation so nothing competes for the card. Prints the
# person-class AP alongside inorm and ufall, because that is the decisive number:
# person is where the thermal signature is strongest and where RGB-only currently
# wins by the widest margin.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

# Runs last: E14 is a mechanism probe, not a submission candidate, so it yields
# to everything that can change the metric or the model choice.
while pgrep -f "run_ufall_reval\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for the BDD + ufall re-eval chain..."; sleep 120
done
echo "[$(ts)] GPU free -> E14"
cd "$DEIM" || exit 1

E14=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_nodrop
UFS=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall
INIT=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall/best_stg2.pth
VALSTEM=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall   # has input_norm

if [ ! -f "$INIT" ]; then echo "FATAL: missing warm start $INIT"; exit 2; fi

echo "[$(ts)] ===== E14: train $E14 (no modality dropout) ====="
: > /tmp/train_nodrop.log
$PY train.py -c "configs/deimv2/${E14}.yml" -t "$INIT" --use-amp \
    >> /tmp/train_nodrop.log 2>&1
echo "[$(ts)] E14 training done"
grep -a "best_stat" /tmp/train_nodrop.log | tail -1 | sed 's/^/  in-domain /'

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null | head -1 | sed 's/.*\] = //' | tr -d '\r'; }
percat() { awk '/per-category/,0' "$1" 2>/dev/null | grep -aE "^ *$2 " | sed 's/^ *//'; }

C=$(ck "outputs/$E14")
[ -z "$C" ] && { echo "  WARN: no E14 ckpt"; exit 0; }
echo "[$(ts)] E14 ckpt=$C"

sd=0; sn=0
for lv in dark50 dark25 dark12 dark05 night50 night25 night12 night05; do
    d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
    $PY train.py -c "configs/deimv2/${VALSTEM}_val_${lv}.yml" -r "$C" --use-amp --test-only \
        > "$d/nd14_${lv}.log" 2>&1
    v=$(ap "$d/nd14_${lv}.log")
    printf '  %-8s AP=%s\n' "$lv" "$v"
    case "$lv" in
        dark*)  [ -n "$v" ] && sd=$(echo "$sd+$v" | bc -l) ;;
        night*) [ -n "$v" ] && sn=$(echo "$sn+$v" | bc -l) ;;
    esac
done
printf '  E14 darkXX MEAN=%.4f  nightXX MEAN=%.4f\n' \
    "$(echo "$sd/4" | bc -l)" "$(echo "$sn/4" | bc -l)"

for split in night daytime; do
    $PY train.py -c "configs/deimv2/${E14}_bdd_${split}.yml" -r "$C" --use-amp --test-only \
        > "/tmp/bdd_night/nd14_${split}.log" 2>&1
    printf '  BDD %-8s AP=%s\n' "$split" "$(ap /tmp/bdd_night/nd14_${split}.log)"
done

echo
echo "===== did the fusion finally read the heat?  (person AP, AIC val) ====="
for lv in dark05 night05; do
    d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
    printf '  %s\n' "$lv"
    printf '    inorm (RGB-only) : %s\n' "$(percat "$d/inorm_${lv}.log" person)"
    printf '    ufall (E11)      : %s\n' "$(percat "$d/ufall_${lv}.log" person)"
    printf '    E14   (no drop)  : %s\n' "$(percat "$d/nd14_${lv}.log" person)"
done
echo "[$(ts)] E14 DONE"
