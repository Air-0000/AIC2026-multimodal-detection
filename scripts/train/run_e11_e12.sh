#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E11 + E12, run back to back on the (now idle) GPU.
#
# Standing after E1-E10 (darkXX / nightXX MEAN over dark50,25,12,05):
#   inorm  RGB-only, 22.9M trainable   dark 0.5130  night 0.4567   <-- best
#   nli    5ch,      4.8M  trainable   dark 0.4483  night 0.3992
#   rnight 5ch,      4.8M  trainable   dark 0.4428  night 0.4135   <-- best 5ch
#
# E11 = rnight + input_norm: instance + unfreeze_rgb_stages: 12.
#   Closes the confound: every 5ch run so far trained only 1.3-4.8M of 22.9M
#   params (log line "RGB backbone frozen=True"), while inorm trains all 22.9M.
#   If a fully-trainable RGB stream still cannot beat inorm, IR/Depth carry no
#   usable signal here and the multimodal line should be dropped.
#
# E12 = inorm + RandomNight5ch (realistic night augmentation at train time).
#   inorm's only weak spot is the realistic protocol (dark->night drops 0.056);
#   it was trained only with a global brightness attenuation.
#
# Both evaluated on BOTH protocols, all 8 levels, so the table stays comparable.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

while pgrep -f "train\.py" >/dev/null 2>&1; do
    echo "[$(ts)] GPU busy (train.py running), waiting..."; sleep 120
done
echo "[$(ts)] GPU free -> E11/E12"
cd "$DEIM" || exit 1

mkdir -p /tmp/night_robust /tmp/night_real

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" 2>/dev/null \
       | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

both() {   # both <val_stem> <ckpt> <tag>
    local stem=$1 ckpt=$2 tag=$3 lv d
    for lv in dark50 dark25 dark12 dark05 night50 night25 night12 night05; do
        d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$ckpt" \
            --use-amp --test-only > "$d/${tag}_${lv}.log" 2>&1
        printf '    %-8s AP=%s\n' "$lv" "$(ap "$d/${tag}_${lv}.log")"
    done
}

# ------------------------------------------------------------------ E11
E11=deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall
INIT11=outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight/best_stg2.pth
echo "[$(ts)] ===== E11: train $E11 (unfreeze ALL 12 blocks + input_norm) ====="
echo "  warm start: $INIT11"
: > /tmp/train_ufall.log
$PY train.py -c "configs/deimv2/${E11}.yml" -t "$INIT11" --use-amp \
    >> /tmp/train_ufall.log 2>&1
echo "[$(ts)] E11 training done"
grep -aE "trainable params" /tmp/train_ufall.log | head -1 | sed 's/^/  /'
grep -a "best_stat" /tmp/train_ufall.log | tail -1 | sed 's/^/  in-domain /'
C=$(ck "outputs/$E11")
if [ -n "$C" ]; then
    echo "[$(ts)] E11 ckpt=$C"
    both "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight" "$C" ufall
else
    echo "  WARN: no E11 ckpt"
fi

# ------------------------------------------------------------------ E12
E12=deimv2_dinov3_l_aic_nightsim_inorm_rnight
INIT12=outputs/deimv2_dinov3_l_aic_nightsim_inorm/best_stg1.pth
echo "[$(ts)] ===== E12: train $E12 (inorm + RandomNight5ch) ====="
echo "  warm start: $INIT12"
: > /tmp/train_inorm_rnight.log
$PY train.py -c "configs/deimv2/${E12}.yml" -t "$INIT12" --use-amp \
    >> /tmp/train_inorm_rnight.log 2>&1
echo "[$(ts)] E12 training done"
grep -a "best_stat" /tmp/train_inorm_rnight.log | tail -1 | sed 's/^/  in-domain /'
C=$(ck "outputs/$E12")
if [ -n "$C" ]; then
    echo "[$(ts)] E12 ckpt=$C"
    both "deimv2_dinov3_l_aic_nightsim_inorm" "$C" irn
else
    echo "  WARN: no E12 ckpt"
fi

echo "[$(ts)] E11/E12 DONE"
