#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# Re-evaluate every trained model on the REALISTIC night protocol
# (prep_night_real.py: non-uniform lamp lighting + Poisson-Gaussian sensor
# noise + colour shift + global attenuation), so model selection stops being
# driven by a metric that global-invariance can game.
#
# checkpoint resolution: best_stg2 > best_stg1 > last (best_stg2 is only written
# once epoch >= collate_fn.stop_epoch, which several short runs never reached).
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_real
mkdir -p "$OUT"
ts() { date '+%F %T'; }
cd "$DEIM" || exit 1

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }

# stem:tag
PAIRS="
deimv2_dinov3_l_aic:rgb3ch
deimv2_dinov3_l_aic_rdttrack_heat:prompt
deimv2_dinov3_l_aic_rdtdiffuse:diffuse
deimv2_dinov3_l_aic_rdtdiffuse_unfreeze:unfrz
deimv2_dinov3_l_aic_rdttrack_heat_night:nightaug
deimv2_dinov3_l_aic_rdttrack_heat_night_log:nightlog
deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night:difnight
deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf6:uf6
deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf12:uf12
deimv2_dinov3_l_aic_nightsim:nightsim
deimv2_dinov3_l_aic_nightsim_inorm:inorm
deimv2_dinov3_l_aic_nightcut:nightcut
"

for p in $PAIRS; do
    stem="${p%%:*}"; tag="${p##*:}"
    C=$(ck "outputs/$stem")
    if [ -z "$C" ]; then echo "[$(ts)] $tag: no checkpoint, skip"; continue; fi
    echo "[$(ts)] $tag  ($(basename "$C"))"
    for lv in night50 night25 night12 night05; do
        cfg="configs/deimv2/${stem}_val_${lv}.yml"
        [ -f "$cfg" ] || { echo "    missing $cfg"; continue; }
        $PY train.py -c "$cfg" -r "$C" --use-amp --test-only > "$OUT/${tag}_${lv}.log" 2>&1
        printf '    %-8s AP=%s\n' "$lv" \
            "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/${tag}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
    done
done
echo "[$(ts)] NIGHT-REAL EVAL DONE"
