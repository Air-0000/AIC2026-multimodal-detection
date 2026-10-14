#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E9 = nightlog + input_norm: instance -- merge the two winners.
#
# Realistic-night protocol (prep_night_real.py) MEAN so far:
#     inorm    0.4567  (RGB-only; wins on brightness invariance)
#     nightlog 0.4060  (multimodal; wins on fusion + augmentation, most
#                       protocol-stable: dark 0.4285 -> night 0.4060, -0.023)
# The two mechanisms are independent, so this trains the combination.
#
# Also re-evaluates uf6/uf12 on the realistic protocol -- the first pass ran
# while uf6 was still training, so it scored a partially-trained checkpoint.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }

while pgrep -f "run_recover\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for E6/E7 (run_recover.sh)..."; sleep 120
done
echo "[$(ts)] E6/E7 drained -> E9"
cd "$DEIM" || exit 1

ck() { for f in best_stg2.pth best_stg1.pth last.pth; do [ -f "$1/$f" ] && { echo "$1/$f"; return; }; done; echo ""; }
ap() { grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$1" | head -1 | sed 's/.*\] = //' | tr -d '\r'; }

both() {   # both <stem> <ckpt> <tag>
    local stem=$1 ckpt=$2 tag=$3 lv
    mkdir -p /tmp/night_robust /tmp/night_real
    for lv in dark50 dark25 dark12 dark05 night50 night25 night12 night05; do
        d=/tmp/night_robust; case "$lv" in night*) d=/tmp/night_real;; esac
        cfg="configs/deimv2/${stem}_val_${lv}.yml"
        [ -f "$cfg" ] || { echo "    missing $cfg"; continue; }
        $PY train.py -c "$cfg" -r "$ckpt" --use-amp --test-only > "$d/${tag}_${lv}.log" 2>&1
        printf '    %-8s AP=%s\n' "$lv" "$(ap "$d/${tag}_${lv}.log")"
    done
}

echo "===== re-eval uf6/uf12 on the realistic protocol (previous pass was mid-training) ====="
for tag in uf6 uf12; do
    stem="deimv2_dinov3_l_aic_rdttrack_heat_night_log_${tag}"
    C=$(ck "outputs/$stem")
    echo "[$(ts)] $tag ck=${C:-NONE}"
    [ -z "$C" ] && continue
    for lv in night50 night25 night12 night05; do
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$C" --use-amp --test-only \
            > "/tmp/night_real/${tag}_${lv}.log" 2>&1
        printf '    %-8s AP=%s\n' "$lv" "$(ap /tmp/night_real/${tag}_${lv}.log)"
    done
done

echo "===== training E9 (nightlog + input_norm) ====="
E9=deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm
: > /tmp/train_nli.log
$PY train.py -c "configs/deimv2/${E9}.yml" \
    -t outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth --use-amp \
    >> /tmp/train_nli.log 2>&1
echo "[$(ts)] E9 training done"
grep -a "best_stat" /tmp/train_nli.log | tail -1 | sed 's/^/  in-domain /'
C=$(ck "outputs/$E9")
if [ -n "$C" ]; then echo "===== E9 both protocols ====="; both "$E9" "$C" nli; else echo "  WARN: no E9 ckpt"; fi
echo "[$(ts)] E9 DONE"
