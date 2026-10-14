#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E6/E7: deeper backbone unfreeze on the nightlog recipe.
#
#   nightlog = RDTTrack prompt + unfreeze 2 + RandomRGBDegrade5ch(log)   MEAN 0.4285
#   RGB-only baseline trains the WHOLE ViT-S backbone and reaches 0.525 in-domain
#   vs our 0.475, and unfreeze 0->2 was the biggest single lever (+0.049), so the
#   unfreeze depth is the natural next knob. DINOv3 ViT-S/16 has 12 blocks.
#
#   E6  unfreeze  6, backbone lr 5e-5     clean depth-only delta vs nightlog
#   E7  unfreeze 12, backbone lr 2.5e-5   full depth with a gentler LR (the RGB
#        baseline, which unfreezes everything, uses 1.25e-5 -- 5e-5 over 12
#        blocks would very likely wash out the DINOv3 prior)
#
# Both resume from the *prompt* checkpoint like nightlog, so the comparison is
# apples-to-apples. Serialised behind the E4/E5 queue on the single GPU.
# =============================================================================
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT"
ts() { date '+%F %T'; }

while pgrep -f "run_aic_nightsim\.sh" >/dev/null 2>&1; do
    echo "[$(ts)] waiting for E4/E5 queue..."; sleep 120
done
echo "[$(ts)] E4/E5 drained -> start E6/E7 deeper unfreeze"
cd "$DEIM" || exit 1

curve() {   # curve <stem> <ckpt> <tag>
    local stem=$1 ck=$2 tag=$3 lv ap
    for lv in dark50 dark25 dark12 dark05; do
        $PY train.py -c "configs/deimv2/${stem}_val_${lv}.yml" -r "$ck" --use-amp --test-only \
            > "$OUT/${tag}_${lv}.log" 2>&1
        ap=$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/${tag}_${lv}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')
        printf '  %-6s %-8s AP=%s\n' "$tag" "$lv" "$ap"
    done
}

for tag in uf6 uf12; do
    stem="deimv2_dinov3_l_aic_rdttrack_heat_night_log_${tag}"
    echo "[$(ts)] === training $tag ==="
    $PY train.py -c "configs/deimv2/${stem}.yml" \
        -t outputs/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth --use-amp \
        >> "/tmp/train_${tag}.log" 2>&1
    echo "[$(ts)] $tag training done"
    grep -a "best_stat" "/tmp/train_${tag}.log" | tail -1 | sed 's/^/  in-domain /'
    CK="outputs/${stem}/best_stg2.pth"
    if [ -f "$CK" ]; then curve "$stem" "$CK" "$tag"; else echo "  WARN: no ckpt for $tag"; fi
done
echo "[$(ts)] E6/E7 DONE"
