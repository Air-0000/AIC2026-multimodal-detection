#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# E4/E5: day->night SIMULATED training for the RGB-only baseline.
#
# Evidence (BDD100K night zero-shot, rgb3ch, AIC-trained):
#     baseline                              AP 0.147 / AP50 0.284
#     + test-time brightness rescale / gamma  0.129-0.133 / 0.246-0.253  WORSE
#     + per-image standardisation             0.131 / 0.249             WORSE
# -> the night drop is not photometric drift at the input; the *weights* never
#    learned low-light features. So the fix is training, not input correction.
#
#   E4 deimv2_dinov3_l_aic_nightsim        RGB + RandomRGBDegrade5ch(3ch, p0.5,
#                                          brightness 0.05-0.80 log-uniform)
#   E5 deimv2_dinov3_l_aic_nightsim_inorm  E4 + backbone input_norm: instance
#      (expected weaker: the test-time standardisation probe already hurt, but
#       it is a one-line flag and rules the model-side option in or out)
#
# Both resume from the converged AIC RGB checkpoint and fine-tune 40 epochs.
# Serialised behind the night-training queue on the single GPU.
# =============================================================================
set -u
AIC=/home/constant/CodeBuddy/Dev/AIC
DEIM=$AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/bdd_night
mkdir -p "$OUT"
ts() { date '+%F %T'; }

while pgrep -f "run_night_e2e3\.sh" >/dev/null 2>&1; do sleep 120; done
echo "[$(ts)] night queue drained -> E4/E5 day->night training"
cd "$DEIM" || exit 1

eval_bdd_night() {   # eval_bdd_night <stem> <tag>
    $PY train.py -c "configs/deimv2/$1_bdd_night.yml" \
        -r "outputs/$1/best_stg2.pth" --use-amp --test-only > "$OUT/$2.log" 2>&1
    echo "  --- $2 (BDD night) ---"
    grep -a "Average Precision" "$OUT/$2.log" | head -3 | sed 's/^/    /'
    grep -a "per-category" -A 12 "$OUT/$2.log" | sed 's/^/    /'
}

for tag in nightsim nightsim_inorm; do
    echo "[$(ts)] === training AIC rgb3ch + $tag ==="
    $PY train.py -c "configs/deimv2/deimv2_dinov3_l_aic_${tag}.yml" \
        -t outputs/deimv2_dinov3_l_aic/best_stg2.pth --use-amp \
        >> "/tmp/train_${tag}.log" 2>&1
    echo "[$(ts)] $tag training done"
    # in-domain AIC val (clean day) for reference
    grep -a "best_stat" "/tmp/train_${tag}.log" | tail -1
    if [ -f "outputs/deimv2_dinov3_l_aic_${tag}/best_stg2.pth" ]; then
        eval_bdd_night "deimv2_dinov3_l_aic_${tag}" "rgb3ch_${tag}"
    else
        echo "  WARN: no ckpt for $tag"
    fi
done

echo "[$(ts)] E4/E5 DONE"
