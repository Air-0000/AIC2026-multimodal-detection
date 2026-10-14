#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Night-scenario robustness curve: AP vs RGB light level on AIC val, for
# RGB-only vs the two multimodal designs. IR/depth untouched -> any gap is
# attributable to the auxiliary modalities.
# Queued behind the other GPU jobs (single 15.5GiB card).
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/night_robust
mkdir -p "$OUT"
ts() { date '+%F %T'; }

# wait for the inject A/B AND the LLVIP ablation queue to be done
while pgrep -f "run_inject_ab.sh" >/dev/null 2>&1 \
   || pgrep -f "run_llvip_feat_ablation.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] queues drained -> start night robustness"

cd "$DEIM" || exit 1

run() {   # run <tag> <config> <ckpt>
    $PY train.py -c "configs/deimv2/$2" -r "$3" --use-amp --test-only \
        > "$OUT/$1.log" 2>&1
    printf '  %-28s AP=%s  AP50=%s\n' "$1" \
        "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
        "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
}

O=outputs
# model -> (config stem, ckpt)
declare -A CK=(
  [rgb3ch]=$O/deimv2_dinov3_l_aic/best_stg2.pth
  [prompt]=$O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
  [diffuse]=$O/deimv2_dinov3_l_aic_rdtdiffuse/best_stg2.pth
)
declare -A STEM=(
  [rgb3ch]=deimv2_dinov3_l_aic
  [prompt]=deimv2_dinov3_l_aic_rdttrack_heat
  [diffuse]=deimv2_dinov3_l_aic_rdtdiffuse
)

for lv in dark50 dark25 dark12 dark05; do
    echo "===== RGB level $lv ====="
    for m in rgb3ch prompt diffuse; do
        run "${m}_${lv}" "${STEM[$m]}_val_${lv}.yml" "${CK[$m]}"
    done
done

echo "[$(ts)] NIGHT ROBUSTNESS COMPLETE"
