#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Re-measure the IR/depth contribution with a VALID feature-space ablation
# (feature -> 0) instead of the invalid pixel-black ablation, on the three
# existing checkpoints. No retraining -- eval only, ~20s per run.
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/featabl
mkdir -p "$OUT"
ts() { date '+%F %T'; }

cd "$DEIM" || exit 1

run() {   # run <tag> <config> <ckpt>
    echo "[$(ts)] $1"
    $PY train.py -c "configs/deimv2/$2" -r "$3" --use-amp --test-only \
        > "$OUT/$1.log" 2>&1
    printf '  %-26s AP=%s  AP50=%s\n' "$1" \
        "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
        "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
}

O=outputs

echo "===== 5ch RDTTrack prompt (68ep, heat IR, ckpt=best_stg2) ====="
run rdttrack_full       deimv2_dinov3_l_aic_rdttrack_heat.yml                $O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
run rdttrack_noir       deimv2_dinov3_l_aic_rdttrack_heat_val_featnoir.yml   $O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
run rdttrack_nodepth    deimv2_dinov3_l_aic_rdttrack_heat_val_featnodepth.yml $O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
run rdttrack_noaux      deimv2_dinov3_l_aic_rdttrack_heat_val_featnoaux.yml  $O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth

echo "===== 4ch DiffFuse (50ep, raw IR, ckpt=best_stg2) ====="
run diffuse4ch_full     deimv2_dinov3_l_aic_diffuse_4ch.yml                  $O/deimv2_dinov3_l_aic_diffuse_4ch/best_stg2.pth
run diffuse4ch_noir     deimv2_dinov3_l_aic_diffuse_4ch_val_featnoir.yml     $O/deimv2_dinov3_l_aic_diffuse_4ch/best_stg2.pth

echo "===== 5ch RDTDiffuse hybrid (30ep, heat IR, ckpt=best_stg2) ====="
run rdtdiffuse_full     deimv2_dinov3_l_aic_rdtdiffuse_30ep.yml              $O/deimv2_dinov3_l_aic_rdtdiffuse_30ep/best_stg2.pth
run rdtdiffuse_noir     deimv2_dinov3_l_aic_rdtdiffuse_30ep_val_featnoir.yml   $O/deimv2_dinov3_l_aic_rdtdiffuse_30ep/best_stg2.pth
run rdtdiffuse_nodepth  deimv2_dinov3_l_aic_rdtdiffuse_30ep_val_featnodepth.yml $O/deimv2_dinov3_l_aic_rdtdiffuse_30ep/best_stg2.pth
run rdtdiffuse_noaux    deimv2_dinov3_l_aic_rdtdiffuse_30ep_val_featnoaux.yml  $O/deimv2_dinov3_l_aic_rdtdiffuse_30ep/best_stg2.pth

echo "[$(ts)] FEAT-ABLATION COMPLETE"
