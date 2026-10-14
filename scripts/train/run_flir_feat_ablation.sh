#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# FLIR IR ablation with the VALID feature-space protocol (test-only, never trained).
# Mirrors run_llvip_feat_ablation.sh but on the FLIR thermal driving set.
# Queued behind EVERYTHING else: waits for run_inject_ab, run_llvip_feat_ablation,
# run_night_robust and any train.py to finish first (single 15.5GiB card) so it
# never contends for the GPU.
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/flir_featabl
mkdir -p "$OUT"
cd "$DEIM" || exit 1
ts() { date '+%F %T'; }

echo "[$(ts)] waiting for inject_ab + llvip + night + train.py to finish"
while pgrep -f "run_inject_ab.sh" >/dev/null 2>&1 \
   || pgrep -f "run_llvip_feat_ablation.sh" >/dev/null 2>&1 \
   || pgrep -f "run_night_robust.sh" >/dev/null 2>&1 \
   || pgrep -f "train.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] GPU free -> start FLIR feat-ablation"

run() {   # run <tag> <config> <ckpt>
    [ -f "$3" ] || { echo "  skip $1 (no ckpt $3)"; return; }
    echo "[$(ts)] $1"
    $PY train.py -c "configs/deimv2/$2" -r "$3" --use-amp --test-only \
        > "$OUT/$1.log" 2>&1
    printf '  %-34s AP=%s  AP50=%s\n' "$1" \
        "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
        "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
}

O=outputs
D=$O/deimv2_dinov3_l_aic_rdtdiffuse/best_stg2.pth
R=$O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
F=$O/deimv2_dinov3_l_aic_diffuse_4ch/best_stg2.pth
G=$O/deimv2_dinov3_l_aic/best_stg2.pth

echo "===== FLIR: 5ch RDTDiffuse hybrid (68ep) ====="
run flir_rdtdiffuse_full     deimv2_dinov3_l_aic_rdtdiffuse_flir.yml               $D
run flir_rdtdiffuse_noir     deimv2_dinov3_l_aic_rdtdiffuse_flir_val_featnoir.yml  $D
run flir_rdtdiffuse_noaux    deimv2_dinov3_l_aic_rdtdiffuse_flir_val_featnoaux.yml $D

echo "===== FLIR: 5ch RDTTrack prompt (68ep) ====="
run flir_rdttrack_full       deimv2_dinov3_l_aic_rdttrack_heat_flir.yml                $R
run flir_rdttrack_noir       deimv2_dinov3_l_aic_rdttrack_heat_flir_val_featnoir.yml   $R
run flir_rdttrack_noaux      deimv2_dinov3_l_aic_rdttrack_heat_flir_val_featnoaux.yml  $R

echo "===== FLIR: 4ch DiffFuse (50ep) ====="
run flir_diffuse4ch_full     deimv2_dinov3_l_aic_diffuse_4ch_flir.yml                $F
run flir_diffuse4ch_noir     deimv2_dinov3_l_aic_diffuse_4ch_flir_val_featnoir.yml   $F

echo "===== FLIR: RGB-only (3ch) baseline ====="
run flir_rgb3ch_full         deimv2_dinov3_l_aic_rgb3ch_flir.yml                     $G

P=$O/deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt/best_stg2.pth
B=$O/deimv2_dinov3_l_aic_rdtdiffuse_30ep_both/best_stg2.pth
echo "===== FLIR: 30ep prompt (inject=prompt) ====="
run flir_prompt_full   deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt_flir.yml          $P
run flir_prompt_noir   deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt_flir_val_featnoir.yml  $P
run flir_prompt_noaux  deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt_flir_val_featnoaux.yml $P
echo "===== FLIR: 30ep both (inject=both) ====="
run flir_both_full     deimv2_dinov3_l_aic_rdtdiffuse_30ep_both_flir.yml           $B
run flir_both_noir     deimv2_dinov3_l_aic_rdtdiffuse_30ep_both_flir_val_featnoir.yml   $B
run flir_both_noaux    deimv2_dinov3_l_aic_rdtdiffuse_30ep_both_flir_val_featnoaux.yml  $B

echo "[$(ts)] FLIR FEAT-ABLATION COMPLETE"
