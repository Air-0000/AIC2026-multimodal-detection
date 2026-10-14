#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# LLVIP IR ablation with the VALID feature-space protocol.
# The old LLVIP numbers came from the same invalid pixel-black trick, and LLVIP
# carries a null depth channel, so the aux branch there was measured badly too.
# Waits for any running training/eval to finish first (single 15.5GiB card).
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
OUT=/tmp/llvip_featabl
mkdir -p "$OUT"
ts() { date '+%F %T'; }

cd "$DEIM" || exit 1

echo "[$(ts)] waiting for the GPU to be free"
while pgrep -f "train.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] GPU free"

run() {   # run <tag> <config> <ckpt>
    echo "[$(ts)] $1"
    $PY train.py -c "configs/deimv2/$2" -r "$3" --use-amp --test-only \
        > "$OUT/$1.log" 2>&1
    printf '  %-32s AP=%s  AP50=%s\n' "$1" \
        "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
        "$(grep -aF 'IoU=0.50      | area=   all | maxDets=100 ] = ' "$OUT/$1.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')"
}

O=outputs
D=$O/deimv2_dinov3_l_aic_rdtdiffuse/best_stg2.pth
R=$O/deimv2_dinov3_l_aic_rdttrack_heat/best_stg2.pth
F=$O/deimv2_dinov3_l_aic_diffuse_4ch/best_stg2.pth

echo "===== LLVIP: 5ch RDTDiffuse hybrid (68ep) ====="
run llvip_rdtdiffuse_full     deimv2_dinov3_l_aic_rdtdiffuse_llvip.yml              $D
run llvip_rdtdiffuse_noir     deimv2_dinov3_l_aic_rdtdiffuse_llvip_val_featnoir.yml $D
run llvip_rdtdiffuse_noaux    deimv2_dinov3_l_aic_rdtdiffuse_llvip_val_featnoaux.yml $D

echo "===== LLVIP: 5ch RDTTrack prompt (68ep) ====="
run llvip_rdttrack_full       deimv2_dinov3_l_aic_rdttrack_heat_llvip.yml                $R
run llvip_rdttrack_noir       deimv2_dinov3_l_aic_rdttrack_heat_llvip_val_featnoir.yml   $R
run llvip_rdttrack_noaux      deimv2_dinov3_l_aic_rdttrack_heat_llvip_val_featnoaux.yml  $R

echo "===== LLVIP: 4ch DiffFuse (50ep) ====="
run llvip_diffuse4ch_full     deimv2_dinov3_l_aic_diffuse_4ch_llvip.yml                $F
run llvip_diffuse4ch_noir     deimv2_dinov3_l_aic_diffuse_4ch_llvip_val_featnoir.yml   $F

echo "[$(ts)] LLVIP FEAT-ABLATION COMPLETE"
