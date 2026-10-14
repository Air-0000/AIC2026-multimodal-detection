#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Controlled A/B on the `inject` switch at the SAME recipe as the existing 30ep
# diffuse run (30ep, batch 4, heat IR+depth). Answers the "can we combine the
# prompt path and the diffuse path?" question:
#     prompt  vs  diffuse (already have: 0.345 / aux -12.2%)  vs  both
# Each variant trains, then gets the three feature-space ablations.
set -u
DEIM=/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
PY=/home/constant/miniconda3/envs/deimv2/bin/python
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" >> /tmp/inject_ab_progress.log; }

cd "$DEIM" || exit 1
mkdir -p outputs

train_and_ablate() {   # train_and_ablate <variant> <cfg> <out_dir_name>
    local V="$1" C="$2" D="$3"
    log "==== train $V ===="
    $PY train.py -c "configs/deimv2/$C" --use-amp > "/tmp/inj_${V}_train.log" 2>&1
    log "train $V exit=$?"

    local CK="$DEIM/outputs/$D/best_stg2.pth"
    [ -f "$CK" ] || CK="$DEIM/outputs/$D/best_stg1.pth"
    if [ ! -f "$CK" ]; then log "$V NO CHECKPOINT -> skip"; return; fi
    log "$V ckpt=$CK"

    $PY train.py -c "configs/deimv2/${C%.yml}.yml"            -r "$CK" --use-amp --test-only > "/tmp/inj_${V}_base.log"    2>&1
    $PY train.py -c "configs/deimv2/${C%.yml}_val_featnoir.yml"   -r "$CK" --use-amp --test-only > "/tmp/inj_${V}_noir.log"    2>&1
    $PY train.py -c "configs/deimv2/${C%.yml}_val_featnodepth.yml" -r "$CK" --use-amp --test-only > "/tmp/inj_${V}_nodepth.log" 2>&1
    $PY train.py -c "configs/deimv2/${C%.yml}_val_featnoaux.yml"  -r "$CK" --use-amp --test-only > "/tmp/inj_${V}_noaux.log"   2>&1

    for k in base noir nodepth noaux; do
        printf '  %-8s %-9s AP=%s\n' "$V" "$k" \
            "$(grep -aF 'IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' "/tmp/inj_${V}_${k}.log" | head -1 | sed 's/.*\] = //' | tr -d '\r')" \
            >> /tmp/inject_ab_progress.log
    done
    log "==== $V done ===="
}

train_and_ablate prompt deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt.yml deimv2_dinov3_l_aic_rdtdiffuse_30ep_prompt
train_and_ablate both   deimv2_dinov3_l_aic_rdtdiffuse_30ep_both.yml   deimv2_dinov3_l_aic_rdtdiffuse_30ep_both

log "INJECT-AB COMPLETE"
