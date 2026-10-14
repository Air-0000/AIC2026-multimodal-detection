#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Mirror-download variant: waits for work_dir/M3FD_raw/M3FD_mirror.zip (the
# HTTP/1.1 hf-mirror pull that is actually stable) to reach full size + pass
# unzip -t, then unzips, runs prepare_m3fd.py, derives M3FD eval configs from
# the FLIR ones, and launches run_m3fd_feat_ablation.sh. The original
# m3fd_autocomplete.sh watches the slow main-site M3FD.zip; this one watches
# the fast mirror file so the eval actually fires ~80min from now.
set -u
AIC=/home/constant/CodeBuddy/Dev/AIC
cd "$AIC" || exit 1
Z=work_dir/M3FD_raw/M3FD_mirror.zip
TARGET=6328099404
PY=/home/constant/miniconda3/envs/deimv2/bin/python
CFG=vendor/DEIMv2/configs/deimv2
log(){ echo "[$(date +%T)] $*"; }
while true; do
    sz=$(stat -c %s "$Z" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$TARGET" ]; then
        if unzip -t "$Z" >/dev/null 2>&1; then
            log "M3FD_mirror.zip intact ($((sz/1024/1024)) MiB)"
            if [ -z "$(find work_dir/M3FD_raw -name instances_val.json 2>/dev/null | head -1)" ]; then
                log "extracting..."
                unzip -q -o "$Z" -d work_dir/M3FD_raw
            fi
            coco=$(find work_dir/M3FD_raw -name instances_val.json 2>/dev/null | head -1)
            if [ -z "$coco" ]; then
                log "ERROR: no instances_val.json after unzip; top of zip:"; unzip -l "$Z" | head -20; exit 2
            fi
            log "prepare_m3fd.py (coco=$coco)"
            $PY prepare_m3fd.py --m3fd work_dir/M3FD_raw --out work_dir/M3FD 2>&1 | tee /tmp/m3fd_prepare.log
            log "derive configs FLIR->M3FD"
            for f in $CFG/*_flir.yml; do sed 's/FLIR/M3FD/g' "$f" > "${f%_flir.yml}_m3fd.yml"; done
            for f in $CFG/*_flir_val_feat*.yml; do sed 's/FLIR/M3FD/g' "$f" > "${f/_flir_val_feat/_m3fd_val_feat}"; done
            sed 's/flir/m3fd/g' run_flir_feat_ablation.sh > run_m3fd_feat_ablation.sh
            nohup bash /home/constant/CodeBuddy/Dev/AIC/scripts/train/run_m3fd_feat_ablation.sh >/tmp/m3fd_run.log 2>&1 &
            log "M3FD eval launched pid=$!"
            break
        else
            log "size reached but zip fails unzip -t; curl may still be writing, retry later"
        fi
    fi
    sleep 120
done
