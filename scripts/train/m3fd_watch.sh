#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# Watch for a COMPLETE M3FD_Detection.zip (Google-Drive throttled; the patient
# downloader retries until the quota lifts). Once the zip verifies, auto-build
# the OOD test set and queue its feature-ablation eval behind everything else
# (run_inject_ab + run_llvip + run_night + run_flir + any train.py). Pure CPU
# here; does not touch the GPU until the eval stage.
set -u
AIC=/home/constant/CodeBuddy/Dev/AIC
Z=$AIC/work_dir/M3FD_raw/M3FD_Detection.zip
cd "$AIC" || exit 1

while true; do
    if unzip -t "$Z" >/dev/null 2>&1; then
        echo "[$(date +%T)] M3FD zip intact -> build test set + queue eval"
        /home/constant/miniconda3/envs/deimv2/bin/python prepare_m3fd.py \
            --m3fd work_dir/M3FD_raw --out work_dir/M3FD 2>&1 | tee /tmp/m3fd_prepare.log
        # derive M3FD eval configs from the already-generated FLIR ones (FLIR->M3FD)
        for f in vendor/DEIMv2/configs/deimv2/*_flir.yml; do
            sed 's/FLIR/M3FD/g' "$f" > "${f%_flir.yml}_m3fd.yml"
        done
        for f in vendor/DEIMv2/configs/deimv2/*_flir_val_feat*.yml; do
            sed 's/FLIR/M3FD/g' "$f" > "${f/_flir_val_feat/_m3fd_val_feat}"
        done
        sed 's/flir/m3fd/g' run_flir_feat_ablation.sh > run_m3fd_feat_ablation.sh
        nohup bash /home/constant/CodeBuddy/Dev/AIC/scripts/train/run_m3fd_feat_ablation.sh >/tmp/m3fd_run.log 2>&1 &
        echo "[$(date +%T)] M3FD eval launched pid=$!"
        break
    fi
    sleep 300
done
