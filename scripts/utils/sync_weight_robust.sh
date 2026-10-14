#!/usr/bin/env bash
# Robust checkpoint sync with auto-resume on broken pipe

SRC="/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/sovereign_v30_cssp_fast8ep/best_stg2.pth"
DEST="ailab40:/home/ailab/ssd/bigdata/liuzhengyang/weights/sovereign_v30_cssp_best_stg2.pth"

echo "[*] Resuming checkpoint sync to ailab40..."
MAX_RETRIES=15
count=0

while [ $count -lt $MAX_RETRIES ]; do
    count=$((count+1))
    echo ">>> Sync attempt $count of $MAX_RETRIES..."
    if rsync -avP --partial --append-verify \
        -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes" \
        "$SRC" "$DEST"; then
        echo "[+] Checkpoint sync SUCCESSFUL!"
        exit 0
    else
        echo "[-] Connection dropped, auto-reconnecting in 3s..."
        sleep 3
    fi
done

echo "[-] Failed after $MAX_RETRIES attempts."
exit 1
