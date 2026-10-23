#!/usr/bin/env bash
# =============================================================================
# Watchdog for clean baseline training. Automatically shuts down the machine
# when training completes 50 epochs (verified by final.pth or clean exit).
# =============================================================================
set -u

PID=${1:-11163}
ROOT=${ROOT:-/root/autodl-tmp/AIC_clean_v1}
FINAL_PTH="/autodl-fs/data/clean_v1/outputs/aic_clean_baseline_v1/final.pth"
LOG="$ROOT/logs/auto_shutdown.log"

mkdir -p "$ROOT/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog started. Monitoring PID $PID for completion..." | tee -a "$LOG"

while kill -0 "$PID" 2>/dev/null; do
    sleep 20
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $PID has finished." | tee -a "$LOG"

# Wait 15 seconds to ensure all file writes and flushes are complete
sleep 15
sync

# Check if another train.py was manually started in the meantime
if pgrep -f "train.py -c configs/deimv2" | grep -v "$PID" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another training process detected. Aborting shutdown." | tee -a "$LOG"
    exit 0
fi

if [ -f "$FINAL_PTH" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: 50 epochs completed! Verified $FINAL_PTH exists." | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing disks and initiating shutdown in 20 seconds..." | tee -a "$LOG"
    sleep 20
    sync
    if [ -x /usr/bin/shutdown ]; then
        /usr/bin/shutdown
    else
        poweroff
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: PID $PID exited but $FINAL_PTH not found." | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Grace period: waiting 300 seconds before shutdown to save instance cost..." | tee -a "$LOG"
    sleep 300
    sync
    if [ -x /usr/bin/shutdown ]; then
        /usr/bin/shutdown
    else
        poweroff
    fi
fi
