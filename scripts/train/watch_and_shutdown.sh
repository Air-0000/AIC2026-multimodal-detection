#!/usr/bin/env bash
# =============================================================================
# Watchdog for clean baseline training. Waits for the given train.py PID to exit, then shuts the machine
# down: about 35 s after exit when FINAL_PTH exists, and about 5 min after exit when it does not (failed
# or interrupted runs power off too, to save instance cost). Skips the shutdown if another train.py is running.
# Usage: [ROOT=...] [FINAL_PTH=.../final.pth] bash scripts/train/watch_and_shutdown.sh <train.py PID>
# =============================================================================
set -u

PID=${1:?usage: watch_and_shutdown.sh <train.py PID>}
ROOT=${ROOT:-/root/autodl-tmp/AIC_clean_v1}
FINAL_PTH=${FINAL_PTH:-/autodl-fs/data/clean_v1/outputs/aic_clean_baseline_v1/final.pth}
LOG="$ROOT/logs/auto_shutdown.log"

mkdir -p "$ROOT/logs"
# A wrong or stale PID would otherwise count as "finished" at once and power the machine off.
if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID is not running; refusing to start the watchdog." >&2
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog started. Monitoring PID $PID for completion..." | tee -a "$LOG"

while kill -0 "$PID" 2>/dev/null; do
    sleep 20
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $PID has finished." | tee -a "$LOG"

# Wait 15 seconds to ensure all file writes and flushes are complete
sleep 15
sync

# Check if another train.py was manually started in the meantime
if pgrep -f "train.py -c configs/deimv2" | grep -vx "$PID" >/dev/null 2>&1; then
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
