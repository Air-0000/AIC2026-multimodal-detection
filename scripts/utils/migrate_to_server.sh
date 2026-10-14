#!/usr/bin/env bash
# ==============================================================================
# Sovereign V30 Server Migration & Sync Tool
# Packs newly developed CSSP + ABC modules, configs, and scripts,
# and syncs them directly to the remote server via rsync/ssh.
# ==============================================================================
set -e

ROOT="/home/constant/CodeBuddy/Dev/AIC"
cd "$ROOT"

ARCHIVE_DIR="$ROOT/work_dir/archives"
mkdir -p "$ARCHIVE_DIR"
ARCHIVE_TAR="$ARCHIVE_DIR/sovereign_v30_migration_package.tar.gz"

echo "======================================================================"
echo "Sovereign V30: Building Migration Package..."
echo "======================================================================"

# 1. Package core code, configs, and runner scripts (lightweight code package)
tar -czvf "$ARCHIVE_TAR" \
    vendor/DEIMv2/engine/backbone/cssp_module.py \
    vendor/DEIMv2/engine/backbone/hgnetv2_mm.py \
    vendor/DEIMv2/configs/deimv2/sovereign_v30_cssp_fast8ep.yml \
    vendor/DEIMv2/configs/deimv2/sovereign_v30_cssp_2560_remote.yml \
    scripts/train/run_sovereign_v30_remote_2560.sh \
    scripts/train/run_sovereign_v30_cssp_fast8ep.sh \
    scripts/inference/infer_test2_v30_cssp.py \
    scripts/analysis/test_cssp_spectral_saliency.py \
    scripts/diagnostics/test_cssp_module.py \
    docs/MODEL_RECORDS.md \
    outputs/reports/cssp_abc_experimental_evaluation_report.md

echo "[+] Successfully created migration package: $ARCHIVE_TAR ($(( $(stat -c%s "$ARCHIVE_TAR") / 1024 )) KB)"

# 2. Sync to remote server if target provided or accessible
TARGET="${1:-seeta}"
REMOTE_AIC_PATH="${2:-/root/autodl-tmp/AIC}"

echo "\n----------------------------------------------------------------------"
echo "Attempting rsync sync to target: $TARGET -> $REMOTE_AIC_PATH"
echo "----------------------------------------------------------------------"

if ssh -o ConnectTimeout=5 -o BatchMode=yes "$TARGET" "echo test_conn" >/dev/null 2>&1; then
    echo "[+] Remote host $TARGET is reachable! Performing live rsync sync..."
    
    # Sync core files directly into remote git/tree structure
    rsync -avP \
        vendor/DEIMv2/engine/backbone/cssp_module.py \
        vendor/DEIMv2/engine/backbone/hgnetv2_mm.py \
        "$TARGET:$REMOTE_AIC_PATH/vendor/DEIMv2/engine/backbone/"
        
    rsync -avP \
        vendor/DEIMv2/configs/deimv2/sovereign_v30_cssp_2560_remote.yml \
        vendor/DEIMv2/configs/deimv2/sovereign_v30_cssp_fast8ep.yml \
        "$TARGET:$REMOTE_AIC_PATH/vendor/DEIMv2/configs/deimv2/"
        
    rsync -avP \
        scripts/train/run_sovereign_v30_remote_2560.sh \
        scripts/train/run_sovereign_v30_cssp_fast8ep.sh \
        "$TARGET:$REMOTE_AIC_PATH/scripts/train/"
        
    rsync -avP \
        scripts/inference/infer_test2_v30_cssp.py \
        "$TARGET:$REMOTE_AIC_PATH/scripts/inference/"
        
    rsync -avP \
        docs/MODEL_RECORDS.md \
        "$TARGET:$REMOTE_AIC_PATH/docs/"

    echo "\n[+] Live rsync completed successfully on $TARGET!"
    echo "To launch remote training on $TARGET, run:"
    echo "  ssh $TARGET 'bash $REMOTE_AIC_PATH/scripts/train/run_sovereign_v30_remote_2560.sh'"
else
    echo "[-] Warning: Host '$TARGET' is currently not directly reachable via SSH (instance stopped or port changed)."
    echo "    The standalone package has been built and ready at:"
    echo "    $ARCHIVE_TAR"
    echo "    Once the server instance is online, simply run:"
    echo "    bash scripts/utils/migrate_to_server.sh <user@host -p port>"
fi
echo "======================================================================"
