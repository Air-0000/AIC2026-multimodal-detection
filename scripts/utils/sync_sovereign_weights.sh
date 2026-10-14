#!/usr/bin/env bash
# ==============================================================================
# Sovereign V20/V21 Checkpoint Synchronizer
# Downloads all latest Sovereign checkpoints from SeetaCloud RTX 6000 to local.
# ==============================================================================
set -e

DEST_DIR="/home/constant/CodeBuddy/Dev/AIC/weights/sovereign_checkpoints"
mkdir -p "$DEST_DIR"

declare -A WEIGHTS=(
  ["deim_hgnetv2_x_aic_sovereign_pure_lmm_soup/best_stg1.pth"]="sovereign_v20_pure_lmm_soup_585.pth"
  ["deim_hgnetv2_x_aic_sovereign_pure_soup/best_stg1.pth"]="sovereign_pure_soup.pth"
  ["deim_hgnetv2_x_aic_sovereign_v14_soup/best_stg1.pth"]="sovereign_v14_soup.pth"
  ["deim_hgnetv2_x_aic_sovereign_lmm_rtx6000/best_stg1.pth"]="sovereign_lmm_rtx6000.pth"
  ["deim_hgnetv2_x_aic_sovereign_pure_rtx6000/best_stg1.pth"]="sovereign_pure_rtx6000.pth"
  ["deim_hgnetv2_x_aic_sovereign_v14_rtx6000/best_stg1.pth"]="sovereign_v14_rtx6000.pth"
  ["deim_hgnetv2_x_aic_sovereign_v12_rtx6000/best_stg1.pth"]="sovereign_v12_rtx6000.pth"
)

REMOTE_PREFIX="/root/autodl-tmp/AIC/vendor/DEIMv2/outputs"

echo "======================================================================"
echo "Sovereign Checkpoint Sync Starting..."
echo "Target: $DEST_DIR"
echo "======================================================================"

for remote_subpath in "${!WEIGHTS[@]}"; do
  local_name="${WEIGHTS[$remote_subpath]}"
  target_file="$DEST_DIR/$local_name"
  
  if [ -f "$target_file" ]; then
    local_size=$(stat -c%s "$target_file" 2>/dev/null || echo 0)
    # If file is larger than 800MB, check if complete
    if [ "$local_size" -gt 800000000 ]; then
      echo "[SKIP] $local_name already exists ($(( local_size / 1024 / 1024 )) MB)."
      continue
    fi
  fi
  
  echo ">>> [DOWNLOADING] $remote_subpath -> $local_name ..."
  rsync -avP --partial "seeta:$REMOTE_PREFIX/$remote_subpath" "$target_file"
  echo "[DONE] $local_name downloaded."
done

echo "======================================================================"
echo "All Sovereign Checkpoints Synchronized Successfully!"
ls -lh "$DEST_DIR"
echo "======================================================================"
