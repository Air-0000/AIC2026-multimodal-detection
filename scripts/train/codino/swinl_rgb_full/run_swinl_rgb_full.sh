#!/usr/bin/env bash
# Co-DINO Swin-L, pure RGB, all 2000 images: the teammate's s0918 run (Phase 3 56.744; config sha 9862518c without the
# launcher line) with only the training images changed from his fused PNGs (/root/fused_all2000) to visible-only PNGs
# (/root/rgb_all2000, same names and annotation files; dump_rgb_images.py). Same env (venv_codino = his card-A pins),
# seed 20260918, 12 epochs, lr 1e-4, fp16 AMP, lr x0.1 at epoch 11, model-only checkpoints.
# Launch: ( setsid nohup bash run_swinl_rgb_full.sh > /autodl-fs/data/lfb_runs/swinl_rgb_full_s20260918.log 2>&1 < /dev/null & )
set -uo pipefail
MM=/root/tm/src/onedl-mmdetection
PY=/root/venv_codino/bin/python
R=/autodl-fs/data/lfb_runs/swinl_rgb_full_s20260918
CFG=$R/codino_aic_full_rgb_s20260918.py
cd "$MM"
export PYTHONPATH="$MM:/root/tm/setup"
echo "=== train start $(date '+%F %T') cfg_sha $(sha256sum "$CFG" | cut -c1-16) data_manifest_sha $(sha256sum /root/rgb_all2000/manifest.json | cut -c1-16) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$PY tools/train.py "$CFG" --work-dir "$R"
rc=$?
echo "=== train exit $rc $(date '+%F %T') ==="
[ $rc -ne 0 ] && exit $rc
CK=$(ls -1 "$R"/epoch_*.pth 2>/dev/null | sort -V | tail -1)
echo "final ckpt: $CK"
sha256sum "$CK"
echo "=== all done $(date '+%F %T') ==="
