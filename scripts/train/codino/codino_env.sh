# Shared settings for the Co-DINO scripts in this folder; sourced, not run.
# MMDET_DIR is fixed because the training configs reach the upstream Co-DETR configs by a relative _base_ path, and
# the checkpoint paths are fixed because the configs load them relative to ROOT.
ROOT=${ROOT:?set ROOT to the repo checkout}
ENV_PREFIX=${ENV_PREFIX:-/root/autodl-tmp/envs/codino}
PY=${PY:-$ENV_PREFIX/bin/python}
MMDET_DIR=$ROOT/vendor/mmdetection
CODINO_DIR=$ROOT/scripts/train/codino
# Swin-L: OpenMMLab's Objects365->COCO checkpoint, loaded as is.
SWIN_CKPT=$ROOT/weights/codino/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth
# ViT-L: Sense-X's Objects365->COCO checkpoint (huggingface.co/zongzhuofan/co-detr-vit-large-coco, sha256 733d2ccd...)
# and its conversion for the mmdet 3.x port (convert_codino_vit_ckpt.py), which the ViT-L configs load.
VIT_CKPT_SRC=$ROOT/weights/codino/co_detr_vit_large_coco_hf.pth
VIT_CKPT=$ROOT/weights/codino/co_dino_vit_l_o365tococo_mmdet3.pth

# Only mmdetection (for projects.CO-DETR) and this folder (aic_codino_*); an inherited PYTHONPATH pointing at
# vendor/DEIMv2 is not wanted in this env.
export PYTHONPATH="$MMDET_DIR:$CODINO_DIR"
# Packages in ~/.local (pip --user) would otherwise shadow the env's own (seen on the workstation: setuptools,
# pytorch-lightning).
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# mmengine 0.10.7 calls torch.load without weights_only, and torch >= 2.6 defaults it to True, which rejects
# mmengine's own resume checkpoints (message hub objects, numpy arrays).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Training config per MODEL (swin_l or vit_l).
codino_config() {
  case "$1" in
    swin_l) echo co_dino_swin_l_aic_rgb_1800 ;;
    vit_l) echo co_dino_vit_l_aic_rgb_1800 ;;
    *) echo "MODEL must be swin_l or vit_l" >&2; return 1 ;;
  esac
}

# Runs mmdetection's tools/train.py on NGPUS GPUs (torchrun when more than one) with the given arguments.
codino_train() {
  if [ "${NGPUS:-1}" -gt 1 ]; then
    "$PY" -m torch.distributed.run --nproc_per_node="$NGPUS" --master_port="${MASTER_PORT:-29517}" \
      "$MMDET_DIR/tools/train.py" "$@" --launcher pytorch
  else
    "$PY" -u "$MMDET_DIR/tools/train.py" "$@"
  fi
}

# accumulative_counts that keeps 4 images per optimizer step at 1 image per GPU.
codino_accum() {
  case "${NGPUS:-1}" in
    1) echo 4 ;; 2) echo 2 ;; 4) echo 1 ;;
    *) echo "NGPUS must be 1, 2 or 4 (4 images per step; other counts need a new lr)" >&2; return 1 ;;
  esac
}
