"""
Train a DINOv3-backbone RF-DETR for the multi-modal detection challenge.

This reuses everything from RF-DETR's training pipeline but swaps the DINOv2
encoder for a DINOv3 ViT-L/16 (lvd1689m pretrain) via a small monkeypatch:
  * build_backbone -> builds a DinoV3Backbone (see vendor/dinov3_backbone.py)
    instead of the stock DINOv2 one when the encoder name starts with "dinov3".
  * interpolate_position_embeddings -> strips the DINOv2 encoder keys from the
    base checkpoint so they don't overwrite the separately-loaded DINOv3 weights.

Usage:
    python3 train_rgb_dinov3.py [EPOCHS] [OUTPUT_SUFFIX] [BATCH] [FREEZE_BLOCKS] [--gc]

Defaults: EPOCHS=80, OUTPUT_SUFFIX=rgb_baseline_dinov3_b, BATCH=auto, FREEZE_BLOCKS=20
  (trainable blocks = 24 - FREEZE_BLOCKS; lower FREEZE_BLOCKS = unfreeze more layers)
"""
import sys
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor"))

from rfdetr.detr import RFDETR
from rfdetr.config import RFDETRBaseConfig
from rfdetr.models.backbone import build_backbone as _orig_build_backbone, Joiner
from rfdetr.models.backbone import build_position_encoding
from rfdetr.models.weights import interpolate_position_embeddings as _orig_interp
from rfdetr.utilities.state_dict import validate_checkpoint_compatibility as _orig_validate
import rfdetr.models.backbone as _bb
import rfdetr.models.lwdetr as _lwdetr
import rfdetr.models.weights as _weights

import dinov3_backbone as dv3

DINOV3_WEIGHTS = "vendor/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# Projector / decoder warm-start source. The DINOv3 encoder is loaded directly
# from DINOV3_WEIGHTS; the rf-detr decoder+projector need a pretrained init too.
# Candidates (all share the base MultiScaleProjector structure, 26 tensors,
# cv1=(256,1536,1,1), so they transfer 1:1):
#   - rf-detr-base-o365.pth     : Objects365 pretrain, 366 classes
#   - rf-detr-large-2026.pth    : newer (2026) pretrain, 91 classes, best match for
#                                 a 12-class urban detection task; DEFAULT.
#   - rf-detr-large.pth         : INCOMPATIBLE (different projector arch, 72 tensors)
PRETRAIN = "vendor/rfdetr_models/rf-detr-large-2026.pth"
assert os.path.exists(DINOV3_WEIGHTS), f"DINOv3 weights missing: {DINOV3_WEIGHTS}"
assert os.path.exists(PRETRAIN), f"pretrain weights missing: {PRETRAIN}"

# Number of the 24 DINOv3 transformer blocks to KEEP FROZEN (from the front).
# Trainable blocks = 24 - FREEZE_BLOCKS. 20 -> last 4 (conservative), 16 -> last 8,
# 12 -> last 12, 0 -> full fine-tune. Lower = more adaptation, more risk of
# destroying the pretrain. Pass via CLI 4th positional arg.
FREEZE_BLOCKS = 20
PRETRAIN = "vendor/rfdetr_models/rf-detr-large-2026.pth"
assert os.path.exists(DINOV3_WEIGHTS), f"DINOv3 weights missing: {DINOV3_WEIGHTS}"
assert os.path.exists(PRETRAIN), f"pretrain weights missing: {PRETRAIN}"

_USE_DINOV3 = False  # flipped True once patches are installed


# --------------------------------------------------------------------------- #
# Monkeypatch build_backbone: route dinov3 encoder names to DinoV3Backbone.    #
# --------------------------------------------------------------------------- #
def _patched_build_backbone(
    encoder, vit_encoder_num_layers, pretrained_encoder, window_block_indexes,
    drop_path, out_channels, out_feature_indexes, projector_scale, use_cls_token,
    hidden_dim, position_embedding, freeze_encoder, layer_norm, target_shape,
    rms_norm, backbone_lora, force_no_pretrain, gradient_checkpointing,
    load_dinov2_weights, **kwargs,
):
    if str(encoder).startswith("dinov3"):
        bb = dv3.DinoV3Backbone(
            name=str(encoder),
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            patch_size=16,
            num_windows=1,
            positional_encoding_size=35,
            gradient_checkpointing=gradient_checkpointing,
            weights_path=DINOV3_WEIGHTS,
            # Partial unfreeze: keep the first FREEZE_BLOCKS/24 DINOv3 blocks frozen
            # (preserve the strong pretrain), train only the last blocks + projector + decoder.
            freeze_blocks=FREEZE_BLOCKS,
            # Warm-start the projector from the base(o365) checkpoint so every part
            # has a pretrained init (its 768-dim projector is interpolated to 1024).
            init_projector_from=PRETRAIN,
        )
        pos = build_position_encoding(hidden_dim, position_embedding)
        return Joiner(bb, pos)
    return _orig_build_backbone(
        encoder, vit_encoder_num_layers, pretrained_encoder, window_block_indexes,
        drop_path, out_channels, out_feature_indexes, projector_scale, use_cls_token,
        hidden_dim, position_embedding, freeze_encoder, layer_norm, target_shape,
        rms_norm, backbone_lora, force_no_pretrain, gradient_checkpointing,
        load_dinov2_weights, **kwargs,
    )


_bb.build_backbone = _patched_build_backbone
_lwdetr.build_backbone = _patched_build_backbone


# --------------------------------------------------------------------------- #
# Monkeypatch interpolate_position_embeddings: drop DINOv2 encoder keys so the  #
# separately-loaded DINOv3 encoder survives the base checkpoint load.          #
# --------------------------------------------------------------------------- #
def _patched_interp(checkpoint_state, pe_size):
    if _USE_DINOV3:
        # The base(o365) checkpoint carries a DINOv2 encoder + a projector wired
        # for DINOv2's hidden dim (768 -> 1536). DINOv3 has dim 1024 (->4096), so
        # both must be dropped; the encoder is loaded separately and the projector
        # is freshly initialized (it adapts 1024->256 and is trained from scratch).
        for k in [k for k in list(checkpoint_state)
                  if k.startswith("backbone.0.encoder") or k.startswith("backbone.0.projector")]:
            del checkpoint_state[k]
    return _orig_interp(checkpoint_state, pe_size)


_weights.interpolate_position_embeddings = _patched_interp
_lwdetr.interpolate_position_embeddings = _patched_interp
_USE_DINOV3 = True


# --------------------------------------------------------------------------- #
# Monkeypatch validate_checkpoint_compatibility: the base(o365) checkpoint is   #
# DINOv2 (patch_size 14) and would raise on our patch_size=16. Since we strip   #
# the DINOv2 encoder and load DINOv3 separately, neutralize the patch_size      #
# compatibility check for DINOv3 runs.                                          #
# --------------------------------------------------------------------------- #
def _patched_validate(checkpoint, model_args):
    if _USE_DINOV3:
        model = checkpoint.get("model", {})
        # drop the DINOv2 patch-projection key the fallback uses to infer patch_size
        model.pop(
            "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
            None,
        )
        # neutralize an explicit args.patch_size if present
        args = checkpoint.get("args")
        if isinstance(args, dict) and "patch_size" in args:
            args["patch_size"] = getattr(model_args, "patch_size", 16)
    return _orig_validate(checkpoint, model_args)


import rfdetr.utilities.state_dict as _sd

_sd.validate_checkpoint_compatibility = _patched_validate
# load_pretrain_weights imports validate_checkpoint_compatibility directly into
# rfdetr.models.weights, so patch that binding too.
_weights.validate_checkpoint_compatibility = _patched_validate


# --------------------------------------------------------------------------- #
# Model config that accepts a free-form encoder name (pydantic Literal bypass). #
# DINOv3 is patch_size=16 -> 560/16 = 35x35 grid; positional_encoding_size must  #
# match so the Joiner's sinusoidal PE aligns with the feature grid.              #
# --------------------------------------------------------------------------- #
class DinoV3Config(RFDETRBaseConfig):
    encoder: str = "dinov3_windowed_large"
    patch_size: int = 16
    positional_encoding_size: int = 35


class RFDETRDinoV3(RFDETR):
    _model_config_class = DinoV3Config


if __name__ == "__main__":
    args = sys.argv[1:]
    GC = "--gc" in args
    args = [a for a in args if a != "--gc"]
    EPOCHS = int(args[0]) if len(args) > 0 else 80
    OUT = args[1] if len(args) > 1 else "rgb_baseline_dinov3_b"
    BATCH = args[2] if len(args) > 2 else "auto"
    if len(args) > 3:
        FREEZE_BLOCKS = int(args[3])   # blocks kept frozen (trainable = 24 - FREEZE_BLOCKS)

    print(f"[train-dinov3] epochs={EPOCHS} out={OUT} batch={BATCH} "
          f"freeze_blocks={FREEZE_BLOCKS} (trainable={24 - FREEZE_BLOCKS}/24) "
          f"backbone=DINOv3-ViT-L/16 gc={GC}")

    model = RFDETRDinoV3(
        encoder="dinov3_windowed_large",
        pretrain_weights=PRETRAIN,
        resolution=560,
        gradient_checkpointing=GC,
    )

    model.train(
        dataset_dir="work_dir/rfdetr_dataset",
        output_dir=f"work_dir/{OUT}",
        batch_size=BATCH,
        epochs=EPOCHS,
        grad_accum_steps=4,
        num_workers=4,
        tensorboard=False,
        project="rgb_baseline_dinov3",
        run=OUT,
    )

    print(f"[train-dinov3] finished -> work_dir/{OUT}")
