"""Converts Sense-X's released Co-DETR ViT-L checkpoint (mmdet 2.25.3) for the mmdet 3.x port and checks it.

The check builds the 80-class COCO version of co_dino_vit_l_aic_rgb_1800.py and requires every tensor of it to be
present in the converted file with the same shape, and nothing left over; it exits non-zero otherwise. Changes:
  - drops the ema_* tensors mmdet 2's EMA hook keeps beside the weights (the main tensors hold the EMA weights,
    swapped in before saving, which the released scores come from), after checking there is one per weight;
  - moves query_head.label_embedding into query_head.dn_generator (mmdet 3's CdnQueryGenerator owns it).
The output carries COCO class names as dataset_meta, so mmdet's inferencer labels its boxes.

Run from the repo root with codino_env.sh sourced:
    python scripts/train/codino/convert_codino_vit_ckpt.py weights/codino/co_detr_vit_large_coco_hf.pth \
        weights/codino/co_dino_vit_l_o365tococo_mmdet3.pth
"""
import argparse
import hashlib
import sys

import torch

CONFIG = 'scripts/train/codino/co_dino_vit_l_aic_rgb_1800.py'
RENAMES = {'query_head.label_embedding.weight': 'query_head.dn_generator.label_embedding.weight'}


def build_coco_model(config):
    """The model of `config` with the 80 COCO classes, on CPU, without loading load_from."""
    from mmengine.config import Config
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(config)
    cfg.model.query_head.num_classes = 80
    cfg.model.roi_head[0].bbox_head.num_classes = 80
    cfg.model.bbox_head[0].num_classes = 80
    return MODELS.build(cfg.model)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    print(f'source keys: {sorted(ckpt)}' if isinstance(ckpt, dict) else type(ckpt))
    state = ckpt.get('state_dict', ckpt)
    ema = {k for k in state if k.startswith('ema_')}
    weights = {k: v for k, v in state.items() if not k.startswith('ema_')}
    print(f'{len(weights)} tensors, {len(ema)} ema_ copies dropped')
    if ema and len(ema) != sum(v.is_floating_point() for v in weights.values()):
        print(f'NOTE: {len(ema)} ema_ tensors for {sum(v.is_floating_point() for v in weights.values())} float '
              'weights (mmdet 2 also tracks float buffers)')
    out = {RENAMES.get(k, k): v for k, v in weights.items()}

    ref = build_coco_model(args.config).state_dict()
    missing = sorted(set(ref) - set(out))
    unexpected = sorted(set(out) - set(ref))
    mismatched = sorted(k for k in set(ref) & set(out) if ref[k].shape != out[k].shape)
    print(f'model tensors {len(ref)}, converted {len(out)}: missing {len(missing)}, unexpected {len(unexpected)}, '
          f'shape mismatches {len(mismatched)}')
    for title, keys in (('missing', missing), ('unexpected', unexpected), ('shape mismatch', mismatched)):
        for k in keys[:40]:
            detail = f' model {tuple(ref[k].shape)} ckpt {tuple(out[k].shape)}' if title == 'shape mismatch' else ''
            print(f'  {title}: {k}{detail}')
    if missing or unexpected or mismatched:
        sys.exit('FAIL: converted state dict does not match the model')
    # The window-attention RoPE tables are buffers upstream saved; the port recomputes them at construction. The
    # released tables differ from freshly computed ones by ~1e-4 (both the EMA and the raw copy, and from each
    # other by ~5e-5): mmdet 2's EMA hook averages buffers too and swaps EMA and model every epoch, so float32
    # rounding drifts the constant tables. A wrong formula differs by order 1; loading keeps the released values.
    rope = [k for k in ref if '.freqs_' in k]
    rope_diff = max((ref[k] - out[k]).abs().max().item() for k in rope)
    print(f'RoPE tables: {len(rope)} buffers, max |port - checkpoint| {rope_diff:.2e}')
    if rope_diff > 1e-3:
        sys.exit('FAIL: the port computes different RoPE tables than upstream saved')

    from mmdet.datasets import CocoDataset
    meta = dict(dataset_meta=dict(classes=CocoDataset.METAINFO['classes']),
                converted_from=args.src, source_sha256=sha256(args.src))
    torch.save(dict(state_dict=out, meta=meta), args.dst)
    print(f'saved {args.dst} sha256 {sha256(args.dst)}')


if __name__ == '__main__':
    main()
