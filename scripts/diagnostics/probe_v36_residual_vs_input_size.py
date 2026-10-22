#!/usr/bin/env python3
"""Same image, same weights, multiple input resolutions: fusion residual vs trunk.

Reports global RMS ratio and the per-spatial-position ratio distribution
(median / p95), plus per-stage figures. Frozen V36 weights, CPU FP32, read-only.
Residual magnitude says nothing about AP gain.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'vendor/DEIMv2')]

MEAN = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
STD = [0.229, 0.224, 0.225, 0.5, 0.5, 0.5]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--image-id', type=int, required=True)
    p.add_argument('--sizes', nargs='+', required=True, help='HxW, e.g. 720x1280 1440x2560')
    args = p.parse_args()

    from engine.data.dataset.coco_dataset_mm import CocoDetectionMultiModal
    from engine.backbone.hgnetv2_mm import HGNetv2MultiModal

    ds = CocoDetectionMultiModal(
        img_folder=str(ROOT / 'dataset/train/visible'), ann_file=str(args.annotations),
        ir_folder=str(ROOT / 'dataset/train/infrared_aligned'),
        raw_ir_folder=str(ROOT / 'work_dir/audits/v36_mechanism_20261018/train_lbg_ir'),
        depth_folder=str(ROOT / 'dataset/train/depth'),
        n_channels=6, depth_read='raw16', destripe_ir=True)

    model = HGNetv2MultiModal(name='B5', ir_in_chans=2, inject_stages=[2, 3],
                              return_idx=[0, 1, 2, 3], pretrained=False, dcn_stages=[],
                              use_dcda=True, use_dog=True, use_3d_lift=True,
                              pretrained_ir_stem=True, ir_drop_prob=0.05).float().eval()
    sd = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)['model']
    model.load_state_dict({k[len('backbone.'):]: v for k, v in sd.items()
                           if k.startswith('backbone.')}, strict=True)
    del sd

    cap = {}
    handles = []
    for stage_i in (2, 3):
        handles.append(model.rgb_net.stages[stage_i].register_forward_hook(
            lambda m, i, o, s=stage_i: cap.__setitem__(f'trunk_s{s}', o.detach())))
        h = model.ir_res[list((2, 3)).index(stage_i)].register_forward_hook(
            lambda m, i, o, s=stage_i: cap.__setitem__(f'res_s{s}', o.detach()))
        handles.append(h)

    idx = next(i for i in range(len(ds.ids)) if ds.coco.loadImgs(ds.ids[i])[0]['id'] == args.image_id)
    meta = ds.coco.loadImgs(ds.ids[idx])[0]
    img0, _ = ds.load_item(idx)
    mean = torch.tensor(MEAN).view(-1, 1, 1)
    std = torch.tensor(STD).view(-1, 1, 1)

    rows = []
    for size in args.sizes:
        h, w = (int(v) for v in size.lower().split('x'))
        img = F.interpolate(img0[None].float(), size=(h, w), mode='bilinear',
                            align_corners=False, antialias=True)[0]
        x = ((img - mean) / std)[None]
        cap.clear()
        t0 = time.time()
        with torch.no_grad():
            model(x)
        row = {'input': f'{h}x{w}', 'seconds': round(time.time() - t0, 1), 'stages': {}}
        for stage_i in (2, 3):
            y, r = cap[f'trunk_s{stage_i}'], cap[f'res_s{stage_i}']
            ymap = y.pow(2).mean(1).sqrt()[0].flatten()
            rmap = r.pow(2).mean(1).sqrt()[0].flatten()
            ratio = (rmap / ymap.clamp_min(1e-12))
            stride = {2: 16, 3: 32}[stage_i]
            row['stages'][f'stride{stride}'] = {
                'trunk_rms': round(float(y.pow(2).mean().sqrt()), 4),
                'residual_rms': round(float(r.pow(2).mean().sqrt()), 4),
                'global_ratio': round(float(r.pow(2).mean().sqrt() / y.pow(2).mean().sqrt()), 5),
                'pos_ratio_median': round(float(ratio.median()), 5),
                'pos_ratio_p95': round(float(ratio.quantile(0.95)), 5),
                'pos_ratio_max': round(float(ratio.max()), 5),
            }
        rows.append(row)
        print(json.dumps(row))

    for h_ in handles:
        h_.remove()
    out = {'image_id': args.image_id, 'native': f"{meta['width']}x{meta['height']}",
           'checkpoint': str(args.checkpoint), 'precision': 'FP32/CPU',
           'note': 'Residual magnitude is not an AP predictor; per-position ratio = channel-RMS ratio per cell.',
           'runs': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
