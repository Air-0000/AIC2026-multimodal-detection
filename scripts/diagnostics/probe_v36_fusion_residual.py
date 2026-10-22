#!/usr/bin/env python3
"""Per-image ||fusion residual|| / ||trunk|| at each injected stage, frozen V36.

Multiple validation images at the training resolution (1440x2560), CPU FP32.
Read-only. Reports per-image ratios plus a group summary by native resolution.
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
    p.add_argument('--size', type=int, nargs=2, default=[1440, 2560])
    p.add_argument('--image-ids', type=int, nargs='+', required=True)
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
    for k, stage_i in enumerate((2, 3)):
        handles.append(model.ir_res[k].register_forward_hook(
            lambda m, i, o, s=stage_i: cap.__setitem__(f'res_s{s}', o.detach())))

    mean = torch.tensor(MEAN).view(-1, 1, 1)
    std = torch.tensor(STD).view(-1, 1, 1)
    rows = []
    for image_id in args.image_ids:
        idx = next(i for i in range(len(ds.ids)) if ds.coco.loadImgs(ds.ids[i])[0]['id'] == image_id)
        meta = ds.coco.loadImgs(ds.ids[idx])[0]
        img, _ = ds.load_item(idx)
        img = F.interpolate(img[None].float(), size=tuple(args.size), mode='bilinear',
                            align_corners=False, antialias=True)[0]
        x = ((img - mean) / std)[None]
        cap.clear()
        t0 = time.time()
        with torch.no_grad():
            model(x)
        row = {'image_id': image_id, 'native': f"{meta['width']}x{meta['height']}",
               'seconds': round(time.time() - t0, 1)}
        for stage_i in (2, 3):
            y, r = cap[f'trunk_s{stage_i}'], cap[f'res_s{stage_i}']
            row[f'ratio_s{stage_i}'] = round(float(r.pow(2).mean().sqrt() / y.pow(2).mean().sqrt()), 5)
            row[f'absmax_s{stage_i}'] = round(float(r.abs().max()), 4)
        rows.append(row)
        print(json.dumps(row))

    for h in handles:
        h.remove()

    groups = {}
    for row in rows:
        groups.setdefault(row['native'], []).append(row)
    summary = {}
    for native, items in groups.items():
        def stat(key):
            vals = sorted(r[key] for r in items)
            return {'n': len(vals), 'mean': round(sum(vals) / len(vals), 5),
                    'min': vals[0], 'max': vals[-1]}
        summary[native] = {'ratio_s2_stride16_pct': stat('ratio_s2'),
                           'ratio_s3_stride32_pct': stat('ratio_s3')}
    out = {'checkpoint': str(args.checkpoint), 'input_size': args.size,
           'precision': 'FP32/CPU', 'per_image': rows, 'by_native_resolution': summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
