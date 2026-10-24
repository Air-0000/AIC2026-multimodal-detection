#!/usr/bin/env python3
"""Run a config's real training data pipeline (worker processes, transforms, collate) at chosen epochs; no model.

Checks what the epoch-gated policy and the collate actually produce inside the DataLoader workers: batch shape,
finite values, boxes per image and their range, and the share of mixed-up images (targets carrying 'mixup').
Saves, per epoch, RGB / aligned IR / depth panels of the first images with their boxes, so Mosaic, ZoomOut and
IoUCrop can be seen. CPU only; needs the dataset and the deimv2 environment. OUT must not exist.

Example (repo root), for aic_clean_baseline_v2_aug_all2000.yml (policy [2, 30, 46], mixup [2, 30]):
  python scripts/diagnostics/smoke_aug_pipeline.py --config configs/deimv2/aic_clean_baseline_v2_aug_all2000.yml \
    --epochs 0 2 30 46 --batches 6 --out work_dir/audits/smoke_aug_v2
Expected: no mixup at epochs 0, 30 and 46; at epoch 2 each batch is mixed up with probability 0.5 and about half
the images are mosaics; at epoch 30 ZoomOut / IoUCrop / photometric changes but no Mosaic; at 0 and 46 plain images.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
sys.path[:0] = [str(FW), str(ROOT/'scripts/lib')]


def panel(img, boxes, path, mean, std):
    """Denormalize one [6,H,W] sample, draw cxcywh-normalized boxes on RGB, IR and depth side by side."""
    import numpy as np
    from PIL import Image, ImageDraw
    x = img.float().numpy()
    x = x * np.asarray(std, np.float32)[:, None, None] + np.asarray(mean, np.float32)[:, None, None]
    x = np.clip(x, 0, 1)
    rgb = (x[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
    ir = np.repeat((x[3:4] * 255).astype(np.uint8), 3, 0).transpose(1, 2, 0)
    depth = np.repeat((x[-1:] * 255).astype(np.uint8), 3, 0).transpose(1, 2, 0)
    h, w = rgb.shape[:2]
    canvas = Image.fromarray(np.concatenate([rgb, ir, depth], axis=1))
    draw = ImageDraw.Draw(canvas)
    for cx, cy, bw, bh in boxes.tolist():
        for k in range(3):
            x0, y0 = (cx - bw / 2) * w + k * w, (cy - bh / 2) * h
            draw.rectangle([x0, y0, x0 + bw * w, y0 + bh * h], outline=(255, 0, 0), width=3)
    canvas.thumbnail((3 * 960, 540))
    canvas.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='path relative to vendor/DEIMv2, as for train.py')
    p.add_argument('--epochs', type=int, nargs='+', required=True)
    p.add_argument('--batches', type=int, default=6)
    p.add_argument('--workers', type=int, help='override train num_workers (keep > 0 to exercise worker processes)')
    p.add_argument('--panels', type=int, default=2, help='images saved per epoch')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(FW)
    import torch
    from engine.core import YAMLConfig
    override = {'train_dataloader': {'num_workers': a.workers}} if a.workers is not None else {}
    cfg = YAMLConfig(a.config, **override)
    loader = cfg.train_dataloader
    norm = next(op for op in cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops'] if op['type'] == 'Normalize')
    report = dict(config=a.config, num_workers=loader.num_workers, persistent_workers=loader.persistent_workers,
                  batch_size=loader.batch_size, epochs={})
    for epoch in a.epochs:
        loader.set_epoch(epoch)
        stats = dict(batches=0, images=0, mixed_images=0, boxes=0, boxes_out_of_range=0, nonfinite_batches=0, shapes=[])
        it = iter(loader)
        for b in range(a.batches):
            images, targets = next(it)
            stats['batches'] += 1
            stats['images'] += len(targets)
            stats['shapes'].append(list(images.shape))
            stats['nonfinite_batches'] += int(not torch.isfinite(images).all())
            for i, t in enumerate(targets):
                boxes = t['boxes']
                stats['boxes'] += int(boxes.shape[0])
                stats['boxes_out_of_range'] += int(((boxes < 0) | (boxes > 1)).any(dim=1).sum())
                stats['mixed_images'] += int('mixup' in t)
                n = b * len(targets) + i
                if n < a.panels:
                    panel(images[i], boxes, out/f'epoch{epoch:02d}_img{n}.jpg', norm['mean'], norm['std'])
        del it
        stats['boxes_per_image'] = round(stats['boxes'] / max(stats['images'], 1), 2)
        stats['mixed_share'] = round(stats['mixed_images'] / max(stats['images'], 1), 3)
        report['epochs'][str(epoch)] = stats
        print(f"epoch {epoch}: shapes {stats['shapes'][0]}, boxes/img {stats['boxes_per_image']}, "
              f"mixed {stats['mixed_share']:.0%}, out-of-range boxes {stats['boxes_out_of_range']}, "
              f"non-finite batches {stats['nonfinite_batches']}", flush=True)
    (out/'report.json').write_text(json.dumps(report, indent=1))
    bad = [e for e, s in report['epochs'].items() if s['nonfinite_batches'] or s['boxes_out_of_range']]
    if bad:
        sys.exit(f'FAIL: non-finite images or out-of-range boxes at epochs {bad}')
    print(f'OK -> {out}')


if __name__ == '__main__':
    main()
