"""Competition-format TXT for the pure-RGB or the night-gated Co-DINO (aic_nightgate.py), with optional probes.

The prediction loop and TXT format are the teammate's predict_codino_txt.py (top 100 boxes per image, `class cx cy w h
conf` normalised by the original size, sorted by confidence, empty file when nothing is left). Added:
  --night LEVEL   AICNightProbeBGR right after the loader (deterministic per image; photon 30 / read 0.02)
  --zero-aux      AICZeroAux after it (thermal, window and depth zeroed; gated config only)
  --ir-dir / --depth-dir   override the gated loader's folders (e.g. the Phase 3 test set)
gates.json in --out records every image's illumination gate (gated config), so 'the gate never fired' is checked
per image, not assumed.
Usage (onedl-mmdetection root, venv_codino, PYTHONPATH as for training):
  python predict_nightgate_txt.py --config CFG --ckpt PTH --ann splitA_val.json --img-dir /root/rgb_all2000/images \
      --out DIR [--night 0.01] [--zero-aux]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from mmengine.config import Config
from PIL import Image

import aic_nightgate  # noqa: F401  (registers the transforms for the pure-RGB config too)
from mmdet.apis import inference_detector, init_detector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--night', type=float)
    ap.add_argument('--zero-aux', action='store_true')
    ap.add_argument('--ir-dir')
    ap.add_argument('--depth-dir')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    pipe = list(cfg.test_dataloader.dataset.pipeline)
    gated = pipe[0]['type'] == 'LoadAICGateInputs'
    if not gated and (args.zero_aux or args.ir_dir or args.depth_dir):
        raise ValueError('--zero-aux / --ir-dir / --depth-dir need the gated config')
    if args.ir_dir:
        pipe[0]['ir_dir'] = args.ir_dir
    if args.depth_dir:
        pipe[0]['depth_dir'] = args.depth_dir
    extra = []
    if args.night is not None:
        extra.append(dict(type='AICNightProbeBGR', level=args.night))
    if args.zero_aux:
        extra.append(dict(type='AICZeroAux'))
    cfg.test_dataloader.dataset.pipeline = [pipe[0]] + extra + pipe[1:]

    os.makedirs(args.out, exist_ok=True)
    images = json.load(open(args.ann))['images']
    model = init_detector(cfg, args.ckpt, device='cuda:0')
    print('classes', model.dataset_meta.get('classes'), 'pipeline', [t['type'] for t in cfg.test_dataloader.dataset.pipeline])
    gates = {}
    n_box = 0
    t0 = time.time()
    for i, im in enumerate(sorted(images, key=lambda x: x['file_name'])):
        path = os.path.join(args.img_dir, im['file_name'])
        W, H = Image.open(path).size
        assert (W, H) == (im['width'], im['height']), (path, W, H, im['width'], im['height'])
        with torch.no_grad():
            res = inference_detector(model, path)
        if gated:
            gates[im['file_name']] = model.data_preprocessor.last_gates[0]
        pi = res.pred_instances
        boxes = pi.bboxes.float().cpu().numpy()
        scores = pi.scores.float().cpu().numpy()
        labels = pi.labels.cpu().numpy().astype(int)
        order = np.argsort(-scores, kind='stable')[: args.topk]
        lines = []
        for j in order:
            x1, y1, x2, y2 = boxes[j]
            x1, x2 = np.clip([x1, x2], 0, W)
            y1, y2 = np.clip([y1, y2], 0, H)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            lines.append(f'{labels[j]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {w / W:.6f} {h / H:.6f} '
                         f'{scores[j]:.6f}')
        stem = os.path.splitext(im['file_name'])[0]
        with open(os.path.join(args.out, stem + '.txt'), 'w') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        n_box += len(lines)
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(images)} boxes={n_box} {(time.time() - t0) / (i + 1):.3f}s/img', flush=True)
    if gated:
        g = np.array(list(gates.values()))
        summary = dict(n=len(g), open=int((g > 0).sum()), max=float(g.max()), mean=float(g.mean()))
        json.dump(dict(summary=summary, gates=gates), open(os.path.join(args.out, 'gates.json'), 'w'), indent=1)
        print('gates', summary)
    print(f'DONE {len(images)} txt, {n_box} boxes, {(time.time() - t0) / max(1, len(images)):.3f}s/img')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
