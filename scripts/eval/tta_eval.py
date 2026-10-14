#!/usr/bin/env python3
"""Test-time-augmentation (multi-scale + h-flip) evaluation with COCO AP for DEIMv2.

Runs several augmented views (scales x flips) per image, maps the predicted boxes
of the flipped views back, merges all views with the SAME WBF recipe the project
uses for submissions (scripts/eval/predict_mm_tta.py), and feeds the merged set to
the CocoEvaluator. Faithful "TTA vs no-TTA" comparison on the same checkpoint and
the same val pipeline.

Usage (cwd must be vendor/DEIMv2, or anywhere -- paths are self-resolved):
  python scripts/tta_eval.py -c <val_cfg.yml> -r <ckpt.pth> --scales 640,800 --flip

Notes
-----
* orig_size is [w, h] (engine/data/dataset/coco_dataset.py) -> un-flip uses idx 0.
* --thresh defaults to 0.0 so the TTA-vs-noTTA delta isolates the *merge*; the
  submission pipeline uses 0.1 (set --thresh 0.1 to reproduce submission numbers).
"""
import argparse
import os
import sys
from collections import defaultdict
from contextlib import nullcontext

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, 'vendor', 'DEIMv2'),
           os.path.join(_ROOT, 'scripts', 'lib'),
           os.path.join(_ROOT, 'scripts', 'eval')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
import torchvision

from engine.core.yaml_config import YAMLConfig
from engine.solver.det_solver import DetSolver
from predict_mm_tta import wbf  # project-validated weighted box fusion


@torch.no_grad()
def tta_evaluate(module, postprocessor, data_loader, coco_evaluator, device,
                 scales, use_flip, use_amp, merge='wbf', iou_thr=0.7,
                 pre_thresh=0.05, thresh=0.0, conf_mode='avg', max_boxes=100):
    module.eval()
    coco_evaluator.cleanup()

    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    n_views = len(scales) * (2 if use_flip else 1)
    all_preds = {}

    for samples, targets in data_loader:
        # samples: plain tensor [B,6,S,S] (already resized+normalized by the val tf)
        orig_sizes = torch.stack([t['orig_size'] for t in targets]).to(device)  # [B,2] w,h
        img_ids = [t['image_id'].item() for t in targets]

        views = defaultdict(list)   # img_id -> [(cls, [x1,y1,x2,y2], conf)]
        for s in scales:
            for flip in ([False, True] if use_flip else [False]):
                view = samples
                if s != samples.shape[-1]:
                    view = F.interpolate(view, size=(s, s), mode='bilinear',
                                         align_corners=False)
                if flip:
                    view = view.flip(-1)
                view = view.to(device)
                with amp_ctx():
                    outputs = module(view)
                results = postprocessor(outputs, orig_sizes, view)
                for i, res in enumerate(results):
                    boxes = res['boxes'].float()
                    labels = res['labels']
                    scores = res['scores'].float()
                    if flip:
                        # orig_size == [w, h] -> width is index 0
                        W = float(orig_sizes[i, 0].item())
                        x1, y1, x2, y2 = boxes.unbind(1)
                        boxes = torch.stack([W - x2, y1, W - x1, y2], dim=1)
                    keep = scores > pre_thresh
                    iid = img_ids[i]
                    for k in torch.nonzero(keep).flatten().tolist():
                        views[iid].append((int(labels[k].item()),
                                           [float(v) for v in boxes[k].tolist()],
                                           float(scores[k].item())))

        for iid, dets in views.items():
            if merge == 'wbf':
                fused = wbf(dets, iou_thr=iou_thr, n_views=n_views,
                            conf_mode=conf_mode) if dets else []
                fused = [f for f in fused if f[2] >= thresh]
                fused = sorted(fused, key=lambda b: b[2], reverse=True)[:max_boxes]
                if fused:
                    labels = torch.tensor([f[0] for f in fused], dtype=torch.long)
                    boxes = torch.tensor([f[1] for f in fused], dtype=torch.float32)
                    scores = torch.tensor([f[2] for f in fused], dtype=torch.float32)
                else:
                    labels = torch.zeros(0, dtype=torch.long)
                    boxes = torch.zeros(0, 4, dtype=torch.float32)
                    scores = torch.zeros(0, dtype=torch.float32)
            else:  # per-class NMS fallback (unvalidated, kept for ablation)
                labels = torch.tensor([d[0] for d in dets], dtype=torch.long)
                boxes = torch.tensor([d[1] for d in dets], dtype=torch.float32)
                scores = torch.tensor([d[2] for d in dets], dtype=torch.float32)
                keep = []
                for c in labels.unique().tolist():
                    m = labels == c
                    idx = torchvision.ops.nms(boxes[m].contiguous(),
                                              scores[m].contiguous(), iou_thr)
                    keep.append(torch.where(m)[0][idx])
                m = torch.cat(keep) if keep else torch.empty(0, dtype=torch.long)
                order = scores[m].argsort(descending=True)[:max_boxes]
                m = m[order]
                labels, boxes, scores = labels[m], boxes[m], scores[m]

            all_preds[iid] = {'labels': labels, 'boxes': boxes, 'scores': scores}

    coco_evaluator.update(all_preds)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = coco_evaluator.coco_eval['bbox'].stats
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--scales', default='640,800')
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--no-flip', dest='flip', action='store_false')
    ap.add_argument('--merge', choices=['wbf', 'nms'], default='wbf')
    ap.add_argument('--iou-thr', type=float, default=0.7)
    ap.add_argument('--pre-thresh', type=float, default=0.05)
    ap.add_argument('--thresh', type=float, default=0.0)
    ap.add_argument('--conf-mode', choices=['avg', 'max', 'sat'], default='avg')
    ap.add_argument('--use-amp', action='store_true')
    ap.set_defaults(flip=True)
    args = ap.parse_args()

    scales = [int(x) for x in args.scales.split(',') if x.strip()]

    cfg = YAMLConfig(args.config)
    cfg.resume = args.resume
    cfg.device = args.device
    cfg.tuning = None

    solver = DetSolver(cfg)
    solver.eval()

    module = solver.ema.module if solver.ema else solver.model
    # dfine 系 decoder 会按 eval_spatial_size 缓存 anchors/valid_mask, 多尺度
    # 插值视图 token 数对不上直接崩; 置 None 让其按实际 spatial_shapes 现算
    dec = getattr(module, "decoder", None)
    if dec is not None and hasattr(dec, "eval_spatial_size"):
        dec.eval_spatial_size = None
    print(f"[TTA] scales={scales} flip={args.flip} merge={args.merge} "
          f"iou={args.iou_thr} pre={args.pre_thresh} thresh={args.thresh} "
          f"conf={args.conf_mode}", flush=True)

    stats = tta_evaluate(
        module, solver.postprocessor, solver.val_dataloader, solver.evaluator,
        solver.device, scales, args.flip, args.use_amp, args.merge,
        args.iou_thr, args.pre_thresh, args.thresh, args.conf_mode,
    )
    ap_all, ap50 = float(stats[0]), float(stats[1])
    print(f"RESULT_AP {ap_all:.4f} RESULT_AP50 {ap50:.4f}", flush=True)
    return ap_all, ap50


if __name__ == '__main__':
    main()
