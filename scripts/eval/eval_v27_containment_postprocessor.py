#!/usr/bin/env python3
"""
Evaluate Sovereign V27 with Native Containment Fusion & Cross-Class Conflict Resolution
========================================================================================
Validates on:
  1. Old Raw Validation Set (instances_val.json, 1,523 annotations)
  2. Comprehensive Clean Quality Validation Set (instances_val_clean_quality.json, 1,757 annotations)
"""

import os
import sys
import copy
import time
import torch
import torchvision
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

_root = '/root/autodl-tmp/AIC'
if not os.path.exists(_root):
    _root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(_root, 'vendor/DEIMv2'))
sys.path.insert(0, os.path.join(_root, 'scripts/lib'))
os.chdir(os.path.join(_root, 'vendor/DEIMv2'))

from engine.core import YAMLConfig
from engine.data import CocoEvaluator
from engine.solver.det_engine import evaluate
from pycocotools.coco import COCO


def py_box_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-9 else 0.0


def py_containment(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


class AdvancedContainmentPostProcessor(torch.nn.Module):
    def __init__(self, base_postprocessor, iou_thr=0.45, cont_thr=0.65, cross_conflict_thr=0.50):
        super().__init__()
        self.base = base_postprocessor
        self.iou_thr = iou_thr
        self.cont_thr = cont_thr
        self.cross_conflict_thr = cross_conflict_thr
        self.deploy_mode = False

    def forward(self, outputs, orig_target_sizes, samples=None):
        base_results = self.base(outputs, orig_target_sizes, samples)
        refined_results = []

        for res in base_results:
            labels = res['labels'].cpu().numpy()
            boxes = res['boxes'].float().cpu().numpy()
            scores = res['scores'].float().cpu().numpy()

            # Filter candidates above minimal confidence threshold
            keep_mask = scores >= 0.01
            labels = labels[keep_mask]
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]

            # 1. Same-Class Containment Fusion (解决一个东西多个框)
            by_class = defaultdict(list)
            for lab, box, sc in zip(labels, boxes, scores):
                by_class[int(lab)].append((box, sc))

            fused_candidates = []
            for cid, items in by_class.items():
                # Sort by area descending
                items = sorted(items, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]), reverse=True)
                merged = []
                skip = set()
                for i in range(len(items)):
                    if i in skip:
                        continue
                    cur_box, cur_sc = items[i]
                    for j in range(i + 1, len(items)):
                        if j in skip:
                            continue
                        other_box, other_sc = items[j]
                        iou = py_box_iou(cur_box, other_box)
                        cont = py_containment(cur_box, other_box)
                        if iou >= self.iou_thr or cont >= self.cont_thr:
                            # Merge into enclosing envelope
                            cur_box = np.array([
                                min(cur_box[0], other_box[0]),
                                min(cur_box[1], other_box[1]),
                                max(cur_box[2], other_box[2]),
                                max(cur_box[3], other_box[3])
                            ], dtype=np.float32)
                            cur_sc = max(cur_sc, other_sc)
                            skip.add(j)
                    merged.append((cid, cur_box, cur_sc))
                fused_candidates.extend(merged)

            # 2. Cross-Class Spatial Conflict Resolution (解决类别错误)
            fused_candidates = sorted(fused_candidates, key=lambda x: x[2], reverse=True)
            final_boxes = []
            for item in fused_candidates:
                cid, box, sc = item
                conflict = False
                for kept_cid, kept_box, kept_sc in final_boxes:
                    if cid == kept_cid:
                        continue
                    # Legitimate co-existence: rider on bike, person on bench
                    if (cid == 0 and kept_cid in (5, 11)) or (kept_cid == 0 and cid in (5, 11)):
                        continue
                    if (cid == 0 and kept_cid == 3) or (kept_cid == 0 and cid == 3):
                        continue
                    iou = py_box_iou(box, kept_box)
                    cont = py_containment(box, kept_box)
                    if iou >= self.cross_conflict_thr or cont >= 0.75:
                        conflict = True
                        break
                if not conflict:
                    final_boxes.append((cid, box, sc))

            if len(final_boxes) > 0:
                out_labs = torch.tensor([x[0] for x in final_boxes], dtype=torch.long, device=res['labels'].device)
                out_bxs = torch.tensor(np.array([x[1] for x in final_boxes]), dtype=torch.float32, device=res['boxes'].device)
                out_scs = torch.tensor([x[2] for x in final_boxes], dtype=torch.float32, device=res['scores'].device)
            else:
                out_labs = torch.zeros(0, dtype=torch.long, device=res['labels'].device)
                out_bxs = torch.zeros((0, 4), dtype=torch.float32, device=res['boxes'].device)
                out_scs = torch.zeros(0, dtype=torch.float32, device=res['scores'].device)

            refined_results.append(dict(labels=out_labs, boxes=out_bxs, scores=out_scs))

        return refined_results


def main():
    cfg_path = "configs/deimv2/sovereign_v27_pure_unbiased_2048.yml"
    ckpt_path = "outputs/sovereign_v27_pure_unbiased_2048/best_stg2.pth"

    cfg = YAMLConfig(cfg_path, resume=ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['ema']['module'] if 'ema' in ckpt and ckpt['ema'] else ckpt.get('model', ckpt)
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.cuda().eval()

    base_postprocessor = cfg.postprocessor.cuda().eval()
    adv_postprocessor = AdvancedContainmentPostProcessor(base_postprocessor, iou_thr=0.45, cont_thr=0.65).cuda()

    ann_old = "/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    ann_clean = "/root/autodl-tmp/AIC/work_dir/deimv2_dataset/valid/annotations/instances_val_clean_quality.json"

    for ann_name, ann_file in [("Old Raw Validation Set", ann_old), ("Clean Quality Validation Set", ann_clean)]:
        print("\n" + "=" * 80)
        print(f" Evaluating {ann_name} with Containment PostProcessor ".center(80, "="))
        print("=" * 80)

        coco = COCO(ann_file)
        val_dataset = cfg.val_dataloader.dataset
        val_dataset.ann_file = ann_file
        val_dataset.coco = coco
        val_dataset.ids = list(sorted(coco.imgs.keys()))

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=cfg.val_dataloader.collate_fn,
        )

        coco_evaluator = CocoEvaluator(coco, ('bbox',))
        stats_dict, evaluator = evaluate(
            model,
            criterion=cfg.criterion.cuda(),
            postprocessor=adv_postprocessor,
            data_loader=val_loader,
            coco_evaluator=coco_evaluator,
            device="cuda:0"
        )
        stats = evaluator.coco_eval['bbox'].stats
        print(f"\n[Result on {ann_name}]:")
        print(f"  • mAP (0.50:0.95): {stats[0]:.4f}")
        print(f"  • AP50:           {stats[1]:.4f}")
        print(f"  • AP75:           {stats[2]:.4f}")
        print(f"  • AP (small):     {stats[3]:.4f}")
        print(f"  • AP (medium):    {stats[4]:.4f}")
        print(f"  • AP (large):     {stats[5]:.4f}")


if __name__ == '__main__':
    main()
