"""Fuse raw detections from several models with WBF and evaluate on the val split.

Raw detections come from `eval_mm_tta_val.py --save-dt` (pre-fusion, low
threshold). Fusing them offline means the fusion hyper-parameters (IoU, conf
threshold, confidence rule) can be swept WITHOUT re-running inference, which
costs ~5 min per model.

Usage:
  python fuse_dts_eval.py work_dir/dt_cache/l4u3_val_raw.json \
         work_dir/dt_cache/l4_val_raw.json [--iou-thr 0.55] [--thresh 0.2]
  # sweep
  python fuse_dts_eval.py work_dir/dt_cache/*_val_raw.json --sweep
"""
import argparse
import contextlib
import io
import itertools
import json
import os
import sys

import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VAL_ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def wbf(boxes, iou_thr, n_models, conf_mode):
    """boxes: list of (cls, [x,y,w,h], conf). Returns fused list."""
    out = []
    for cls in sorted(set(b[0] for b in boxes)):
        members = sorted([b for b in boxes if b[0] == cls], key=lambda b: b[2], reverse=True)
        clusters = []
        for box in members:
            for cl in clusters:
                if iou(box[1], cl[0][1]) >= iou_thr:
                    cl.append(box)
                    break
            else:
                clusters.append([box])
        for cl in clusters:
            cf = np.array([b[2] for b in cl])
            ws = cf / cf.sum()
            bx = sum(b[1][0] * w for b, w in zip(cl, ws))
            by = sum(b[1][1] * w for b, w in zip(cl, ws))
            bw = sum(b[1][2] * w for b, w in zip(cl, ws))
            bh = sum(b[1][3] * w for b, w in zip(cl, ws))
            if conf_mode == "max":
                c = float(cf.max())
            elif conf_mode == "sat":
                c = float(np.clip(cf.sum() / (len(cl) ** 0.5), 0, 1))
            else:  # avg, damped by how many of the models saw it
                c = float(np.clip(cf.mean() * (len(cl) / max(n_models, 1)) ** 0.25, 0, 1))
            out.append((cls, [bx, by, bw, bh], c))
    return out


def load_raw(paths, views_per_model):
    """-> {(image_id, cls): [(conf, [x,y,w,h]), ...]}, total_views

    total_views = views_per_model * n_models. Getting this wrong silently breaks
    the confidence damping: using n_models=2 while each model contributes 4 views
    makes (len(cl)/n)^0.25 > 1, i.e. it *inflates* scores and lets low-quality
    boxes survive the threshold (measured: AP dropped from .551 to .536).
    """
    by_img = {}
    for p in paths:
        for d in json.load(open(p)):
            key = (d["image_id"], d["category_id"])
            by_img.setdefault(key, []).append((d["score"], d["bbox"]))
    return by_img, views_per_model * len(paths)


def build_dt(by_img, iou_thr, thresh, n_models, conf_mode, max_boxes=100, pre_thresh=0.0):
    """pre_thresh  = drop per-view boxes before fusion (what predict_mm_tta.py does)
       thresh      = drop fused boxes after fusion
    These are NOT interchangeable: the shipped submissions filter BEFORE fusion
    only, so comparing a pre-filtered run against a post-filtered one is invalid.
    """
    dt = []
    for (image_id, cls), items in by_img.items():
        boxes = [(cls, b, c) for c, b in items if c >= pre_thresh]
        if not boxes:
            continue
        fused = wbf(boxes, iou_thr, n_models, conf_mode)
        fused = [f for f in fused if f[2] >= thresh]
        fused.sort(key=lambda b: b[2], reverse=True)
        for _cls, (x, y, w, h), c in fused[:max_boxes]:
            dt.append({"image_id": image_id, "category_id": cls,
                       "bbox": [x, y, max(w, 1.0), max(h, 1.0)], "score": c})
    return dt


def evaluate(dt):
    from faster_coco_eval import COCO, COCOeval_faster
    gt = COCO(VAL_ANN)
    pred = gt.loadRes(dt) if dt else gt.loadRes([])
    ev = COCOeval_faster(gt, pred, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return ev.stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dts", nargs="+", help="raw detection json files (one per model)")
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--thresh", type=float, default=0.2)
    ap.add_argument("--conf-mode", choices=["avg", "max", "sat"], default="avg")
    ap.add_argument("--views-per-model", type=int, default=4,
                    help="TTA views each model contributed (scales x flips)")
    ap.add_argument("--pre-thresh", type=float, default=0.0,
                    help="drop per-view boxes below this BEFORE fusion")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    by_img, n_models = load_raw(args.dts, args.views_per_model)
    print(f"[fuse] {len(args.dts)} models, {sum(len(v) for v in by_img.values())} raw boxes")

    if args.sweep:
        print(f"{'iou':>5s} {'thr':>5s} {'mode':>5s} {'#pred':>7s} {'AP':>8s} {'AP50':>8s}")
        best = None
        for iou_t, thr in itertools.product([0.5, 0.55, 0.6, 0.65, 0.7],
                                            [0.1, 0.15, 0.2, 0.25, 0.3]):
            dt = build_dt(by_img, iou_t, thr, n_models, args.conf_mode,
                          pre_thresh=args.pre_thresh)
            st = evaluate(dt)
            print(f"{iou_t:5.2f} {thr:5.2f} {args.conf_mode:>5s} {len(dt):7d} "
                  f"{st[0]:8.4f} {st[1]:8.4f}")
            if best is None or st[0] > best[0]:
                best = (st[0], iou_t, thr, st[1], len(dt))
        print(f"BEST AP={best[0]:.4f} at iou={best[1]} thr={best[2]} "
              f"(AP50={best[3]:.4f}, #pred={best[4]})")
        return

    dt = build_dt(by_img, args.iou_thr, args.thresh, n_models, args.conf_mode,
                  pre_thresh=args.pre_thresh)
    st = evaluate(dt)
    print(f"[fuse] iou={args.iou_thr} thr={args.thresh} mode={args.conf_mode} "
          f"#pred={len(dt)} AP={st[0]:.4f} AP50={st[1]:.4f} AP75={st[2]:.4f} "
          f"APs={st[3]:.4f} APm={st[4]:.4f} APl={st[5]:.4f}")


if __name__ == "__main__":
    main()
