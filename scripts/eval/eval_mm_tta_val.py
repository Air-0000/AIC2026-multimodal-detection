"""A/B test: does TTA actually help? -- same model, same 200-image holdout,
only the test-time views differ.

Runs the 6ch multimodal model on `work_dir/deimv2_dataset/valid` (200 images with
COCO annotations, held out from the 1800-image train split) and reports AP /
AP50 via faster_coco_eval.

Compare e.g.:
  # no TTA (single scale, no flip) -- the training-resolution baseline
  python eval_mm_tta_val.py --scales 640 --no-flip
  # current submission recipe
  python eval_mm_tta_val.py --scales 640,800
  # single scale + flip only
  python eval_mm_tta_val.py --scales 640

Usage: --limit N for a quick smoke run.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIMV2_ROOT = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, DEIMV2_ROOT)
os.chdir(DEIMV2_ROOT)
from engine.core import YAMLConfig

# Reuse the exact channel assembly + fusion from the submission script so the
# A/B differs ONLY in the number of views.
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))
from predict_mm_tta import build_6ch, to_model_input, wbf

VAL_IMG = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/images")
VAL_ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
HEAT = os.path.join(ROOT, "work_dir/deimv2_heat/valid")
RAWIR = os.path.join(ROOT, "work_dir/deimv2_ir/valid")
DEPTH = os.path.join(ROOT, "dataset/train/depth")


def iou_xywh(a, b):
    """IoU of two [x, y, w, h] absolute-pixel boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def tp_fp_fn(dt, gt, iou_thr=0.5):
    """Greedy per-image, per-class matching at a single IoU threshold.

    COCO's AP hides this: it sweeps 10 IoU thresholds and 101 recall points, so
    you cannot read "how many boxes are simply wrong" off it. Here each GT is
    matched at most once (highest-confidence prediction wins); unmatched
    predictions are false positives and unmatched GTs are misses.
    """
    from collections import defaultdict
    gts = defaultdict(list)
    for a in gt.dataset["annotations"]:
        gts[(a["image_id"], a["category_id"])].append(a["bbox"])
    preds = defaultdict(list)
    for d in dt:
        preds[(d["image_id"], d["category_id"])].append((d["score"], d["bbox"]))

    img_ids = [im["id"] for im in gt.dataset["images"]]
    cats = sorted(c["id"] for c in gt.dataset["categories"])
    out = {}
    for cid in cats:
        tp = fp = fn = 0
        for iid in img_ids:
            g = gts.get((iid, cid), [])
            p = sorted(preds.get((iid, cid), []), key=lambda t: -t[0])
            matched = [False] * len(g)
            for _s, pb in p:
                best_i, best_j = iou_thr, -1
                for j, gb in enumerate(g):
                    if matched[j]:
                        continue
                    v = iou_xywh(pb, gb)
                    if v > best_i:
                        best_i, best_j = v, j
                if best_j >= 0:
                    matched[best_j] = True
                    tp += 1
                else:
                    fp += 1
            fn += sum(1 for m in matched if not m)
        n_gt = sum(1 for a in gt.dataset["annotations"] if a["category_id"] == cid)
        n_pr = sum(1 for d in dt if d["category_id"] == cid)
        out[cid] = dict(gt=n_gt, pred=n_pr, tp=tp, fp=fp, fn=fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deimv2/"
                    "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3.yml")
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "vendor/DEIMv2/outputs/"
        "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3/best_stg1.pth"))
    ap.add_argument("--scales", default="640,800")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--conf-mode", choices=["avg", "sat"], default="avg",
                    help="avg=standard WBF; sat=legacy sum/sqrt(n) negative control")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-class", action="store_true", help="also print per-class AP")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="IoU threshold used for the TP/FP/FN breakdown")
    ap.add_argument("--save-dt", default=None,
                    help="dump raw detections to a json so fusion can be swept "
                         "offline without re-running inference")
    args = ap.parse_args()
    # os.chdir(DEIMV2_ROOT) already happened at import time, so relative paths
    # would land inside vendor/DEIMv2 -- resolve them against ROOT.
    if args.save_dt and not os.path.isabs(args.save_dt):
        args.save_dt = os.path.join(ROOT, args.save_dt)
    if args.ckpt and not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(ROOT, args.ckpt)

    scales = [int(s) for s in args.scales.split(",")]
    flips = [False] if args.no_flip else [False, True]
    n_views = len(scales) * len(flips)

    cfg = YAMLConfig(args.config)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["ema"]["module"] if "ema" in ck else ck.get("model", ck)
    cfg.model.load_state_dict(sd, strict=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig):
            return self.postprocessor(self.model(images), orig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer = M().to(device).eval()

    ann = json.load(open(VAL_ANN))
    id_by_name = {os.path.basename(im["file_name"]): im["id"] for im in ann["images"]}

    files = sorted(glob.glob(os.path.join(VAL_IMG, "*")))
    files = [f for f in files if os.path.basename(f) in id_by_name]
    if args.limit:
        files = files[: args.limit]
    print(f"[ab] {len(files)} val images | scales={scales} flips={flips} "
          f"n_views={n_views} thresh={args.thresh} iou_thr={args.iou_thr}", flush=True)

    dt = []
    raw_dump = [] if args.save_dt else None
    t0 = time.time()
    with torch.no_grad():
        for i, imp in enumerate(files):
            name = os.path.basename(imp)
            image_id = id_by_name[name]
            rgb_pil = Image.open(imp).convert("RGB")
            w, h = rgb_pil.size
            orig = torch.tensor([[w, h]]).to(device)
            x = build_6ch(rgb_pil, HEAT, RAWIR, DEPTH, name)

            views = []
            raw_all = [] if args.save_dt else None
            for size in scales:
                t = to_model_input(x, size).to(device)
                for flip in flips:
                    ti = torch.flip(t, dims=[2]) if flip else t
                    labels, boxes, scores = infer(ti.unsqueeze(0), orig)
                    b = boxes[0].clone()
                    if flip:
                        x1, x2 = b[:, 0].clone(), b[:, 2].clone()
                        b[:, 0], b[:, 2] = w - x2, w - x1
                    keep = scores[0] > args.thresh
                    for k in torch.nonzero(keep).flatten():
                        views.append((int(labels[0][k].item()),
                                      [float(b[k, 0]), float(b[k, 1]),
                                       float(b[k, 2]), float(b[k, 3])],
                                      float(scores[0][k].item())))
                        if raw_all is not None:
                            raw_all.append({
                                "image_id": image_id,
                                "category_id": int(labels[0][k].item()),
                                "bbox": [float(b[k, 0]), float(b[k, 1]),
                                         float(b[k, 2] - b[k, 0]), float(b[k, 3] - b[k, 1])],
                                "score": float(scores[0][k].item())})
            if raw_all is not None:
                raw_dump.extend(raw_all)

            fused = wbf(views, args.iou_thr, n_views=n_views,
                        conf_mode=args.conf_mode) if n_views > 1 else views
            for cls, (x1, y1, x2, y2), c in fused:
                dt.append({"image_id": image_id, "category_id": cls,
                           "bbox": [x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)],
                           "score": c})
            if (i + 1) % 50 == 0:
                print(f"[ab] {i + 1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

    if args.save_dt:
        json.dump(raw_dump, open(args.save_dt, "w"))
        print(f"[ab] saved {len(raw_dump)} RAW (pre-fusion) detections -> {args.save_dt}", flush=True)

    from faster_coco_eval import COCO, COCOeval_faster
    gt = COCO(VAL_ANN)
    pred = gt.loadRes(dt) if dt else gt.loadRes([])
    ev = COCOeval_faster(gt, pred, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    # per-class AP: pinpoints WHICH classes hold the score back
    if args.per_class:
        cats = gt.dataset["categories"]
        per = {}
        for c in sorted(cats, key=lambda x: x["id"]):
            cid = c["id"]
            # a fresh COCOeval per class -- reusing one object and just swapping
            # params.catIds silently keeps the global result (verified: every
            # class reported the identical overall AP).
            dt_c = [d for d in dt if d["category_id"] == cid]
            pred_c = gt.loadRes(dt_c) if dt_c else gt.loadRes([])
            ev_c = COCOeval_faster(gt, pred_c, iouType="bbox")
            ev_c.params.catIds = [cid]
            ev_c.evaluate()
            ev_c.accumulate()
            with contextlib.redirect_stdout(io.StringIO()):  # summarize() prints
                ev_c.summarize()                             # and fills .stats
            ap = float(ev_c.stats[0])
            n_gt = sum(1 for a in gt.dataset["annotations"] if a["category_id"] == cid)
            n_pr = len(dt_c)
            per[cid] = (ap, n_gt, n_pr)
        conf = tp_fp_fn(dt, gt, iou_thr=args.match_iou)
        print(f"\n--- per-class on val (match IoU>={args.match_iou}) ---")
        print(f"{'id':>3s} {'AP':>8s} {'#GT':>5s} {'#pred':>6s} {'TP':>5s} {'FP(错框)':>9s} "
              f"{'FN(漏检)':>9s} {'precision':>10s} {'recall':>8s}")
        for cid, (ap, n_gt, n_pr) in sorted(per.items(), key=lambda kv: kv[1][0]):
            c = conf[cid]
            prec = c["tp"] / c["pred"] if c["pred"] else 0.0
            rec = c["tp"] / c["gt"] if c["gt"] else 0.0
            print(f"{cid:3d} {ap:8.4f} {c['gt']:5d} {c['pred']:6d} {c['tp']:5d} "
                  f"{c['fp']:9d} {c['fn']:9d} {prec:10.3f} {rec:8.3f}")
        tot = {k: sum(v[k] for v in conf.values()) for k in ("gt", "pred", "tp", "fp", "fn")}
        print(f"{'ALL':>3s} {'':>8s} {tot['gt']:5d} {tot['pred']:6d} {tot['tp']:5d} "
              f"{tot['fp']:9d} {tot['fn']:9d} "
              f"{tot['tp']/max(tot['pred'],1):10.3f} {tot['tp']/max(tot['gt'],1):8.3f}")

    res = {"tag": args.tag, "scales": scales, "flips": flips, "n_views": n_views,
           "thresh": args.thresh, "iou_thr": args.iou_thr, "n_images": len(files),
           "n_pred": len(dt),
           "AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
           "APs": float(stats[3]), "APm": float(stats[4]), "APl": float(stats[5]),
           "secs": round(time.time() - t0, 1)}
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
