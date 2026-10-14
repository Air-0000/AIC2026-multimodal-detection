#!/usr/bin/env python3
"""Paired image-level bootstrap of mAP@50-95 on val200.

The val split is only 200 images / 1523 boxes, so the top of MODEL_RECORDS tableA
spans 0.5345 -> 0.5222 (1.2 pt) and we cannot tell signal from noise by staring
at point estimates. This resamples the 200 images with replacement and recomputes
COCO mAP each draw, giving:

  * a 95% CI per model, and
  * a *paired* 95% CI on each pairwise delta (identical draws for both models,
    which cancels the image-difficulty variance and is far tighter than
    comparing two independent CIs).

Run on the GPU host (detection dumps must already exist):
  python3 scripts/analysis/bootstrap_val_ci.py --iters 200
"""

import argparse
import contextlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VENDOR = os.path.join(ROOT, "vendor", "DEIMv2")
sys.path.insert(0, VENDOR)

from faster_coco_eval import COCO, COCOeval_faster  # noqa: E402

ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets-dir", default=os.path.join(ROOT, "outputs/analysis/_dets"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-floor", type=float, default=0.005)
    ap.add_argument("--out-json", default=os.path.join(ROOT, "outputs/metrics/val_ci_bootstrap.json"))
    ap.add_argument("--out-npz", default=os.path.join(ROOT, "outputs/metrics/val_ci_bootstrap.npz"))
    args = ap.parse_args()

    base = json.load(open(ANN))
    imgs = base["images"]
    cats = base["categories"]
    anns_by_img = {}
    for a in base["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    dumps = sorted(
        os.path.join(args.dets_dir, x)
        for x in os.listdir(args.dets_dir)
        if x.endswith(".pkl")
    )
    assert dumps, f"no dumps in {args.dets_dir}"

    # ---- pre-split each model's detections into per-image [x,y,w,h,score,label]
    prep = {}
    for p in dumps:
        with open(p, "rb") as f:
            dets = pickle.load(f)["dets"]
        per_img = {}
        for iid, d in dets.items():
            s = d["scores"]
            m = s >= args.score_floor
            if not m.any():
                per_img[iid] = np.zeros((0, 6))
                continue
            b = d["boxes"][m]
            xywh = np.stack(
                [b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1
            )
            per_img[iid] = np.concatenate(
                [xywh, s[m, None], d["labels"][m, None].astype(np.float64)], axis=1
            )
        prep[os.path.basename(p)[:-4]] = per_img
    names = list(prep.keys())
    n_img = len(imgs)
    print(f"models: {names}   images={n_img}  boxes={len(base['annotations'])}")

    def gt_dict(ids):
        images, annotations = [], []
        aid = 1
        for k, old in enumerate(ids):
            src = imgs[old]
            nid = k + 1
            images.append(
                {
                    "id": nid,
                    "width": src["width"],
                    "height": src["height"],
                    "file_name": src["file_name"],
                }
            )
            for a in anns_by_img.get(src["id"], []):
                annotations.append(
                    {
                        "id": aid,
                        "image_id": nid,
                        "category_id": a["category_id"],
                        "bbox": list(a["bbox"]),
                        "area": a["area"],
                        "iscrowd": 0,
                    }
                )
                aid += 1
        return {"images": images, "annotations": annotations, "categories": cats}

    def dt_list(per_img, ids):
        out = []
        for k, old in enumerate(ids):
            arr = per_img.get(imgs[old]["id"])
            if arr is None or len(arr) == 0:
                continue
            nid = k + 1
            for row in arr:
                out.append(
                    {
                        "image_id": nid,
                        "category_id": int(row[5]),
                        "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                        "score": float(row[4]),
                    }
                )
        return out

    devnull = open(os.devnull, "w")

    def score_ids(ids):
        """mAP@50-95 for every model on one fixed draw of image indices."""
        gt = COCO(gt_dict(ids))
        out = {}
        for nm in names:
            dt = dt_list(prep[nm], ids)
            if not dt:
                out[nm] = 0.0
                continue
            cd = gt.loadRes(dt)
            ev = COCOeval_faster(gt, cd, iouType="bbox", separate_eval=True)
            with contextlib.redirect_stdout(devnull):
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            out[nm] = float(ev.stats[0])
        return out

    # ---- point estimate on the full val200 (must reproduce the training logs)
    full_idx = np.arange(n_img)
    point = score_ids(full_idx)
    print("\n=== point estimate (full val200) ===")
    for nm in names:
        print(f"  {nm:14s} mAP@50-95 = {point[nm]:.4f}")

    # ---- paired bootstrap: one draw of image indices shared by all models
    rng = np.random.default_rng(args.seed)
    acc = {nm: np.zeros(args.iters) for nm in names}
    for it in range(args.iters):
        idx = rng.integers(0, n_img, size=n_img)
        s = score_ids(idx)
        for nm in names:
            acc[nm][it] = s[nm]
        if (it + 1) % 10 == 0:
            print(f"  bootstrap {it + 1}/{args.iters}", flush=True)

    # ---- summarize
    report = {"point": point, "iters": args.iters, "seed": args.seed,
              "n_images": n_img, "n_boxes": len(base["annotations"]),
              "score_floor": args.score_floor, "models": {}}
    print(f"\n=== per-model 95pct CI (bootstrap, n={args.iters}) ===")
    print(f"{'model':16s} {'mAP':>7s} {'boot_mean':>10s} {'sd':>7s} {'95% CI':>18s}")
    for nm in names:
        a = acc[nm]
        lo, hi = np.percentile(a, [2.5, 97.5])
        report["models"][nm] = {
            "mAP": point[nm], "boot_mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "ci95": [float(lo), float(hi)],
        }
        print(f"{nm:16s} {point[nm]:7.4f} {a.mean():10.4f} {a.std(ddof=1):7.4f} "
              f"[{lo:6.4f}, {hi:6.4f}]")

    print("\n=== paired delta (same draws) ===")
    print(f"{'A - B':34s} {'delta':>8s} {'95% CI':>20s} {'P(d>0)':>8s} {'sig':>5s}")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = acc[a] - acc[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            ppos = float((d > 0).mean())
            sig = "YES" if (lo > 0 or hi < 0) else "no"
            pairs.append({"A": a, "B": b, "delta": float(point[a] - point[b]),
                          "ci95": [float(lo), float(hi)],
                          "p_gt0": ppos, "significant": sig == "YES"})
            print(f"{a + ' - ' + b:34s} {point[a] - point[b]:8.4f} "
                  f"[{lo:8.4f}, {hi:8.4f}] {ppos:8.3f} {sig:>5s}")
    report["pairs"] = pairs

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    np.savez(args.out_npz, **{f"boot_{k}": v for k, v in acc.items()})
    print(f"\nsaved -> {args.out_json}")


if __name__ == "__main__":
    main()
