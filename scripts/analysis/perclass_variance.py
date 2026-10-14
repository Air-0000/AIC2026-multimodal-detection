#!/usr/bin/env python3
"""Per-class decomposition of mAP variance on val200.

Question: why does epoch-to-epoch val mAP swing ~5 pt when consecutive epochs
differ only by a tiny lr-scaled weight update?

mAP@50-95 is the *unweighted* mean of 12 per-class APs. On val200 four of those
classes have almost no ground truth (tricycle=2 boxes, ball=13, boat=14, uav=19
out of 1523), so they still carry 4/12 of the score while being dominated by
sampling noise. This script quantifies that:

  * per-class AP for every dumped model,
  * image-level bootstrap sd per class (how much a single class moves under
    resampling of the 200 images), and
  * how much of the mAP std each class contributes.

Run on a machine with faster_coco_eval / pycocotools:
  python3 scripts/analysis/perclass_variance.py --iters 200
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

try:
    from faster_coco_eval import COCO, COCOeval_faster as _Eval
    EVAL_KW = dict(separate_eval=True)
except Exception:  # pragma: no cover
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval as _Eval
    EVAL_KW = {}

ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets-dir", default=os.path.join(ROOT, "outputs/analysis/_dets"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-floor", type=float, default=0.005)
    ap.add_argument("--out-json", default=os.path.join(ROOT, "outputs/metrics/perclass_variance.json"))
    args = ap.parse_args()

    base = json.load(open(ANN))
    imgs = base["images"]
    cats = base["categories"]
    cat_names = {c["id"]: c["name"] for c in cats}
    anns_by_img = {}
    for a in base["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    from collections import Counter
    n_gt = Counter(a["category_id"] for a in base["annotations"])

    dumps = sorted(
        os.path.join(args.dets_dir, x)
        for x in os.listdir(args.dets_dir)
        if x.endswith(".pkl")
    )
    assert dumps, "no dumps in %s" % args.dets_dir

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
            xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
            per_img[iid] = np.concatenate(
                [xywh, s[m, None], d["labels"][m, None].astype(np.float64)], axis=1
            )
        prep[os.path.basename(p)[:-4]] = per_img
    names = list(prep.keys())
    n_img = len(imgs)
    print("models: %s   images=%d  boxes=%d" % (names, n_img, len(base["annotations"])))

    devnull = open(os.devnull, "w")
    _stats_mAP = {}
    _cids_sorted = sorted(cat_names)
    _cid_idx = {c: i for i, c in enumerate(_cids_sorted)}
    _K = len(_cids_sorted)

    def dt_list(per_img, ids):
        out = []
        for k, old in enumerate(ids):
            arr = per_img.get(imgs[old]["id"])
            if arr is None or len(arr) == 0:
                continue
            nid = k + 1
            for row in arr:
                out.append({"image_id": nid, "category_id": int(row[5]),
                            "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                            "score": float(row[4])})
        return out

    def gt_dict(ids):
        images, annotations = [], []
        aid = 1
        for k, old in enumerate(ids):
            src = imgs[old]
            nid = k + 1
            images.append({"id": nid, "width": src["width"], "height": src["height"],
                           "file_name": src["file_name"]})
            for a in anns_by_img.get(src["id"], []):
                annotations.append({"id": aid, "image_id": nid, "category_id": a["category_id"],
                                    "bbox": list(a["bbox"]), "area": a["area"], "iscrowd": 0})
                aid += 1
        return {"images": images, "annotations": annotations, "categories": cats}

    def per_class_ap(ids):
        """-> {model: np.array(12) of per-class AP} (NaN when a class has no GT in
        this draw, mirroring COCOeval which drops GT-less classes from the mean)."""
        gtd = gt_dict(ids)
        ngt = np.zeros(_K)
        for a in gtd["annotations"]:
            ngt[_cid_idx[a["category_id"]]] += 1
        gt = COCO(gtd)
        out = {}
        for nm in names:
            dt = dt_list(prep[nm], ids)
            if not dt:
                out[nm] = np.full(_K, np.nan)
                _stats_mAP[nm] = 0.0
                continue
            cd = gt.loadRes(dt)
            ev = _Eval(gt, cd, iouType="bbox", **EVAL_KW)
            with contextlib.redirect_stdout(devnull):
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            # sanity: our per-class mean must reproduce the official mAP
            _stats_mAP[nm] = float(ev.stats[0])
            # precision: (T=10 iou, R=101 recall, K=12 class, A=4 area, M=3 maxDets)
            # official AP = mean over T of precision at area=all(0), maxDets=100(2)
            prec = ev.eval["precision"]
            aps = []
            for k in range(prec.shape[2]):
                if ngt[k] == 0:
                    aps.append(np.nan)  # class absent from this draw -> excluded
                    continue
                pk = prec[:, :, k, 0, 2]
                m = pk > -1
                aps.append(float(pk[m].mean()) if m.any() else 0.0)
            out[nm] = np.array(aps)
        return out, _cids_sorted

    full = np.arange(n_img)
    point, cids_sorted = per_class_ap(full)
    print("\n=== per-class AP on full val200 ===")
    hdr = "%-14s %6s " % ("class", "n_gt") + " ".join("%10s" % n[:10] for n in names)
    print(hdr)
    for k, cid in enumerate(cids_sorted):
        row = "%-14s %6d " % (cat_names[cid], n_gt[cid])
        row += " ".join("%10.3f" % point[nm][k] for nm in names)
        print(row)
    print("%-14s %6d " % ("mAP", len(base["annotations"]))
          + " ".join("%10.3f" % point[nm].mean() for nm in names))
    print("%-14s %6s " % ("(ev.stats[0])", "-")
          + " ".join("%10.3f" % _stats_mAP[nm] for nm in names))
    for nm in names:
        d = abs(point[nm].mean() - _stats_mAP[nm])
        assert d < 1e-6, "per-class mean %.6f != stats[0] %.6f for %s (delta %.2e)" % (
            point[nm].mean(), _stats_mAP[nm], nm, d)
    print("sanity OK: per-class mean reproduces ev.stats[0] for all models")

    # ---- bootstrap: per-class AP distribution over image resamples
    rng = np.random.default_rng(args.seed)
    K = len(cids_sorted)
    acc = {nm: np.zeros((args.iters, K)) for nm in names}
    for it in range(args.iters):
        idx = rng.integers(0, n_img, size=n_img)
        s, _ = per_class_ap(idx)
        for nm in names:
            acc[nm][it] = s[nm]
        if (it + 1) % 20 == 0:
            print("  bootstrap %d/%d" % (it + 1, args.iters), flush=True)

    report = {"n_gt": {cat_names[c]: int(n_gt[c]) for c in cids_sorted},
              "point": {nm: {cat_names[c]: float(point[nm][k]) for k, c in enumerate(cids_sorted)}
                        for nm in names},
              "iters": args.iters, "classes": [cat_names[c] for c in cids_sorted],
              "per_class": {}, "mAP": {}}
    # use the model that best reproduces the logs as the reference for the sd table
    ref = names[0]
    ref = "fuse_l4u3" if "fuse_l4u3" in names else names[0]

    print("\n=== bootstrap sd, reference model = %s (n=%d draws) ===" % (ref, args.iters))
    print("%-14s %6s %8s %8s %8s %9s %8s"
          % ("class", "n_gt", "AP", "boot_sd", "var_share%", "p05", "p95"))
    # variance of mAP decomposed as (1/K)^2 * sum_k Var(AP_k) + cross terms
    var_each = np.array([np.nanvar(acc[ref][:, k], ddof=1) for k in range(K)])
    var_total = np.nansum(var_each)
    mAPs = np.nanmean(acc[ref], axis=1)
    lo, hi = np.percentile(mAPs, [2.5, 97.5])
    for k, cid in enumerate(cids_sorted):
        s = acc[ref][:, k]
        n_valid = int(np.sum(~np.isnan(s)))
        sd = float(np.nanstd(s, ddof=1))
        p05, p95 = np.nanpercentile(s, [5, 95])
        share = var_each[k] / var_total * 100
        print("%-14s %6d %8.3f %8.3f %8.1f %8.3f %8.3f   (valid draws %d/%d)"
              % (cat_names[cid], n_gt[cid], point[ref][k], sd, share, p05, p95,
                 n_valid, args.iters))
        report["per_class"][cat_names[cid]] = {
            "n_gt": int(n_gt[cid]), "AP": float(point[ref][k]),
            "boot_sd": sd, "boot_p05": float(p05), "boot_p95": float(p95),
            "valid_draws": n_valid, "var_share_pct": float(share),
        }
    print("\nmAP  point=%.4f  boot_mean=%.4f  sd=%.4f  95%% CI=[%.4f, %.4f]"
          % (point[ref].mean(), mAPs.mean(), mAPs.std(ddof=1), lo, hi))
    print("sd if classes were independent = sqrt(sum var/K^2) = %.4f"
          % np.sqrt(var_total / (K ** 2)))
    report["mAP"] = {"point": float(point[ref].mean()), "boot_mean": float(mAPs.mean()),
                     "sd": float(mAPs.std(ddof=1)), "ci95": [float(lo), float(hi)],
                     "sd_independent": float(np.sqrt(var_total / (K ** 2))),
                     "reference_model": ref}

    # ---- what if the 5 rarest classes were excluded?
    # (<=35 boxes each: tricycle 2, ball 13, boat 14, uav 19, garbage can 35)
    tail = {"tricycle", "ball", "boat", "uav", "garbage can"}
    keep = [k for k, c in enumerate(cids_sorted) if cat_names[c] not in tail]
    mAP_keep = np.nanmean(acc[ref][:, keep], axis=1)
    n_gt_keep = sum(n_gt[c] for c in cids_sorted if cat_names[c] not in tail)
    n_gt_tail = sum(n_gt[c] for c in cids_sorted if cat_names[c] in tail)
    print("\n=== excluding the 4 long-tail classes (%s) ===" % ", ".join(sorted(tail)))
    print("kept: %d classes / %d boxes (%.1f%% of annotations)"
          % (len(keep), n_gt_keep, 100.0 * n_gt_keep / len(base["annotations"])))
    print("dropped: 4 classes / %d boxes (%.1f%% of annotations)"
          % (n_gt_tail, 100.0 * n_gt_tail / len(base["annotations"])))
    print("mAP(8 cls) point=%.4f  boot_sd=%.4f  (vs full 12-class sd=%.4f)"
          % (point[ref][keep].mean(), mAP_keep.std(ddof=1), mAPs.std(ddof=1)))
    report["excl_longtail"] = {
        "classes_kept": [cat_names[c] for c in cids_sorted if cat_names[c] not in tail],
        "mAP_point": float(point[ref][keep].mean()),
        "boot_sd": float(mAP_keep.std(ddof=1)),
        "n_gt_kept": int(n_gt_keep), "n_gt_dropped": int(n_gt_tail),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nsaved -> %s" % args.out_json)


if __name__ == "__main__":
    main()
