#!/usr/bin/env python3
"""Corrected V36 val diagnostic: within-category precision@rank + output-level
confusion matrix with background row/column.

Matrix convention: rows = GT label, columns = prediction label, index 0 = BG.
  mat[g, d]   g,d >= 1 : matched, gt class g, predicted class d (off-diagonal = class error)
  mat[g, 0]   g >= 1   : GT with no matched detection  -> FN
  mat[0, d]   d >= 1   : detection with no matched GT   -> FP
Read-only: no model, no GPU, no training.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BG = 'BG'


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def greedy_match(iou, thr):
    """One-to-one greedy assignment by descending IoU. Returns {gt_idx: local_det_idx}."""
    if iou.size == 0:
        return {}
    pairs = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
             if iou[i, j] >= thr]
    pairs.sort(key=lambda t: -t[0])
    used_g, used_d, out = set(), set(), {}
    for _, i, j in pairs:
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        out[i] = j
    return out


def rank_stats(items):
    if not items:
        return {'tps': 0}
    r = np.asarray(items)
    return {'tps': int(len(r)),
            'rank1_pct': round(float((r == 1).mean() * 100), 2),
            'rank_le3_pct': round(float((r <= 3).mean() * 100), 2),
            'rank_le10_pct': round(float((r <= 10).mean() * 100), 2),
            'rank_gt10_pct': round(float((r > 10).mean() * 100), 2),
            'median_rank': float(np.median(r))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--conf-thresholds', type=float, nargs='+', default=[0.05, 0.30])
    p.add_argument('--iou-thresholds', type=float, nargs='+', default=[0.5, 0.75])
    args = p.parse_args()

    ann = json.loads(args.annotations.read_text())
    preds = json.loads(args.predictions.read_text())
    cat_names = {c['id']: c['name'] for c in ann['categories']}
    labels = [BG] + [cat_names[c] for c in sorted(cat_names)]
    idx_of = {name: i for i, name in enumerate(labels)}

    by_img = defaultdict(list)
    for r in preds:
        by_img[r['image_id']].append(r)
    gt_by_img = defaultdict(list)
    for a in ann['annotations']:
        if not a.get('iscrowd', 0):
            gt_by_img[a['image_id']].append(a)

    rank_acc = defaultdict(list)
    conf = defaultdict(lambda: np.zeros((len(labels), len(labels)), dtype=np.int64))

    for img_id, gts in gt_by_img.items():
        dets = sorted(by_img.get(img_id, []), key=lambda r: -r['score'])
        gb = xywh_to_xyxy([g['bbox'] for g in gts])
        db = xywh_to_xyxy([r['bbox'] for r in dets]) if dets else np.zeros((0, 4))
        dcat = np.array([r['category_id'] for r in dets], dtype=int) if dets else np.zeros((0,), int)
        M = iou_matrix(gb, db)

        # (A) within-category rank of every TP
        for thr in args.iou_thresholds:
            for c in np.unique([g['category_id'] for g in gts]):
                gi = [i for i, g in enumerate(gts) if g['category_id'] == c]
                di = [j for j in range(len(dets)) if dcat[j] == c]
                if not gi or not di:
                    continue
                rank_of = {j: k + 1 for k, j in enumerate(di)}
                for i_l, j_l in greedy_match(M[np.ix_(gi, di)], thr).items():
                    a = gts[gi[i_l]]
                    size = 'small' if a['area'] < 32 ** 2 else ('medium' if a['area'] < 96 ** 2 else 'large')
                    rank_acc[(thr, size)].append(rank_of[di[j_l]])
                    rank_acc[(thr, 'all')].append(rank_of[di[j_l]])

        # (B) class-agnostic one-to-one matching -> confusion
        for cthr in args.conf_thresholds:
            keep = [j for j in range(len(dets)) if dets[j]['score'] >= cthr]
            for thr in args.iou_thresholds:
                mat = conf[(cthr, thr)]
                if keep:
                    m = greedy_match(M[:, keep], thr)
                    taken = set(m.values())
                    for i_l, j_l in m.items():
                        mat[idx_of[cat_names[gts[i_l]['category_id']]], idx_of[cat_names[int(dcat[keep[j_l]])]]] += 1
                    for j_l in range(len(keep)):
                        if j_l not in taken:
                            mat[idx_of[BG], idx_of[cat_names[int(dcat[keep[j_l]])]]] += 1
                    for i_l in range(len(gts)):
                        if i_l not in m:
                            mat[idx_of[cat_names[gts[i_l]['category_id']]], idx_of[BG]] += 1
                else:
                    for i_l in range(len(gts)):
                        mat[idx_of[cat_names[gts[i_l]['category_id']]], idx_of[BG]] += 1

    report = {'scope': 'Within-category ranking and output-level confusion at fixed score '
                       'thresholds. Not mAP. No model, no GPU.',
              'matrix_convention': 'rows = GT label, cols = prediction label, index 0 = BG',
              'rank': {}, 'confusion': {}, 'per_class_at_0.30_iou0.5': {}}
    for (thr, size), ranks in sorted(rank_acc.items()):
        report['rank'].setdefault(str(thr), {})[size] = rank_stats(ranks)

    for (cthr, thr), mat in sorted(conf.items()):
        diag = int(np.trace(mat[1:, 1:]))
        off = int(mat[1:, 1:].sum() - diag)
        fn = int(mat[1:, 0].sum())
        fp = int(mat[0, 1:].sum())
        entry = {'gt_total': diag + off + fn, 'matched_correct': diag,
                 'matched_class_error': off, 'gt_unmatched_fn': fn,
                 'det_unmatched_fp': fp,
                 'precision': round(diag / max(diag + off + fp, 1), 4),
                 'recall_nonbg': round(diag / max(diag + off + fn, 1), 4),
                 'labels': labels, 'matrix': mat.tolist()}
        report['confusion'].setdefault(str(cthr), {})[str(thr)] = entry
        if cthr == 0.30 and thr == 0.5:
            for i, name in enumerate(labels[1:], start=1):
                report['per_class_at_0.30_iou0.5'][name] = {
                    'gt': int(mat[i].sum()), 'correct': int(mat[i, i]),
                    'class_err': int(mat[i, 1:].sum() - mat[i, i]),
                    'fn': int(mat[i, 0]), 'fp': int(mat[0, i])}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'precision_confusion.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    for thr, d in report['rank'].items():
        for size in ('all', 'small', 'medium', 'large'):
            if size in d:
                s = d[size]
                print(f"rank@iou{thr:>4} {size:6s} tps={s['tps']:5d} rank>10={s['rank_gt10_pct']:5.1f}% "
                      f"<=10={s['rank_le10_pct']:5.1f}% median={s['median_rank']}")
    for cthr, d in report['confusion'].items():
        for thr, m in d.items():
            print(f"conf@score{cthr}/iou{thr}: gt={m['gt_total']} correct={m['matched_correct']} "
                  f"classerr={m['matched_class_error']} fn={m['gt_unmatched_fn']} fp={m['det_unmatched_fp']} "
                  f"P={m['precision']} R={m['recall_nonbg']}")


if __name__ == '__main__':
    main()
