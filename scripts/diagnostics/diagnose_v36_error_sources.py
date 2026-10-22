#!/usr/bin/env python3
"""Decompose V36 val errors per GT: no-candidate / class-confusion / localization.

Read-only. Consumes the COCO-format prediction dump and the annotation file;
no model, no GPU, no training. Coordinates are original image space (verified:
median (x+w)/W == 1.000).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SMALL, MEDIUM = 32 ** 2, 96 ** 2


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


def size_bucket(area):
    return 'small' if area < SMALL else ('medium' if area < MEDIUM else 'large')


def bucket_of(iou_same, iou_any):
    if iou_same >= 0.75:
        return 'tight'            # localised well enough for AP75
    if iou_same >= 0.50:
        return 'loose'            # counts at AP50, lost at AP75
    if iou_any >= 0.50:
        return 'class_confusion'  # a good box exists, wrong label
    if iou_same >= 0.10:
        return 'coarse'           # overlapping but below AP50
    return 'no_candidate'         # nothing anywhere near


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--min-score', type=float, default=0.0)
    args = p.parse_args()

    ann = json.loads(args.annotations.read_text())
    preds = json.loads(args.predictions.read_text())
    images = {im['id']: im for im in ann['images']}
    cats = {c['id']: c['name'] for c in ann['categories']}

    by_img = defaultdict(list)
    for r in preds:
        if r['score'] >= args.min_score:
            by_img[r['image_id']].append(r)
    for k in by_img:
        by_img[k].sort(key=lambda r: -r['score'])

    rows = []
    for a in ann['annotations']:
        if a.get('iscrowd', 0):
            continue
        rows.append({'ann': a, 'gt_cat': a['category_id'], 'area': a['area']})

    gt_by_img = defaultdict(list)
    for i, r in enumerate(rows):
        gt_by_img[r['ann']['image_id']].append(i)

    rank_hist = Counter()
    confused = Counter()
    per_cat = defaultdict(lambda: Counter())
    iou_same_all, iou_any_all = [], []

    for img_id, idxs in gt_by_img.items():
        gts = [rows[i]['ann'] for i in idxs]
        prs = by_img.get(img_id, [])
        gb = xywh_to_xyxy([g['bbox'] for g in gts])
        pb = xywh_to_xyxy([r['bbox'] for r in prs]) if prs else np.zeros((0, 4))
        pc = np.array([r['category_id'] for r in prs]) if prs else np.zeros((0,), int)
        ps = np.array([r['score'] for r in prs]) if prs else np.zeros((0,))
        M = iou_matrix(gb, pb)

        for j, i in enumerate(idxs):
            same = np.where(pc == rows[i]['gt_cat'])[0] if len(pc) else np.zeros((0,), int)
            iou_same = float(M[j, same].max()) if len(same) else 0.0
            iou_any = float(M[j].max()) if M.shape[1] else 0.0
            rows[i]['iou_same'] = iou_same
            rows[i]['iou_any'] = iou_any
            rows[i]['bucket'] = bucket_of(iou_same, iou_any)
            iou_same_all.append(iou_same)
            iou_any_all.append(iou_any)
            per_cat[rows[i]['gt_cat']][rows[i]['bucket']] += 1

            if rows[i]['bucket'] == 'class_confusion' and M.shape[1]:
                confused[(cats.get(rows[i]['gt_cat'], rows[i]['gt_cat']),
                          cats.get(int(pc[int(M[j].argmax())]), int(pc[int(M[j].argmax())])))] += 1
            if len(same) and iou_same >= 0.5:
                best = same[int(np.argmax(M[j, same]))]
                rank = int(np.where(np.argsort(-ps) == best)[0][0]) + 1
                rank_hist['rank1' if rank == 1 else ('rank<=3' if rank <= 3 else
                                                     ('rank<=10' if rank <= 10 else 'rank>10'))] += 1

    total = len(rows)
    summary = {
        'total_gt': total,
        'prediction_count': sum(len(v) for v in by_img.values()),
        'min_score': args.min_score,
        'bucket_overall': dict(Counter(r['bucket'] for r in rows)),
        'bucket_by_size': {s: dict(Counter(r['bucket'] for r in rows if size_bucket(r['area']) == s))
                           for s in ('small', 'medium', 'large')},
        'gt_by_size': dict(Counter(size_bucket(r['area']) for r in rows)),
        'iou_same_quantiles': {q: float(np.quantile(iou_same_all, q))
                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        'iou_any_quantiles': {q: float(np.quantile(iou_any_all, q))
                              for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        'tp_score_rank': dict(rank_hist),
        'class_confusion_pairs': [{'gt': g, 'pred': p, 'n': n}
                                  for (g, p), n in confused.most_common(10)],
        'per_category': {cats.get(c, c): dict(v) for c, v in sorted(per_cat.items())},
        'scope': 'Per-GT geometry and ranking only. Does not reproduce mAP; no model, no GPU.',
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'error_sources.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('total_gt', 'bucket_overall', 'gt_by_size',
                                              'bucket_by_size', 'tp_score_rank')},
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
