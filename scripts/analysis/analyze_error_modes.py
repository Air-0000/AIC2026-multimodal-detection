#!/usr/bin/env python3
"""Break a model's validation errors into causes from all decoder queries (CPU, no model, no test data).

Input is decoder_outputs.npz from scripts/diagnostics/export_query_outputs.py plus the COCO annotation file.
Detections are rebuilt the way PostProcessor does (top-k over all query x class sigmoid scores), then class-wise
NMS, a score threshold (one value, or a common / rare-class pair) and a per-image cap, and matched to ground truth
COCO-style at IoU 0.5 (per class, highest score first, best-IoU unmatched GT). Reported, by object size and class:

  query redundancy  queries whose box has IoU >= 0.5 with a GT (any class, and with GT-class prob >= --cand-floor)
  misses            no_proposal (best IoU of any query < 0.1), localization (< 0.5), low_score (an IoU >= 0.5 box
                    exists but its GT-class prob is below threshold), wrong_class (no GT-class box above threshold
                    but another class is), suppressed (a qualifying box existed and was lost to NMS, cap or matching)
  false positives   duplicate (IoU >= 0.5 with a same-class GT matched by a higher-scored box), cross_class (IoU >= 0.5
                    with a GT of another class), localization part / oversize / offset (same-class IoU in [0.1, 0.5),
                    box inside the GT / GT inside the box by >= 80% / neither), background (IoU < 0.1 with every GT)
  box tightness     IoU of matched boxes, signed edge offsets and log width / height ratios relative to the GT
  score-IoU gap     IoU of the kept (top-scored) box minus the best IoU among candidate boxes of the GT class
  shift stability   with several arms: GT matched in the reference arm and missed in a shifted arm, by redundancy

Also reports standard COCO mAP (NMS, cap 100, no score cut) so the export can be checked against known numbers.
OUT must not exist; writes report.md, results.json, per_gt.csv and false_positives.csv.

Example (repo root):
  python scripts/analysis/analyze_error_modes.py --decoder-npz work_dir/audits/query_outputs_v1_best/decoder_outputs.npz \
    --annotations work_dir/deimv2_dataset/valid/annotations/instances_val.json --thr 0.50 --thr-rare 0.45 \
    --cap 60 --rare-first --out work_dir/audits/error_modes_v1_best
"""
import argparse
import contextlib
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

SIZES = (('small', 0, 32 ** 2), ('medium', 32 ** 2, 96 ** 2), ('large', 96 ** 2, float('inf')))
MISS_CAUSES = ('no_proposal', 'localization', 'low_score', 'wrong_class', 'suppressed')
FP_CAUSES = ('duplicate', 'cross_class', 'loc_part', 'loc_oversize', 'loc_offset', 'background')
REDUNDANCY_BINS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 9))


def iou_matrix(a, b):
    """IoU between xyxy boxes a [N, 4] and b [M, 4]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def inside_fraction(a, b):
    """Fraction of each box in a [N, 4] that lies inside each box in b [M, 4]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / np.maximum(np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)[:, None], 1e-12)


def nms(boxes, scores, thr):
    order, keep = np.argsort(-scores, kind='stable'), []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0] <= thr]  # torchvision suppresses IoU > thr
    return np.asarray(keep, np.int64)


def size_of(area):
    return next(name for name, lo, hi in SIZES if lo <= area < hi)


def bin_of(n):
    return next(f'{lo}' if lo == hi else (f'{lo}+' if hi > 10 ** 8 else f'{lo}-{hi}') for lo, hi in REDUNDANCY_BINS if lo <= n <= hi)


def candidates(probs, boxes, topk, nms_thr):
    """PostProcessor-style top-k query/class pairs, then class-wise NMS; returns (query, class, score, box) arrays."""
    flat = probs.reshape(-1)
    idx = np.argsort(-flat, kind='stable')[:topk]
    q, c, s = idx // probs.shape[1], idx % probs.shape[1], flat[idx]
    b = boxes[q]
    keep = []
    for cls in np.unique(c):
        m = np.nonzero(c == cls)[0]
        keep.extend(m[nms(b[m], s[m], nms_thr)] if nms_thr > 0 else m)
    keep = np.asarray(sorted(keep, key=lambda i: -s[i]), np.int64)
    return q[keep], c[keep], s[keep], b[keep]


def operate(c, s, thr, rare, cap, rare_first):
    keep = np.nonzero(s >= np.asarray([thr[k] for k in c]))[0] if len(c) else np.zeros(0, np.int64)
    if rare_first:
        keep = np.asarray(sorted(keep, key=lambda i: (c[i] not in rare, -s[i])), np.int64)
    keep = keep[:cap] if cap > 0 else keep
    return np.asarray(sorted(keep, key=lambda i: -s[i]), np.int64)


def match(det_boxes, det_cls, det_scores, gt_boxes, gt_cls, iou_thr=0.5):
    """COCO-style greedy matching per class; returns det->gt index (-1) and gt->det index (-1)."""
    d2g, g2d = np.full(len(det_boxes), -1), np.full(len(gt_boxes), -1)
    ious = iou_matrix(det_boxes, gt_boxes)
    for i in np.argsort(-det_scores, kind='stable'):
        best, best_iou = -1, iou_thr
        for j in np.nonzero(gt_cls == det_cls[i])[0]:
            if g2d[j] < 0 and ious[i, j] >= best_iou:
                best, best_iou = j, ious[i, j]
        if best >= 0:
            d2g[i], g2d[best] = best, i
    return d2g, g2d


def load(npz_path, ann_path):
    data = np.load(npz_path)
    arms = sorted({k[:-len('_boxes')] for k in data.files if k.endswith('_boxes')},
                  key=lambda a: (a != 'dx0_dy0', a))
    ann = json.loads(Path(ann_path).read_text())
    images = {im['id']: im for im in ann['images']}
    gts = defaultdict(list)
    for g in ann['annotations']:
        if not g.get('iscrowd', 0):
            gts[g['image_id']].append(g)
    ids = data['image_ids'].tolist()
    if set(ids) != set(images):
        sys.exit(f'npz has {len(ids)} images, annotation {len(images)}; sets differ')
    for (w, h), i in zip(data['orig_sizes'].tolist(), ids):
        if (w, h) != (images[i]['width'], images[i]['height']):
            sys.exit(f'image {i}: npz orig_size {(w, h)} != annotation {(images[i]["width"], images[i]["height"])}')
    return data, arms, ann, images, gts, ids


def to_xyxy(norm_cxcywh, w, h):
    cx, cy, bw, bh = norm_cxcywh.T
    return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)


def analyze_arm(data, arm, images, gts, ids, a, thr, rare):
    """Per-GT rows, false-positive rows and COCO results for one arm."""
    rows, fps, coco_dets = [], [], []
    for n, image_id in enumerate(ids):
        im = images[image_id]
        boxes = to_xyxy(data[f'{arm}_boxes'][n], im['width'], im['height'])
        probs = 1 / (1 + np.exp(-data[f'{arm}_logits'][n].astype(np.float64)))
        q, c, s, b = candidates(probs, boxes, a.topk, a.nms)
        for k in np.argsort(-s, kind='stable')[:100]:
            x1, y1, x2, y2 = b[k].tolist()
            coco_dets.append(dict(image_id=image_id, category_id=int(c[k]), bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s[k])))
        op = operate(c, s, thr, rare, a.cap, a.rare_first)
        g = gts.get(image_id, [])
        gt_boxes = np.asarray([[x, y, x + w, y + h] for x, y, w, h in (t['bbox'] for t in g)], np.float64).reshape(-1, 4)
        gt_cls = np.asarray([t['category_id'] for t in g], np.int64)
        d2g, g2d = match(b[op], c[op], s[op], gt_boxes, gt_cls)
        q_iou = iou_matrix(boxes, gt_boxes)                       # all queries x GT
        for j, t in enumerate(g):
            cls, area = int(gt_cls[j]), float(t.get('area', t['bbox'][2] * t['bbox'][3]))
            ious, p_gt = q_iou[:, j], probs[:, cls]
            hit = ious >= 0.5
            scored = p_gt >= a.cand_floor
            x, y, w, h = t['bbox']
            w, h = max(w, 1e-6), max(h, 1e-6)
            row = dict(image_id=image_id, gt_id=t['id'], cls=cls, size=size_of(area), area=area,
                       interior=bool(x >= 8 and y >= 8 and x + w <= im['width'] - 8 and y + h <= im['height'] - 8),
                       red_any=int(hit.sum()), red_cls=int((hit & scored).sum()), max_iou=float(ious.max(initial=0)),
                       best_score_iou50=float(p_gt[hit].max(initial=0)), status='tp' if g2d[j] >= 0 else 'miss')
            cand = hit & scored
            row['best_iou_scored'] = float(ious[scored].max(initial=0))
            if g2d[j] >= 0:
                k = op[g2d[j]]
                pb = b[k]
                row.update(score=float(s[k]), iou=float(iou_matrix(pb[None], gt_boxes[j:j + 1])[0, 0]),
                           dx1=(pb[0] - x) / w, dy1=(pb[1] - y) / h, dx2=(pb[2] - x - w) / w, dy2=(pb[3] - y - h) / h,
                           log_w=float(np.log(max(pb[2] - pb[0], 1e-6) / w)), log_h=float(np.log(max(pb[3] - pb[1], 1e-6) / h)))
                row['iou_gap'] = row['iou'] - float(ious[cand].max(initial=row['iou']))
            else:
                if row['max_iou'] < 0.1:
                    cause = 'no_proposal'
                elif row['max_iou'] < 0.5:
                    cause = 'localization'
                elif row['best_score_iou50'] >= thr[cls]:
                    cause = 'suppressed'
                elif any(probs[hit][:, k2].max(initial=0) >= thr[k2] for k2 in range(probs.shape[1]) if k2 != cls):
                    cause = 'wrong_class'
                else:
                    cause = 'low_score'
                row['cause'] = cause
            rows.append(row)
        for i_op, k in enumerate(op):
            if d2g[i_op] >= 0:
                continue
            ious = iou_matrix(b[k:k + 1], gt_boxes)[0] if len(gt_boxes) else np.zeros(0)
            same, other = (gt_cls == c[k]), (gt_cls != c[k])
            same_max = float(ious[same].max(initial=0))
            if same_max >= 0.5:
                cause = 'duplicate'
            elif float(ious[other].max(initial=0)) >= 0.5:
                cause = 'cross_class'
            elif same_max >= 0.1:
                j = int(np.nonzero(same)[0][np.argmax(ious[same])])
                if inside_fraction(b[k:k + 1], gt_boxes[j:j + 1])[0, 0] >= 0.8:
                    cause = 'loc_part'
                elif inside_fraction(gt_boxes[j:j + 1], b[k:k + 1])[0, 0] >= 0.8:
                    cause = 'loc_oversize'
                else:
                    cause = 'loc_offset'
            else:
                cause = 'background' if float(ious.max(initial=0)) < 0.1 else 'loc_offset'
            bw, bh = b[k, 2] - b[k, 0], b[k, 3] - b[k, 1]
            fps.append(dict(image_id=image_id, cls=int(c[k]), score=float(s[k]), cause=cause,
                            size=size_of(float(bw * bh)), same_class_iou=same_max))
    return rows, fps, coco_dets


def coco_metrics(ann_path, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
        dt = gt.loadRes(dets) if dets else None
        if dt is None:
            return {}
        e = COCOeval(gt, dt, 'bbox')
        e.evaluate(); e.accumulate(); e.summarize()
    names = ['mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
    return {n: round(float(v), 6) for n, v in zip(names, e.stats[:6])}


def share(xs, cond):
    return round(sum(1 for x in xs if cond(x)) / len(xs), 4) if xs else None


def med(xs):
    return round(float(np.median(xs)), 4) if xs else None


def summarize(rows, fps, class_names):
    out = {}
    groups = [('all', lambda r: True)] + [(n, (lambda n: lambda r: r['size'] == n)(n)) for n, _, _ in SIZES]
    groups += [(f'class:{class_names.get(k, k)}', (lambda k: lambda r: r['cls'] == k)(k)) for k in sorted({r['cls'] for r in rows})]
    for name, f in groups:
        g = [r for r in rows if f(r)]
        tp = [r for r in g if r['status'] == 'tp']
        miss = [r for r in g if r['status'] == 'miss']
        fp = [x for x in fps if (name == 'all' or (name.startswith('class:') and class_names.get(x['cls'], x['cls']) == name[6:])
                                 or x['size'] == name)]
        out[name] = dict(
            gt=len(g), tp=len(tp), miss=len(miss), recall=round(len(tp) / len(g), 4) if g else None,
            fp=len(fp), precision=round(len(tp) / (len(tp) + len(fp)), 4) if tp or fp else None,
            miss_causes=dict(Counter(r['cause'] for r in miss)), fp_causes=dict(Counter(x['cause'] for x in fp)),
            redundancy_median=med([r['red_cls'] for r in g]), redundancy_any_median=med([r['red_any'] for r in g]),
            tp_iou_median=med([r['iou'] for r in tp]), tp_iou_ge75=share([r['iou'] for r in tp], lambda v: v >= 0.75),
            tp_iou_ge90=share([r['iou'] for r in tp], lambda v: v >= 0.9),
            log_w_median=med([r['log_w'] for r in tp]), log_h_median=med([r['log_h'] for r in tp]),
            wider_share=share([r['log_w'] for r in tp], lambda v: v > 0), taller_share=share([r['log_h'] for r in tp], lambda v: v > 0),
            edge_offsets_median={k: med([r[k] for r in tp]) for k in ('dx1', 'dy1', 'dx2', 'dy2')},
            iou_gap_median=med([r['iou_gap'] for r in tp]), iou_gap_le_minus_0_1=share([r['iou_gap'] for r in tp], lambda v: v <= -0.1))
    out['by_redundancy'] = {
        b: dict(gt=len(g), miss_rate=share(g, lambda r: r['status'] == 'miss'))
        for b in (bin_of(lo) for lo, _ in REDUNDANCY_BINS)
        for g in [[r for r in rows if bin_of(r['red_cls']) == b]]}
    return out


def shift_summary(ref_rows, arm_rows):
    ref = {r['gt_id']: r for r in ref_rows}
    pairs = [(ref[r['gt_id']], r) for r in arm_rows]
    out = {}
    for name, f in (('all', lambda r: True), ('interior', lambda r: r['interior'])):
        p = [(x, y) for x, y in pairs if f(x)]
        lost = [(x, y) for x, y in p if x['status'] == 'tp' and y['status'] == 'miss']
        rec = [(x, y) for x, y in p if x['status'] == 'miss' and y['status'] == 'tp']
        ref_tp = sum(1 for x, _ in p if x['status'] == 'tp')
        out[name] = dict(gt=len(p), ref_tp=ref_tp, lost=len(lost), recovered=len(rec),
                         lost_rate=round(len(lost) / ref_tp, 4) if ref_tp else None,
                         lost_causes=dict(Counter(y['cause'] for _, y in lost)),
                         lost_by_size=dict(Counter(x['size'] for x, _ in lost)),
                         score_drop_median_lost=med([x['best_score_iou50'] - y['best_score_iou50'] for x, y in lost]),
                         by_redundancy={b: dict(ref_tp=len(t), lost_rate=share(t, lambda xy: xy[1]['status'] == 'miss'))
                                        for b in (bin_of(lo) for lo, _ in REDUNDANCY_BINS)
                                        for t in [[(x, y) for x, y in p if x['status'] == 'tp' and bin_of(x['red_cls']) == b]]})
    return out


def table(title, header, rows):
    lines = [f'### {title}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join('' if v is None else str(v) for v in r) + ' |' for r in rows]
    return '\n'.join(lines) + '\n'


def render(res, a):
    s, names = res['arms'][res['reference']]['summary'], ['all'] + [n for n, _, _ in SIZES]
    cls_names = [k for k in s if k.startswith('class:')]
    out = [f"# Error modes: {a.decoder_npz}", '',
           f"Operating point: thr {res['settings']['thr']}, rare-class thr {res['settings']['thr_rare']} for {res['settings']['rare_classes']}, "
           f"cap {a.cap}{' rare-first' if a.rare_first else ''}, class-wise NMS {a.nms}; candidate floor {a.cand_floor}.", '',
           f"Standard COCO (NMS, cap 100, no score cut), reference arm: {res['arms'][res['reference']]['coco']}", '']
    out.append(table('Recall, precision and query redundancy', ['group', 'GT', 'TP', 'miss', 'recall', 'FP', 'precision', 'red. median (cls)', 'red. median (any)'],
                     [[n, s[n]['gt'], s[n]['tp'], s[n]['miss'], s[n]['recall'], s[n]['fp'], s[n]['precision'], s[n]['redundancy_median'], s[n]['redundancy_any_median']] for n in names + cls_names]))
    out.append(table('Miss causes', ['group'] + list(MISS_CAUSES), [[n] + [s[n]['miss_causes'].get(k, 0) for k in MISS_CAUSES] for n in names + cls_names]))
    out.append(table('False-positive causes', ['group'] + list(FP_CAUSES), [[n] + [s[n]['fp_causes'].get(k, 0) for k in FP_CAUSES] for n in names + cls_names]))
    out.append(table('Box tightness of matched boxes', ['group', 'IoU median', 'IoU>=.75', 'IoU>=.90', 'log w median', 'log h median', 'wider', 'taller', 'dx1', 'dy1', 'dx2', 'dy2', 'IoU gap median', 'gap<=-0.1'],
                     [[n, s[n]['tp_iou_median'], s[n]['tp_iou_ge75'], s[n]['tp_iou_ge90'], s[n]['log_w_median'], s[n]['log_h_median'], s[n]['wider_share'], s[n]['taller_share']]
                      + [s[n]['edge_offsets_median'][k] for k in ('dx1', 'dy1', 'dx2', 'dy2')] + [s[n]['iou_gap_median'], s[n]['iou_gap_le_minus_0_1']] for n in names + cls_names]))
    out.append(table('Miss rate by query redundancy (GT-class candidates with IoU >= 0.5)', ['candidates', 'GT', 'miss rate'],
                     [[b, v['gt'], v['miss_rate']] for b, v in s['by_redundancy'].items()]))
    for arm, sh in res.get('shift', {}).items():
        for scope in ('all', 'interior'):
            v = sh[scope]
            out.append(table(f'Shift {arm} vs {res["reference"]} ({scope} GT)', ['ref TP', 'lost', 'recovered', 'lost rate', 'lost causes', 'lost by size', 'median score drop (lost)'],
                             [[v['ref_tp'], v['lost'], v['recovered'], v['lost_rate'], v['lost_causes'], v['lost_by_size'], v['score_drop_median_lost']]]))
            out.append(table(f'Shift {arm}: lost rate by redundancy ({scope} GT)', ['candidates', 'ref TP', 'lost rate'],
                             [[b, x['ref_tp'], x['lost_rate']] for b, x in v['by_redundancy'].items()]))
    out += ['Notes: sizes use the COCO area of the GT in original pixels. Edge offsets are (pred - GT) / GT width or height, so',
            'dx1 < 0 and dx2 > 0 mean a wider box. IoU gap compares the kept box with the best-IoU GT-class candidate.',
            'Selected validation images were used for model selection, so this is diagnostic, not a held-out estimate.']
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--decoder-npz', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--thr', type=float, default=0.48, help='score threshold (common classes)')
    p.add_argument('--thr-rare', type=float, help='threshold for --rare-classes; default = --thr')
    p.add_argument('--rare-classes', default='1,7,9,10,11', help='boat, ball, garbage can, uav, tricycle')
    p.add_argument('--cap', type=int, default=100, help='max boxes per image at the operating point; 0 = none')
    p.add_argument('--rare-first', action='store_true', help='fill the cap with rare-class boxes first')
    p.add_argument('--nms', type=float, default=0.6)
    p.add_argument('--topk', type=int, default=300, help='query/class pairs kept before NMS, as PostProcessor')
    p.add_argument('--cand-floor', type=float, default=0.05, help='GT-class prob for a query to count as a candidate')
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f'refusing: {a.out} exists')
    data, arms, ann, images, gts, ids = load(a.decoder_npz, a.annotations)
    rare = {int(v) for v in a.rare_classes.split(',') if v.strip()}
    n_cls = data[f'{arms[0]}_logits'].shape[-1]
    thr = {k: (a.thr_rare if (a.thr_rare is not None and k in rare) else a.thr) for k in range(n_cls)}
    class_names = {c['id']: c['name'] for c in ann['categories']}
    res = dict(settings=dict(thr=a.thr, thr_rare=a.thr_rare if a.thr_rare is not None else a.thr, rare_classes=sorted(rare),
                             cap=a.cap, rare_first=a.rare_first, nms=a.nms, topk=a.topk, cand_floor=a.cand_floor),
               reference=arms[0], arms={})
    per_arm = {}
    for arm in arms:
        rows, fps, dets = analyze_arm(data, arm, images, gts, ids, a, thr, rare)
        per_arm[arm] = (rows, fps)
        res['arms'][arm] = dict(coco=coco_metrics(a.annotations, dets), summary=summarize(rows, fps, class_names))
        print(f"{arm}: COCO {res['arms'][arm]['coco']}", flush=True)
    if len(arms) > 1:
        res['shift'] = {arm: shift_summary(per_arm[arms[0]][0], per_arm[arm][0]) for arm in arms[1:]}
    a.out.mkdir(parents=True)
    (a.out/'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    (a.out/'report.md').write_text(render(res, a))
    rows, fps = per_arm[arms[0]]
    keys = sorted({k for r in rows for k in r})
    with (a.out/'per_gt.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (a.out/'false_positives.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'cls', 'score', 'cause', 'size', 'same_class_iou'])
        w.writeheader()
        w.writerows(fps)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
