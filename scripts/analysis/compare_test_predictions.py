#!/usr/bin/env python3
"""Compare several models' predictions on an unlabelled test set against one reference model (CPU, no labels).

The test set has no annotations, so nothing here is a score or an inferred label: it describes each model's own
output and where it disagrees with the reference. Every model goes through the same post-processing (class-wise NMS,
a common / rare-class score threshold, a per-image cap filled rare classes first) so post-processing differences do
not masquerade as model differences.

Inputs are raw detection caches, torch-saved dicts {image stem: [(cls, cx, cy, w, h, score, ...), ...]} with
normalized cxcywh boxes, as written by the historical inference scripts. Reported per model: boxes per image, class
and size mix, same-class overlaps that survived NMS (IoU in [0.5, NMS]), same-class nested boxes (one >= 80% inside
the other), cross-class overlaps (IoU >= --cross-iou, by class pair). Against the reference: matched / ref-only /
model-only boxes by class and size, class disagreements, IoU and log-area ratio of matched pairs, and for ref-only
boxes whether the model had a same-class box at IoU >= 0.5 below the threshold in its raw cache (low score) or not.
Lists the images with most disagreement for visual review. OUT must not exist.

A detector always produces candidates, so an image with no raw candidates in some cache means that cache's export
failed there (the 2026-10-04 F5 reproduction on ailab40 has none for the 4th image of every batch of 4, 249 images,
while the official F5 package had 7,184 boxes). Such images are dropped for every model and listed in the report;
--keep-empty-raw keeps them. A cache built from a thresholded submission legitimately has empty images; name it with
--allow-empty so its empty images are not taken for failed exports.

Example (on ailab40):
  python compare_test_predictions.py --images AIC_V29/dataset/test/AIC2026_PHASE_2_1000/visible --reference F5 \
    --model F5=outputs/f5_reproduce/raw_predictions.pt --model V36=.../raw_detections_v36.pt --out OUT
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

SIZES = (('small', 0, 32 ** 2), ('medium', 32 ** 2, 96 ** 2), ('large', 96 ** 2, float('inf')))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def inside(a, b):
    """Fraction of each a-box lying inside each b-box."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / np.maximum(np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)[:, None], 1e-12)


def nms(boxes, scores, thr):
    order, keep = np.argsort(-scores, kind='stable'), []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0] <= thr] if len(order) > 1 else order[:0]
    return np.asarray(keep, np.int64)


def size_of(area):
    return next(n for n, lo, hi in SIZES if lo <= area < hi)


def load_cache(path, sizes):
    import torch
    raw = torch.load(path, map_location='cpu', weights_only=False)
    out = {}
    for stem, (w, h) in sizes.items():
        rows = raw.get(stem, [])
        if len(rows):
            a = np.asarray([[float(v) for v in r[:6]] for r in rows], np.float64)
            cls, cx, cy, bw, bh, s = a.T
            boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
            out[stem] = (cls.astype(np.int64), boxes, s)
        else:
            out[stem] = (np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0))
    missing = [s for s in sizes if s not in raw]
    return out, missing


def process(cls, boxes, scores, a, rare):
    """Class-wise NMS, thresholds, rare-first cap. Returns (nms_set, final_set) index arrays."""
    keep = []
    for c in np.unique(cls):
        m = np.nonzero(cls == c)[0]
        keep.extend(m[nms(boxes[m], scores[m], a.nms)])
    keep = np.asarray(sorted(keep, key=lambda i: -scores[i]), np.int64)
    thr = np.asarray([a.thr_rare if c in rare else a.thr for c in cls[keep]]) if len(keep) else np.zeros(0)
    final = keep[scores[keep] >= thr] if len(keep) else keep
    final = np.asarray(sorted(final, key=lambda i: (cls[i] not in rare, -scores[i])), np.int64)[:a.cap]
    return keep, np.asarray(sorted(final, key=lambda i: -scores[i]), np.int64)


def self_stats(dets, names):
    n_img = len(dets)
    counts = [len(f) for _, _, _, f in dets.values()]
    cls_c, size_c, dup_pairs, nest_pairs, cross = Counter(), Counter(), Counter(), Counter(), Counter()
    for stem, (cls, boxes, s, f) in dets.items():
        c, b = cls[f], boxes[f]
        for k, bb in zip(c, b):
            cls_c[names.get(int(k), int(k))] += 1
            size_c[size_of(float((bb[2] - bb[0]) * (bb[3] - bb[1])))] += 1
        if len(f) < 2:
            continue
        ious, ins = iou_matrix(b, b), inside(b, b)
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                if c[i] == c[j]:
                    if ious[i, j] >= 0.5:
                        dup_pairs[names.get(int(c[i]), int(c[i]))] += 1
                    elif max(ins[i, j], ins[j, i]) >= 0.8:
                        nest_pairs[names.get(int(c[i]), int(c[i]))] += 1
                elif ious[i, j] >= ARGS.cross_iou:
                    pair = tuple(sorted((names.get(int(c[i]), int(c[i])), names.get(int(c[j]), int(c[j])))))
                    cross[' / '.join(pair)] += 1
    total = sum(counts)
    return dict(images=n_img, boxes=total, per_image_mean=round(total / max(n_img, 1), 3), per_image_max=max(counts, default=0),
                empty_images=sum(1 for v in counts if v == 0), classes=dict(cls_c.most_common()),
                sizes={k: size_c.get(k, 0) for k, _, _ in SIZES},
                same_class_overlap_pairs=dict(dup_pairs.most_common()), same_class_nested_pairs=dict(nest_pairs.most_common()),
                cross_class_pairs=dict(cross.most_common(15)))


def compare(model, ref, names, a, rare):
    by_cls, by_size = defaultdict(Counter), defaultdict(Counter)
    ious, log_area, score_pairs, class_disagree = [], [], [], Counter()
    ref_only_kind, per_image = Counter(), {}
    for stem in ref:
        mc, mb, ms, mf = model[stem]
        rc, rb, rs, rf = ref[stem]
        M, R = mf.tolist(), rf.tolist()
        iou = iou_matrix(rb[R], mb[M]) if R and M else np.zeros((len(R), len(M)))
        used, matched_ref = set(), set()
        for i in np.argsort(-rs[R], kind='stable') if R else []:
            best, best_iou = -1, 0.5
            for j in range(len(M)):
                if j not in used and mc[M[j]] == rc[R[i]] and iou[i, j] >= best_iou:
                    best, best_iou = j, iou[i, j]
            if best >= 0:
                used.add(best); matched_ref.add(i)
                ious.append(best_iou)
                ra = (rb[R[i], 2] - rb[R[i], 0]) * (rb[R[i], 3] - rb[R[i], 1])
                ma = (mb[M[best], 2] - mb[M[best], 0]) * (mb[M[best], 3] - mb[M[best], 1])
                log_area.append(float(np.log(max(ma, 1e-6) / max(ra, 1e-6))))
                score_pairs.append((float(rs[R[i]]), float(ms[M[best]])))
        n_ref_only = n_model_only = 0
        for i in range(len(R)):
            k = names.get(int(rc[R[i]]), int(rc[R[i]]))
            sz = size_of(float((rb[R[i], 2] - rb[R[i], 0]) * (rb[R[i], 3] - rb[R[i], 1])))
            if i in matched_ref:
                by_cls[k]['matched'] += 1; by_size[sz]['matched'] += 1
                continue
            n_ref_only += 1
            by_cls[k]['ref_only'] += 1; by_size[sz]['ref_only'] += 1
            other = [j for j in range(len(M)) if j not in used and iou[i, j] >= 0.5]
            if other:
                class_disagree[f"{k} -> {names.get(int(mc[M[other[0]]]), int(mc[M[other[0]]]))}"] += 1
                ref_only_kind['other_class'] += 1
                continue
            raw = np.nonzero(mc == rc[R[i]])[0]
            raw_iou = iou_matrix(rb[R[i]:R[i] + 1], mb[raw])[0] if len(raw) else np.zeros(0)
            if (raw_iou >= 0.5).any():
                thr = a.thr_rare if int(rc[R[i]]) in rare else a.thr
                ref_only_kind['model_low_score' if ms[raw[raw_iou >= 0.5]].max() < thr else 'model_suppressed'] += 1
            elif len(raw_iou) and raw_iou.max() >= 0.1:
                ref_only_kind['model_poor_box'] += 1
            else:
                ref_only_kind['model_no_box'] += 1
        for j in range(len(M)):
            if j not in used:
                n_model_only += 1
                k = names.get(int(mc[M[j]]), int(mc[M[j]]))
                sz = size_of(float((mb[M[j], 2] - mb[M[j], 0]) * (mb[M[j], 3] - mb[M[j], 1])))
                by_cls[k]['model_only'] += 1; by_size[sz]['model_only'] += 1
        per_image[stem] = n_ref_only + n_model_only
    sp = np.asarray(score_pairs) if score_pairs else np.zeros((0, 2))
    return dict(
        matched=int(sum(v['matched'] for v in by_cls.values())), ref_only=int(sum(v['ref_only'] for v in by_cls.values())),
        model_only=int(sum(v['model_only'] for v in by_cls.values())), ref_only_kind=dict(ref_only_kind),
        class_disagreements=dict(class_disagree.most_common(15)),
        by_class={k: dict(v) for k, v in sorted(by_cls.items(), key=lambda kv: -sum(kv[1].values()))},
        by_size={k: dict(by_size[k]) for k, _, _ in SIZES},
        matched_iou_median=round(float(np.median(ious)), 4) if ious else None,
        matched_iou_ge_0_75=round(float(np.mean(np.asarray(ious) >= 0.75)), 4) if ious else None,
        matched_log_area_ratio_median=round(float(np.median(log_area)), 4) if log_area else None,
        matched_model_larger_share=round(float(np.mean(np.asarray(log_area) > 0)), 4) if log_area else None,
        matched_score_median=dict(ref=round(float(np.median(sp[:, 0])), 4), model=round(float(np.median(sp[:, 1])), 4)) if len(sp) else None,
        top_disagreement_images=[s for s, _ in sorted(per_image.items(), key=lambda kv: -kv[1])[:a.top_images]])


def render(res):
    out = [f"# Test-set prediction comparison (no labels)", '',
           f"Post-processing for every model: {res['settings']}", '',
           f"Images with no raw candidates, by cache: {res['empty_raw']} (thresholded, kept: {res['allow_empty'] or 'none'}); "
           + (f"{len(res['excluded_images'])} such images dropped for every model, {res['images']} compared."
              if res['excluded_images'] else f"none dropped, {res['images']} compared."), '',
           '## Each model on its own', '',
           '| model | boxes | per image | max | empty | small | medium | large | same-class IoU>=0.5 pairs | nested pairs | cross-class pairs |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for m, s in res['self'].items():
        out.append(f"| {m} | {s['boxes']} | {s['per_image_mean']} | {s['per_image_max']} | {s['empty_images']} | "
                   + ' | '.join(str(s['sizes'][k]) for k, _, _ in SIZES)
                   + f" | {sum(s['same_class_overlap_pairs'].values())} | {sum(s['same_class_nested_pairs'].values())} | {sum(s['cross_class_pairs'].values())} |")
    out += ['', '### Class mix (boxes)', '', '| class | ' + ' | '.join(res['self']) + ' |', '|---|' + '---|' * len(res['self'])]
    classes = sorted({k for s in res['self'].values() for k in s['classes']})
    for k in classes:
        out.append(f'| {k} | ' + ' | '.join(str(s['classes'].get(k, 0)) for s in res['self'].values()) + ' |')
    for m, s in res['self'].items():
        out += ['', f"**{m}** same-class overlaps by class: {s['same_class_overlap_pairs']}; nested: {s['same_class_nested_pairs']}; cross-class: {s['cross_class_pairs']}"]
    out += ['', f"## Against the reference {res['reference']}", '',
            '| model | matched | ref-only | model-only | ref-only: low score / suppressed / poor box / no box / other class | matched IoU median | IoU>=.75 | log area ratio median | model larger | score median ref/model |',
            '|---|---|---|---|---|---|---|---|---|---|']
    for m, c in res['vs_reference'].items():
        k = c['ref_only_kind']
        out.append(f"| {m} | {c['matched']} | {c['ref_only']} | {c['model_only']} | {k.get('model_low_score', 0)} / {k.get('model_suppressed', 0)} / "
                   f"{k.get('model_poor_box', 0)} / {k.get('model_no_box', 0)} / {k.get('other_class', 0)} | {c['matched_iou_median']} | {c['matched_iou_ge_0_75']} | "
                   f"{c['matched_log_area_ratio_median']} | {c['matched_model_larger_share']} | {c['matched_score_median']} |")
    for m, c in res['vs_reference'].items():
        out += ['', f'### {m} vs {res["reference"]}', '', '| class | matched | ref-only | model-only |', '|---|---|---|---|']
        out += [f"| {k} | {v.get('matched', 0)} | {v.get('ref_only', 0)} | {v.get('model_only', 0)} |" for k, v in c['by_class'].items()]
        out += [f"| size:{k} | {v.get('matched', 0)} | {v.get('ref_only', 0)} | {v.get('model_only', 0)} |" for k, v in c['by_size'].items()]
        out += ['', f"Class disagreements (ref -> model): {c['class_disagreements']}", '',
                f"Most-disagreeing images: {c['top_disagreement_images']}"]
    out += ['', 'The reference is another model, not ground truth: ref-only and model-only boxes are disagreements, not errors.']
    return '\n'.join(out) + '\n'


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', type=Path, required=True, help='test RGB folder, for image sizes')
    p.add_argument('--model', action='append', required=True, metavar='NAME=CACHE')
    p.add_argument('--reference', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--thr', type=float, default=0.50)
    p.add_argument('--thr-rare', type=float, default=0.45)
    p.add_argument('--rare-classes', default='1,7,9,10,11')
    p.add_argument('--cap', type=int, default=60)
    p.add_argument('--nms', type=float, default=0.6)
    p.add_argument('--cross-iou', type=float, default=0.6)
    p.add_argument('--top-images', type=int, default=20)
    p.add_argument('--class-names', default='person,boat,animal,seat,sign,bicycle,car,ball,light,garbage can,uav,tricycle')
    p.add_argument('--keep-empty-raw', action='store_true', help='keep images where some cache has no raw candidates')
    p.add_argument('--allow-empty', nargs='+', default=[], metavar='NAME',
                   help='models whose caches are thresholded, so an empty image there is a real empty result')
    a = ARGS = p.parse_args()
    if a.out.exists():
        sys.exit(f'refusing: {a.out} exists')
    from PIL import Image
    sizes = {}
    for f in sorted(a.images.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS:
            with Image.open(f) as im:
                sizes[f.stem] = im.size
    names = dict(enumerate(a.class_names.split(',')))
    rare = {int(v) for v in a.rare_classes.split(',')}
    models = dict(m.split('=', 1) for m in a.model)
    if a.reference not in models:
        sys.exit('--reference must be one of the --model names')
    dets, res = {}, dict(settings=dict(nms=a.nms, thr=a.thr, thr_rare=a.thr_rare, rare_classes=sorted(rare), cap=a.cap, rare_first=True),
                         reference=a.reference, inputs={}, self={}, vs_reference={})
    caches, empty_raw = {}, {}
    for name, path in models.items():
        caches[name], missing = load_cache(path, sizes)
        res['inputs'][name] = dict(path=str(Path(path).resolve()), missing_images=len(missing))
        empty_raw[name] = sorted(stem for stem, (cls, _, _) in caches[name].items() if len(cls) == 0)
    if set(a.allow_empty) - set(models):
        sys.exit(f'--allow-empty names unknown models: {sorted(set(a.allow_empty) - set(models))}')
    excluded = set() if a.keep_empty_raw else {s for k, v in empty_raw.items() if k not in a.allow_empty for s in v}
    res.update(empty_raw={k: len(v) for k, v in empty_raw.items()}, allow_empty=a.allow_empty, excluded_images=sorted(excluded),
               images=len(sizes) - len(excluded))
    for name, cache in caches.items():
        dets[name] = {}
        for stem, (cls, boxes, s) in cache.items():
            if stem in excluded:
                continue
            keep, final = process(cls, boxes, s, a, rare)
            dets[name][stem] = (cls, boxes, s, final)
        res['self'][name] = self_stats(dets[name], names)
        print(f"{name}: {res['self'][name]['boxes']} boxes after post-processing, "
              f"{res['inputs'][name]['missing_images']} images missing, {len(empty_raw[name])} with no raw candidates", flush=True)
    if excluded:
        print(f'dropped {len(excluded)} images with no raw candidates in some cache', flush=True)
    for name in models:
        if name != a.reference:
            res['vs_reference'][name] = compare(dets[name], dets[a.reference], names, a, rare)
    a.out.mkdir(parents=True)
    (a.out/'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    (a.out/'report.md').write_text(render(res))
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
