#!/usr/bin/env python3
"""Audit training annotations for anomalies and draw the flagged cases for visual review (CPU, no model).

Reads one or more COCO files of disjoint splits (e.g. train 1800 and valid 200) and the RGB images. Flags:
  geometry      boxes outside the image, degenerate (side < 1 px) or tiny (shorter side < --tiny px)
  duplicates    same-class pairs with IoU >= 0.95 (repeated rows) or in [0.7, 0.95) (double annotation)
  nested        same-class pairs where one box is >= 90% inside the other and IoU < 0.7 (part labelled separately)
  cross-class   pairs of different classes with IoU >= 0.8 (one object given two classes), by class pair
  outliers      per-class log-area or log-aspect z-score beyond +-3 (possible wrong class or wrong box)
  frames        near-identical frames by 64-bit difference hash (Hamming <= --hash-dist): pairs split across the
                given annotation files (train / val leakage) and pairs whose labels disagree (a box in one frame
                with no same-class box at IoU >= 0.3 in the other); camera or object motion between the frames also
                breaks the match, so on this data most disagreeing pairs are motion, not missing labels
  sources       label density by file-name source (6-digit, 8-digit, shuming_, hehe_, video sequence NNNNNN_NNN_
                with or without suppl_) and near-identical frames across sources with their box counts: the same
                scene labelled by two sources shows whether they follow one annotation convention
Also summarises boxes per image and sizes by image resolution. Writes report.md, results.json, flagged.json and,
under panels/, JPEGs of the worst cases with their labels drawn. OUT must not exist.

Example (on ailab40):
  python audit_train_labels.py --images AIC_V29/dataset/train/visible \
    --ann train=AIC_V29/work_dir/deimv2_dataset/train/annotations/instances_train.json \
    --ann valid=AIC_V29/work_dir/deimv2_dataset/valid/annotations/instances_val.json --out OUT
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


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
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / np.maximum(np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)[:, None], 1e-12)


def source(name):
    s = Path(name).stem
    if re.fullmatch(r'\d{6}', s):
        return 'num6'
    if re.fullmatch(r'\d{8}', s):
        return 'num8'
    if re.match(r'(shuming|hehe)_', s):
        return s.split('_')[0]
    if re.match(r'\d{6}_\d{3}_suppl_', s):
        return 'seq_suppl'
    return 'seq' if re.match(r'\d{6}_\d{3}_', s) else 'other'


def dhash(path, size=8):
    from PIL import Image
    with Image.open(path) as im:
        g = np.asarray(im.convert('L').resize((size + 1, size)), np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).reshape(-1))


def draw(path, anns, names, out, title, highlight=()):
    from PIL import Image, ImageDraw
    with Image.open(path) as im:
        im = im.convert('RGB')
    d = ImageDraw.Draw(im)
    for g in anns:
        x, y, w, h = g['bbox']
        c = COLORS[g['category_id'] % len(COLORS)]
        width = 6 if g['id'] in highlight else 2
        d.rectangle([x, y, x + w, y + h], outline=c, width=width)
        d.text((x + 2, max(0, y - 12)), f"{names.get(g['category_id'], g['category_id'])}#{g['id']}", fill=c)
    d.text((8, 8), title, fill=(255, 255, 255))
    im.thumbnail((1600, 1600))
    im.save(out, quality=88)


def side_by_side(paths, out):
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    h = min(i.height for i in ims)
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    canvas = Image.new('RGB', (sum(i.width for i in ims), h))
    x = 0
    for i in ims:
        canvas.paste(i, (x, 0)); x += i.width
    canvas.save(out, quality=88)
    for p in paths:
        Path(p).unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--ann', action='append', required=True, metavar='SPLIT=COCO_JSON')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tiny', type=float, default=4.0)
    p.add_argument('--hash-dist', type=int, default=4)
    p.add_argument('--panels', type=int, default=12, help='panels drawn per anomaly kind')
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f'refusing: {a.out} exists')
    images, anns, split_of, names = {}, defaultdict(list), {}, {}
    for spec in a.ann:
        split, path = spec.split('=', 1)
        d = json.loads(Path(path).read_text())
        names.update({c['id']: c['name'] for c in d['categories']})
        local = {}
        for im in d['images']:
            if im['file_name'] in split_of:
                sys.exit(f"{im['file_name']} appears in more than one --ann")
            split_of[im['file_name']] = split
            local[im['id']] = im['file_name']
            images[im['file_name']] = im
        for g in d['annotations']:
            anns[local[g['image_id']]].append(dict(g, file=local[g['image_id']]))
    flagged = defaultdict(list)
    per_img, by_res = {}, defaultdict(lambda: dict(images=0, boxes=0, areas=[]))
    feats = defaultdict(list)                                   # class -> (log area, log aspect, ann)
    for fname, im in images.items():
        W, H = im['width'], im['height']
        g = [x for x in anns.get(fname, []) if not x.get('iscrowd', 0)]
        per_img[fname] = len(g)
        r = by_res[f'{W}x{H}']
        r['images'] += 1; r['boxes'] += len(g)
        if not g:
            flagged['empty_image'].append(dict(file=fname))
            continue
        b = np.asarray([[x['bbox'][0], x['bbox'][1], x['bbox'][0] + x['bbox'][2], x['bbox'][1] + x['bbox'][3]] for x in g], np.float64)
        c = np.asarray([x['category_id'] for x in g])
        for x, bb in zip(g, b):
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            r['areas'].append(w * h)
            out_px = max(0 - bb[0], 0 - bb[1], bb[2] - W, bb[3] - H)
            if out_px > 0.5:
                flagged['outside_image'].append(dict(file=fname, ann=x['id'], cls=names[x['category_id']], by_px=round(float(out_px), 1)))
            if min(w, h) < 1:
                flagged['degenerate'].append(dict(file=fname, ann=x['id'], cls=names[x['category_id']], w=w, h=h))
            elif min(w, h) < a.tiny:
                flagged['tiny'].append(dict(file=fname, ann=x['id'], cls=names[x['category_id']], w=round(w, 1), h=round(h, 1)))
            if w > 0 and h > 0:
                feats[x['category_id']].append((np.log(w * h), np.log(w / h), x))
        ious, ins = iou_matrix(b, b), inside(b, b)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                rec = dict(file=fname, anns=[g[i]['id'], g[j]['id']], iou=round(float(ious[i, j]), 3))
                if c[i] == c[j]:
                    if ious[i, j] >= 0.95:
                        flagged['duplicate_exact'].append(dict(rec, cls=names[c[i]]))
                    elif ious[i, j] >= 0.7:
                        flagged['duplicate_near'].append(dict(rec, cls=names[c[i]]))
                    elif max(ins[i, j], ins[j, i]) >= 0.9:
                        flagged['nested_same_class'].append(dict(rec, cls=names[c[i]]))
                elif ious[i, j] >= 0.8:
                    flagged['cross_class'].append(dict(rec, classes=sorted([names[c[i]], names[c[j]]])))
    for k, rows in feats.items():
        la = np.asarray([r[0] for r in rows]); asp = np.asarray([r[1] for r in rows])
        for (l, s, x), za, zs in zip(rows, (la - la.mean()) / max(la.std(), 1e-6), (asp - asp.mean()) / max(asp.std(), 1e-6)):
            if abs(za) > 3 or abs(zs) > 3:
                flagged['class_outlier'].append(dict(file=x['file'], ann=x['id'], cls=names[k], area_z=round(float(za), 2),
                                                     aspect_z=round(float(zs), 2), bbox=[round(v, 1) for v in x['bbox']]))
    # near-identical frames
    files = sorted(images)
    by_src = defaultdict(list)
    for f in files:
        by_src[source(f)].append(f)
    hashes = np.stack([dhash(a.images/f) for f in files])
    bits = np.unpackbits(hashes, axis=1).astype(np.uint8)
    dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    pairs = [(i, j) for i, j in zip(*np.nonzero(np.triu(dist <= a.hash_dist, 1)))]
    leak, label_gap, cross_src = [], [], defaultdict(list)
    for i, j in pairs:
        fi, fj = files[i], files[j]
        if split_of[fi] != split_of[fj]:
            leak.append(dict(files=[fi, fj], splits=[split_of[fi], split_of[fj]], hamming=int(dist[i, j])))
        if source(fi) != source(fj):
            (sa, fa), (sb, fb) = sorted([(source(fi), fi), (source(fj), fj)])
            cross_src[f'{sa} vs {sb}'].append((fa, fb, per_img[fa], per_img[fb]))
        gi, gj = anns.get(fi, []), anns.get(fj, [])
        bi = np.asarray([[x['bbox'][0], x['bbox'][1], x['bbox'][0] + x['bbox'][2], x['bbox'][1] + x['bbox'][3]] for x in gi]).reshape(-1, 4)
        bj = np.asarray([[x['bbox'][0], x['bbox'][1], x['bbox'][0] + x['bbox'][2], x['bbox'][1] + x['bbox'][3]] for x in gj]).reshape(-1, 4)
        iou = iou_matrix(bi, bj)
        ci, cj = np.asarray([x['category_id'] for x in gi]), np.asarray([x['category_id'] for x in gj])
        only_i = [gi[k]['id'] for k in range(len(gi)) if not ((cj == ci[k]) & (iou[k] >= 0.3)).any()] if len(gi) else []
        only_j = [gj[k]['id'] for k in range(len(gj)) if not ((ci == cj[k]) & (iou[:, k] >= 0.3)).any()] if len(gj) else []
        if only_i or only_j:
            label_gap.append(dict(files=[fi, fj], hamming=int(dist[i, j]), boxes=[len(gi), len(gj)],
                                  unmatched=[only_i, only_j], gap=len(only_i) + len(only_j)))
    label_gap.sort(key=lambda r: -r['gap'])
    counts = np.asarray(list(per_img.values()))
    res = dict(images=len(images), annotations=int(sum(per_img.values())), splits=dict(Counter(split_of.values())),
               boxes_per_image=dict(mean=round(float(counts.mean()), 2), median=float(np.median(counts)), max=int(counts.max()),
                                    zero=int((counts == 0).sum())),
               by_resolution={k: dict(images=v['images'], boxes=v['boxes'], boxes_per_image=round(v['boxes'] / v['images'], 2),
                                      median_box_area=round(float(np.median(v['areas'])), 1) if v['areas'] else None)
                              for k, v in by_res.items()},
               flag_counts={k: len(v) for k, v in flagged.items()},
               cross_class_pairs=dict(Counter(' / '.join(r['classes']) for r in flagged['cross_class']).most_common()),
               duplicate_classes=dict(Counter(r['cls'] for r in flagged['duplicate_exact'] + flagged['duplicate_near']).most_common()),
               nested_classes=dict(Counter(r['cls'] for r in flagged['nested_same_class']).most_common()),
               outlier_classes=dict(Counter(r['cls'] for r in flagged['class_outlier']).most_common()),
               by_source={s: dict(images=len(fs), boxes_per_image=round(sum(per_img[f] for f in fs) / len(fs), 2),
                                  small_share_1920=round(sum(x['bbox'][2] * x['bbox'][3] * (1920 / images[f]['width']) ** 2 < 32 ** 2
                                                             for f in fs for x in anns.get(f, []) if not x.get('iscrowd', 0)) / max(sum(per_img[f] for f in fs), 1), 3))
                          for s, fs in sorted(by_src.items())},
               cross_source_near_identical={k: dict(pairs=len(v), mean_boxes=[round(float(np.mean([r[2] for r in v])), 2),
                                                                             round(float(np.mean([r[3] for r in v])), 2)],
                                                    examples=[list(r) for r in v[:5]])
                                            for k, v in sorted(cross_src.items(), key=lambda kv: -len(kv[1]))},
               near_identical_pairs=len(pairs), cross_split_pairs=len(leak),
               images_with_near_identical_frame_in_other_split=dict(Counter(s for s, f in {(s, f) for r in leak for f, s in zip(r['files'], r['splits'])})),
               label_disagreeing_pairs=len(label_gap), label_disagreeing_unmatched_boxes=int(sum(r['gap'] for r in label_gap)))
    a.out.mkdir(parents=True)
    panels = a.out/'panels'
    panels.mkdir()
    by_file = {f: anns.get(f, []) for f in images}
    for kind in ('duplicate_exact', 'duplicate_near', 'cross_class', 'nested_same_class', 'outside_image', 'class_outlier', 'tiny'):
        for n, r in enumerate(flagged.get(kind, [])[:a.panels]):
            hl = set(r.get('anns', [r.get('ann')]))
            draw(a.images/r['file'], by_file[r['file']], names, panels/f'{kind}_{n:02d}_{Path(r["file"]).stem}.jpg',
                 f'{kind}: {r}', highlight=hl)
    for n, r in enumerate(label_gap[:a.panels]):
        tmp = []
        for f, un in zip(r['files'], r['unmatched']):
            t = panels/f'_tmp_{Path(f).stem}.jpg'
            draw(a.images/f, by_file[f], names, t, f"{f}: {len(by_file[f])} boxes, unmatched {un}", highlight=set(un))
            tmp.append(t)
        side_by_side(tmp, panels/f'label_gap_{n:02d}_{Path(r["files"][0]).stem}_{Path(r["files"][1]).stem}.jpg')
    (a.out/'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    (a.out/'flagged.json').write_text(json.dumps(dict(flagged, cross_split_pairs=leak, label_disagreeing_pairs=label_gap),
                                                 indent=1, ensure_ascii=False, default=float) + '\n')
    lines = ['# Training label audit', '', '```', json.dumps(res, indent=1, ensure_ascii=False), '```', '',
             f'Panels of the first {a.panels} cases per kind are in panels/. Near-identical frames use a 64-bit difference hash',
             f'with Hamming distance <= {a.hash_dist}; a disagreeing pair has a box in one frame with no same-class box at IoU >= 0.3',
             'in the other; camera or object motion also breaks that match, so it is a list for visual review, not a count of',
             'missing labels. cross_source_near_identical compares box counts of the same scene labelled by two sources.']
    (a.out/'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(res['flag_counts'], ensure_ascii=False))
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
