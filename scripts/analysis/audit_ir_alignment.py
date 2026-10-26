#!/usr/bin/env python3
"""Check the IR images the model reads against RGB: RGB leakage, IR coverage and thermal alignment (CPU, no model).

  leak       per image, correlation of fine detail (Laplacian of Gaussian, sigma 1) between grey RGB and IR at zero shift
             over the IR-covered area. A thermal camera does not reproduce visible texture (printed text, window frames),
             so a high value means RGB is blended into the IR file. Structural RGB-IR matching is not used for alignment:
             on leaked files it locks onto the leaked RGB and reports a perfect match that says nothing about the heat.
  coverage   fraction of the frame where IR is exactly 0 (black padding left where the thermal camera saw nothing);
             a box counts as outside IR when less than 90% of it has IR.
  thermal    for isolated warm objects only (--warm classes, box height >= 60 px at 1920 width, no other warm box in the
             window of twice the box size, window inside the image and >= 95% IR-covered): Otsu-threshold the smoothed IR
             window, polarity chosen so the box interior is the hot side, take the largest connected component and compare
             its bounding box with the RGB box (centre offset in pixels at 1920 width, IoU). Only "clear" objects are
             scored: inside-vs-border contrast >= --min-contrast on the window normalised to 0-255 and blob area 0.4-2x the
             box; the rest have too little heat contrast for the blob to be the object. Part of the offset is the blob's
             shape differing from the box, so it is an upper bound on misregistration.
Without --ann only leak and coverage are computed (usable on unlabelled test images; no labels are read or inferred).

Writes report.md, results.json, images.csv, objects.csv and panels/ (per source group: RGB crop | IR crop with the same
box in green and the hot blob in magenta; plus RGB | IR pairs of the most and least leaked images). OUT must not exist.

Example (Linux workstation, repo root):
  python scripts/analysis/audit_ir_alignment.py --images dataset/train/visible --ir dataset/train/infrared_aligned \
    --ann train=work_dir/deimv2_dataset/train/annotations/instances_train.json \
    --ann valid=work_dir/deimv2_dataset/valid/annotations/instances_val.json --out work_dir/audits/ir_alignment_x
"""
import argparse
from collections import defaultdict
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import random
import re
import sys

import cv2
import numpy as np
from PIL import Image

A = None
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


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


def find(folder, stem):
    hits = sorted(Path(folder).glob(stem + '.*'))
    return hits[0] if hits else None


def load(fname):
    rgb = np.asarray(Image.open(Path(A.images)/fname).convert('RGB'))
    with Image.open(find(A.ir, Path(fname).stem)) as im:
        ir = im.convert('L')
        if ir.size != (rgb.shape[1], rgb.shape[0]):
            ir = ir.resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR)
        ir = np.asarray(ir, np.float32)
    return rgb, ir


def leak_score(grey, ir, valid):
    log = lambda x: cv2.Laplacian(cv2.GaussianBlur(x, (0, 0), 1.0), cv2.CV_32F)
    m = valid[::2, ::2]
    a, b = log(grey)[::2, ::2][m], log(ir)[::2, ::2][m]
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 1000 and a.std() > 0 and b.std() > 0 else float('nan')


def hot_blob(ir, box, W, H):
    """Largest hot component in the 2x window around box; None when the window is not usable."""
    x, y, w, h = box
    X0, Y0, X1, Y1 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 1.5 * w), int(y + 1.5 * h)
    if X0 < 0 or Y0 < 0 or X1 > W or Y1 > H:
        return None
    win = ir[Y0:Y1, X0:X1]
    if (win > 0).mean() < 0.95:
        return None
    u8 = cv2.normalize(cv2.GaussianBlur(win, (0, 0), 2.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bx0, by0, bx1, by1 = int(x - X0), int(y - Y0), int(x + w - X0), int(y + h - Y0)
    edge = max(1, bx0 // 2)
    inside, border = float(u8[by0:by1, bx0:bx1].mean()), float(np.concatenate([u8[:, :edge].ravel(), u8[:, -edge:].ravel()]).mean())
    if inside < border:
        u8 = 255 - u8
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    hx, hy, hw, hh = (int(v) for v in st[k, :4])
    ix = max(0, min(bx1, hx + hw) - max(bx0, hx))
    iy = max(0, min(by1, hy + hh) - max(by0, hy))
    return dict(window=(X0, Y0, X1, Y1), box_in_win=(bx0, by0, bx1, by1), blob=(hx, hy, hw, hh),
                dx=(hx + hw / 2) - (bx0 + w / 2), dy=(hy + hh / 2) - (by0 + h / 2),
                iou=ix * iy / (w * h + hw * hh - ix * iy), contrast=abs(inside - border), area_ratio=hw * hh / (w * h))


def one(job):
    fname, boxes = job
    if find(A.ir, Path(fname).stem) is None:
        return dict(file=fname, error='no IR'), []
    rgb, ir = load(fname)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = ir.shape
    sc = W / 1920.0
    covered = ir > 0
    valid = cv2.erode(covered.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    rec = dict(file=fname, source=source(fname), res=f'{W}x{H}', ir_cover=round(float(covered.mean()), 4),
               leak=round(leak_score(grey, ir, valid), 4))
    warm = [b for b in boxes if b['cls'] in A.warm]
    out = []
    for b in boxes:
        x, y, w, h = b['bbox']
        cov = covered[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)]
        row = dict(file=fname, source=rec['source'], res=rec['res'], leak=rec['leak'], ann=b['id'], cls=b['cls'],
                   w=round(w, 1), h=round(h, 1), box_ir_cover=round(float(cov.mean()) if cov.size else 0.0, 3))
        row['outside_ir'] = row['box_ir_cover'] < 0.9
        if b in warm and h >= 60 * sc and w >= 15 * sc:
            X0, Y0, X1, Y1 = x - 0.5 * w, y - 0.5 * h, x + 1.5 * w, y + 1.5 * h
            isolated = not any(o is not b and not (o['bbox'][0] + o['bbox'][2] < X0 or o['bbox'][0] > X1 or
                                                   o['bbox'][1] + o['bbox'][3] < Y0 or o['bbox'][1] > Y1) for o in warm)
            r = hot_blob(ir, b['bbox'], W, H) if isolated else None
            row['thermal'] = 'crowded' if not isolated else ('unusable' if r is None else
                             ('clear' if r['contrast'] >= A.min_contrast and 0.4 <= r['area_ratio'] <= 2.0 else 'low_contrast'))
            if r is not None:
                row.update(dx=round(r['dx'] / sc, 1), dy=round(r['dy'] / sc, 1), iou=round(r['iou'], 3),
                           contrast=round(r['contrast'], 1), area_ratio=round(r['area_ratio'], 2))
        out.append(row)
    return rec, out


def q(v, ps=(25, 50, 75, 90)):
    v = np.asarray(v, float)
    return {f'p{p}': round(float(np.percentile(v, p)), 2) for p in ps} if len(v) else {}


def summarise(recs, rows):
    groups = lambda r: [(r['source'], r['res']), ('ALL', r['res'])]
    img, obj = defaultdict(list), defaultdict(list)
    for r in recs:
        for g in groups(r):
            img[g].append(r)
    for r in rows:
        for g in groups(r):
            obj[g].append(r)
    out = {}
    for key in sorted(img):
        g = img[key]
        lk = [r['leak'] for r in g if r['leak'] == r['leak']]
        e = dict(images=len(g), ir_cover=q([r['ir_cover'] for r in g], (5, 50, 95)),
                 images_with_ir_padding=round(float(np.mean([r['ir_cover'] < 0.999 for r in g])), 3),
                 leak=q(lk, (10, 50, 90)), leak_ge_0_2=round(float(np.mean(np.asarray(lk) >= 0.2)), 3) if lk else None)
        o = obj.get(key, [])
        if o:
            e['boxes'] = len(o)
            e['boxes_outside_ir'] = sum(r['outside_ir'] for r in o)
            th = [r for r in o if 'thermal' in r]
            e['warm_tall'] = len(th)
            e['thermal_status'] = {k: sum(r['thermal'] == k for r in th) for k in ('clear', 'low_contrast', 'crowded', 'unusable')}
            clear = [r for r in th if r['thermal'] == 'clear']
            if clear:
                d = np.hypot([r['dx'] for r in clear], [r['dy'] for r in clear])
                iou = np.asarray([r['iou'] for r in clear])
                e['clear'] = dict(n=len(clear), centre_offset_px=q(d), within_8px=round(float((d <= 8).mean()), 3),
                                  within_16px=round(float((d <= 16).mean()), 3),
                                  dx_median=round(float(np.median([r['dx'] for r in clear])), 1),
                                  dy_median=round(float(np.median([r['dy'] for r in clear])), 1),
                                  iou=q(iou, (25, 50, 75)), iou_ge_0_5=round(float((iou >= 0.5).mean()), 3),
                                  box_h_median=round(float(np.median([r['h'] for r in clear])), 1))
        out[f'{key[0]} {key[1]}'] = e
    return out


def crop_tile(fname, row, sc_h=220):
    rgb, ir = load(fname)
    H, W = ir.shape
    r = hot_blob(ir, (row['bx'], row['by'], row['bw'], row['bh']), W, H)
    X0, Y0, X1, Y1 = r['window']
    bx0, by0, bx1, by1 = r['box_in_win']
    hx, hy, hw, hh = r['blob']
    c1 = np.ascontiguousarray(rgb[Y0:Y1, X0:X1])
    c2 = cv2.cvtColor(cv2.normalize(ir[Y0:Y1, X0:X1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    t = max(1, int(2 * W / 1920))
    for c in (c1, c2):
        cv2.rectangle(c, (bx0, by0), (bx1, by1), (0, 255, 0), t)
    cv2.rectangle(c2, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), t)
    tile = np.hstack([c1, c2])
    tile = cv2.resize(tile, (int(tile.shape[1] * sc_h / tile.shape[0]), sc_h))
    tile = cv2.copyMakeBorder(tile, 18, 2, 2, 2, cv2.BORDER_CONSTANT)
    cv2.putText(tile, f"{Path(fname).stem[:22]} {row['dx']:+.0f},{row['dy']:+.0f} iou{row['iou']:.2f}", (2, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return tile


def sheet(tiles, path, cols=4):
    if not tiles:
        return
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def pair_tile(fname, leak):
    rgb, ir = load(fname)
    ir3 = cv2.cvtColor(ir.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    t = cv2.resize(np.hstack([rgb, ir3]), (960, 270))
    t = cv2.copyMakeBorder(t, 18, 2, 2, 2, cv2.BORDER_CONSTANT)
    cv2.putText(t, f'{Path(fname).stem} leak {leak:.2f}', (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return t


def main():
    global A
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', required=True)
    p.add_argument('--ir', required=True, help='the IR the model reads (e.g. dataset/train/infrared_aligned)')
    p.add_argument('--ann', action='append', default=[], metavar='SPLIT=COCO_JSON', help='omit for leak/coverage only')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warm', default='person,animal')
    p.add_argument('--min-contrast', type=float, default=50.0, help='inside-vs-border contrast on the 0-255 window')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--crops', type=int, default=24, help='crop tiles per source group')
    A = p.parse_args()
    A.warm = set(A.warm.split(','))
    if A.out.exists():
        sys.exit(f'refusing: {A.out} exists')
    boxes = {}
    if A.ann:
        for spec in A.ann:
            _, path = spec.split('=', 1)
            d = json.loads(Path(path).read_text())
            cats = {c['id']: c['name'] for c in d['categories']}
            files = {im['id']: im['file_name'] for im in d['images']}
            for f in files.values():
                boxes.setdefault(f, [])
            for g in d['annotations']:
                if not g.get('iscrowd', 0):
                    boxes[files[g['image_id']]].append(dict(id=g['id'], bbox=g['bbox'], cls=cats[g['category_id']]))
    else:
        boxes = {f.name: [] for f in sorted(Path(A.images).iterdir()) if f.suffix.lower() in IMAGE_EXTS}
    with Pool(A.workers) as pool:
        results = pool.map(one, sorted(boxes.items()), chunksize=8)
    recs = [r for r, _ in results if 'error' not in r]
    errors = [r for r, _ in results if 'error' in r]
    rows = [b for _, bs in results for b in bs]
    res = dict(images=len(recs), errors=errors, labelled=bool(A.ann), boxes=len(rows),
               settings=dict(ir=str(A.ir), warm=sorted(A.warm), min_contrast=A.min_contrast), groups=summarise(recs, rows))
    A.out.mkdir(parents=True)
    (A.out/'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    with (A.out/'images.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'source', 'res', 'ir_cover', 'leak'], extrasaction='ignore')
        w.writeheader()
        w.writerows(recs)
    if rows:
        keys = ['file', 'source', 'res', 'leak', 'ann', 'cls', 'w', 'h', 'box_ir_cover', 'outside_ir', 'thermal', 'dx', 'dy',
                'iou', 'contrast', 'area_ratio']
        with (A.out/'objects.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
    panels = A.out/'panels'
    panels.mkdir()
    rng = random.Random(0)
    group = lambda s: 'leaked_sources' if s in ('seq', 'seq_suppl', 'shuming', 'hehe') else s
    clear = defaultdict(list)
    geo = {(f, b['id']): b['bbox'] for f, bl in boxes.items() for b in bl}
    for r in rows:
        if r.get('thermal') == 'clear':
            r['bx'], r['by'], r['bw'], r['bh'] = geo[(r['file'], r['ann'])]
            clear[group(r['source'])].append(r)
    for g, rs in clear.items():
        sheet([crop_tile(r['file'], r) for r in rng.sample(rs, min(A.crops, len(rs)))], panels/f'clear_thermal_{g}.jpg')
    ranked = sorted((r for r in recs if r['leak'] == r['leak']), key=lambda r: r['leak'])
    if ranked:
        sheet([pair_tile(r['file'], r['leak']) for r in ranked[-4:] + ranked[:4]], panels/'leak_most_and_least.jpg', cols=2)
    lines = ['# IR check: RGB leakage, coverage, thermal alignment', '', '```', json.dumps(res, indent=1, ensure_ascii=False)[:30000],
             '```', '', 'leak: fine-detail correlation of grey RGB and IR at zero shift (high = RGB blended into the IR file).',
             'thermal clear: isolated warm objects with enough heat contrast; offset = hot-blob centre minus RGB box centre at',
             '1920 width, an upper bound on misregistration since the blob shape differs from the box.']
    (A.out/'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(res['groups'], indent=1, ensure_ascii=False))
    print(f'-> {A.out}')


if __name__ == '__main__':
    main()
