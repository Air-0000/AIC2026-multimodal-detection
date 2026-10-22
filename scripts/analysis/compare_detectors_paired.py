#!/usr/bin/env python3
"""CPU-only paired comparison of two detectors on the same labelled images.

Point estimates come from pycocotools. Uncertainty comes from a paired image bootstrap: both
detectors are re-scored on the same resampled image multiset by reweighting their per-image COCO
matches, which is checked on every run to reproduce COCOeval's precision table exactly before any
resampling. Also reports GT-count-weighted AP (small classes no longer get 1/12 of the score each)
and per-GT paired detection counts. Predictions follow the official rules by default: class-wise
NMS 0.60, unknown class ids dropped, at most 100 boxes per image by score.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from analyze_light_val_thresholds import sha, dump, nms, evaluate
from analyze_translation_localization import matches, decompose

METRICS = ['mAP', 'AP50', 'AP75', 'wAP', 'wAP75']


def prepare(rows, coco, nms_thr, cap):
    cats = set(coco.getCatIds())
    by = {}
    for r in rows:
        by.setdefault(r['image_id'], []).append(r)
    out, dropped = [], 0
    for iid in sorted(coco.imgs):
        batch = by.get(iid, [])
        if nms_thr > 0 and batch:
            batch = nms(batch, nms_thr)
        keep = [r for r in batch if r['category_id'] in cats]
        dropped += len(batch)-len(keep)
        if cap > 0:
            keep = sorted(keep, key=lambda r: -r['score'])[:cap]
        out.extend(keep)
    assert set(by) <= set(coco.imgs), 'predictions for images outside the annotation file'
    return out, dropped


class Tables:
    """Per-class detections sorted exactly as COCOeval.accumulate does (area all, maxDets 100)."""

    def __init__(self, e):
        p = e.params
        self.e, self.rec = e, p.recThrs
        self.imgs = list(p.imgIds)
        pos = {iid: i for i, iid in enumerate(self.imgs)}
        A, I = len(p.areaRng), len(p.imgIds)
        md = p.maxDets[-1]
        self.classes = []
        for k in range(len(p.catIds)):
            E = [e.evalImgs[k*A*I + i] for i in range(I)]
            E = [x for x in E if x is not None]
            npig = np.zeros(I)
            if not E:
                self.classes.append(None)
                continue
            scores = np.concatenate([x['dtScores'][:md] for x in E])
            order = np.argsort(-scores, kind='mergesort')
            img = np.concatenate([np.full(len(x['dtScores'][:md]), pos[x['image_id']]) for x in E])[order]
            dtm = np.concatenate([x['dtMatches'][:, :md] for x in E], axis=1)[:, order]
            dtig = np.concatenate([x['dtIgnore'][:, :md] for x in E], axis=1)[:, order]
            for x in E:
                npig[pos[x['image_id']]] = np.count_nonzero(x['gtIgnore'] == 0)
            tp = np.logical_and(dtm, ~dtig).astype(float)
            fp = np.logical_and(np.logical_not(dtm), ~dtig).astype(float)
            self.classes.append(dict(img=img, tp=tp, fp=fp, npig=npig))
        self.gt = np.array([0. if c is None else c['npig'].sum() for c in self.classes])

    def precision(self, C):
        """[S, T, R, K] precision for image multiplicities C [S, n_images]; -1 where a class has no GT."""
        S, T, R, K = len(C), len(self.e.params.iouThrs), len(self.rec), len(self.classes)
        out = -np.ones((S, T, R, K))
        for k, c in enumerate(self.classes):
            if c is None:
                continue
            npig = (C*c['npig']).sum(axis=1)
            if not len(c['img']):
                out[npig > 0, :, :, k] = 0.
                continue
            W = C[:, c['img']]
            for t in range(T):
                tps, fps = np.cumsum(W*c['tp'][t], axis=1), np.cumsum(W*c['fp'][t], axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rc = tps/npig[:, None]
                pr = tps/(fps+tps+np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                for s in range(S):
                    if npig[s] == 0:
                        continue
                    idx = np.searchsorted(rc[s], self.rec, side='left')
                    q = np.zeros(R)
                    ok = idx < rc.shape[1]
                    q[ok] = pr[s, idx[ok]]
                    out[s, t, :, k] = q
        return out

    def summarize(self, P, C):
        """Metric dict of arrays [S] from precision [S, T, R, K]."""
        valid = P[:, 0, 0, :] > -1                                # [S, K]; a class is all -1 or all >= 0
        ap = np.where(valid[:, None, :], P.mean(axis=2), np.nan)  # [S, T, K]
        weights = np.stack([(C*c['npig']).sum(axis=1) if c else np.zeros(len(C)) for c in self.classes], axis=1)
        weights = np.where(valid, weights, 0.)
        thr = list(np.round(self.e.params.iouThrs, 2))
        mean = lambda a: np.nanmean(a, axis=-1)
        wmean = lambda a: np.nansum(np.nan_to_num(a)*weights, axis=-1)/weights.sum(axis=-1)
        return dict(mAP=mean(ap.mean(axis=1)), AP50=mean(ap[:, thr.index(.5)]), AP75=mean(ap[:, thr.index(.75)]),
                    wAP=wmean(ap.mean(axis=1)), wAP75=wmean(ap[:, thr.index(.75)]))


def identity_check(tables):
    P = tables.precision(np.ones((1, len(tables.imgs))))[0]
    ref = tables.e.eval['precision'][:, :, :, 0, -1]
    np.testing.assert_allclose(P, ref, rtol=0, atol=1e-12)
    m = tables.summarize(P[None], np.ones((1, len(tables.imgs))))
    assert np.isclose(m['mAP'][0], tables.e.stats[0]) and np.isclose(m['AP75'][0], tables.e.stats[2])


def bootstrap(ta, tb, n, seed, chunk=100):
    rng = np.random.default_rng(seed)
    I = len(ta.imgs)
    res = {m: ([], []) for m in METRICS}
    for start in range(0, n, chunk):
        C = rng.multinomial(I, np.full(I, 1/I), size=min(chunk, n-start)).astype(float)
        ma, mb = ta.summarize(ta.precision(C), C), tb.summarize(tb.precision(C), C)
        for m in METRICS:
            res[m][0].append(ma[m])
            res[m][1].append(mb[m])
    return {m: (np.concatenate(a), np.concatenate(b)) for m, (a, b) in res.items()}


def compare(coco, rows_a, rows_b, args):
    evals, prep = {}, {}
    for name, rows in (('A', rows_a), ('B', rows_b)):
        kept, dropped = prepare(rows, coco, args.nms, args.cap)
        evals[name], prep[name] = evaluate(coco, kept), dict(kept=len(kept), dropped_unknown_class=dropped)
    tables = {name: Tables(e) for name, e in evals.items()}
    for t in tables.values():
        identity_check(t)
    one = np.ones((1, len(coco.imgs)))
    point = {name: {m: float(v[0]) for m, v in t.summarize(t.precision(one), one).items()} for name, t in tables.items()}
    boot = bootstrap(tables['A'], tables['B'], args.boot, args.seed)
    stats = {}
    for m in METRICS:
        a, b = boot[m]
        d = b-a
        stats[m] = dict(A=point['A'][m], B=point['B'][m], delta=point['B'][m]-point['A'][m],
                        ci_A=np.percentile(a, [2.5, 97.5]).tolist(), ci_B=np.percentile(b, [2.5, 97.5]).tolist(),
                        ci_delta=np.percentile(d, [2.5, 97.5]).tolist(), p_B_gt_A=float(np.mean(d > 0)))
    classes = {}
    for k, cat in enumerate(evals['A'].params.catIds):
        name = coco.cats[cat]['name']
        pa, pb = evals['A'].eval['precision'][:, :, k, 0, -1], evals['B'].eval['precision'][:, :, k, 0, -1]
        if not np.any(pa > -1):
            continue
        ap = lambda p: (float(p[p > -1].mean()), float(p[5][p[5] > -1].mean()))
        (a, a75), (b, b75) = ap(pa), ap(pb)
        classes[name] = dict(gt=int(tables['A'].gt[k]), A=a, B=b, A75=a75, B75=b75)
    gts = [g['id'] for g in coco.anns.values() if not g.get('iscrowd', 0) and not g.get('ignore', 0)]
    paired = decompose(gts, matches(evals['A'], args.score_thr), matches(evals['B'], args.score_thr))
    return dict(prepared=prep, point=point, stats=stats, classes=classes, paired_gt=paired)


def render(out, names, args):
    pp = lambda v: f'{100*v:.2f}'
    L = [f'# Paired comparison: {names[0]} (A) vs {names[1]} (B)', '',
         f'{out["n_images"]} labelled images; class-wise NMS {args.nms}, unknown classes dropped, per-image cap {args.cap}; '
         f'paired image bootstrap n={args.boot}, seed {args.seed}. Bootstrap re-scoring reproduced COCOeval exactly on the full set. '
         'wAP weights each class by its GT count instead of 1/12. Exploratory unless the images were held out from all selection.', '',
         '| metric | A | B | B-A (pp) | 95% CI of B-A (pp) | P(B>A) | 95% CI A | 95% CI B |', '|---|---:|---:|---:|---|---:|---|---|']
    for m in METRICS:
        s = out['stats'][m]
        lo, hi = s['ci_delta']
        L.append(f"| {m} | {pp(s['A'])} | {pp(s['B'])} | {100*s['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {s['p_B_gt_A']:.3f} | "
                 f"[{pp(s['ci_A'][0])}, {pp(s['ci_A'][1])}] | [{pp(s['ci_B'][0])}, {pp(s['ci_B'][1])}] |")
    c = out['paired_gt']
    L += ['', f'## Per-GT paired detections (score >= {args.score_thr:g})', '',
          f"IoU .5: A {c['baseline_tp50']}, B {c['shifted_tp50']}, only-A {c['lost50']}, only-B {c['recovered50']} of {c['gt']} GT. "
          f"IoU .75: A {c['baseline_tp75']}, B {c['shifted_tp75']}; A-only-tight {c['loosened']+c['lost']}, B-only-tight {c['tightened']+c['appeared']}.",
          '', '## Per-class AP (pp)', '', '| class | GT | A mAP | B mAP | delta | A AP75 | B AP75 | delta AP75 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, v in sorted(out['classes'].items(), key=lambda kv: kv[1]['gt']):
        L.append(f"| {name} | {v['gt']} | {pp(v['A'])} | {pp(v['B'])} | {100*(v['B']-v['A']):+.2f} | {pp(v['A75'])} | {pp(v['B75'])} | {100*(v['B75']-v['A75']):+.2f} |")
    L += ['', 'Read the CI, not the point delta: if the 95% CI of B-A contains 0, this image set cannot distinguish the two.']
    return '\n'.join(L)+'\n'


def self_test():
    """Random synthetic detections: exact COCOeval identity, bootstrap shape, and a dominance case."""
    rng = np.random.default_rng(0)
    cats = [dict(id=k, name=f'c{k}') for k in range(3)]
    images = [dict(id=i, width=640, height=480, file_name=f'{i}.jpg') for i in range(30)]
    anns, rows_a, rows_b = [], [], []
    for im in images:
        for _ in range(rng.integers(0, 6)):
            x, y, w, h = rng.uniform(0, 500), rng.uniform(0, 380), rng.uniform(8, 120), rng.uniform(8, 90)
            g = dict(id=len(anns)+1, image_id=im['id'], category_id=int(rng.integers(0, 3)), bbox=[x, y, w, h], area=w*h, iscrowd=0)
            anns.append(g)
            jitter = rng.normal(0, 6, 4)
            rows_a.append(dict(image_id=im['id'], category_id=g['category_id'], bbox=[x+jitter[0], y+jitter[1], w, h], score=float(rng.uniform(.2, 1))))
            rows_b.append(dict(rows_a[-1], bbox=[x+jitter[0]/4, y+jitter[1]/4, w, h]))
        for _ in range(rng.integers(0, 4)):
            fp = dict(image_id=im['id'], category_id=int(rng.integers(0, 3)), bbox=[float(v) for v in rng.uniform(0, 300, 4)], score=float(rng.uniform(0, .6)))
            rows_a.append(fp)
            rows_b.append(dict(fp))
        rows_a.append(dict(image_id=im['id'], category_id=99, bbox=[0, 0, 5, 5], score=.99))
    coco = COCO()
    coco.dataset = dict(images=images, annotations=anns, categories=cats)
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    args = argparse.Namespace(nms=0., cap=100, boot=200, seed=1, score_thr=.48)
    out = compare(coco, rows_a, rows_b, args)   # identity_check runs inside
    assert out['prepared']['A']['dropped_unknown_class'] == 30
    s = out['stats']['AP75']
    assert s['B'] > s['A'] and s['p_B_gt_A'] > .95 and s['ci_delta'][0] > 0, s
    assert out['stats']['mAP']['ci_A'][0] <= out['stats']['mAP']['A'] <= out['stats']['mAP']['ci_A'][1]
    out['n_images'] = len(images)
    render(out, ['a', 'b'], args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--a', help='NAME=predictions.json (reference)')
    p.add_argument('--b', help='NAME=predictions.json (candidate)')
    p.add_argument('--nms', type=float, default=.6, help='class-wise NMS IoU; 0 disables')
    p.add_argument('--cap', type=int, default=100, help='max boxes per image by score (official rule); 0 disables')
    p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--score-thr', type=float, default=.48)
    p.add_argument('--out', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        print('SELF_TEST_PASS')
        return
    if not (a.annotations and a.a and a.b and a.out):
        p.error('--annotations, --a, --b and --out are required')
    (na, fa), (nb, fb) = a.a.split('=', 1), a.b.split('=', 1)
    fa, fb = Path(fa), Path(fb)
    a.out.mkdir(parents=True, exist_ok=False)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(a.annotations))
    out = compare(coco, json.loads(fa.read_text()), json.loads(fb.read_text()), a)
    out.update(n_images=len(coco.imgs), names=[na, nb], inputs={na: dict(path=str(fa), sha256=sha(fa)), nb: dict(path=str(fb), sha256=sha(fb))},
               annotation_sha256=sha(a.annotations), analyzer_sha256=sha(__file__),
               settings=dict(nms=a.nms, cap=a.cap, boot=a.boot, seed=a.seed, score_thr=a.score_thr))
    dump(a.out/'results.json', out)
    (a.out/'report.md').write_text(render(out, [na, nb], a))
    for m in METRICS:
        s = out['stats'][m]
        print(f"{m:6s} A {100*s['A']:.2f}  B {100*s['B']:.2f}  B-A {100*s['delta']:+.2f}  CI [{100*s['ci_delta'][0]:+.2f}, {100*s['ci_delta'][1]:+.2f}]  P(B>A) {s['p_B_gt_A']:.3f}")


if __name__ == '__main__':
    main()
