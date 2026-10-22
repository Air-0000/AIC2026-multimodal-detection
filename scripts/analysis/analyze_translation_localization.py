#!/usr/bin/env python3
"""CPU-only: split the +1px AP drop into looser boxes vs. lost detections.

Reads an existing translation bundle (frozen historical V36; BASELINE plus shifted
arms saved as PostProcessor top-300 predictions, already inverse-mapped to original
pixels). No model, GPU, or training; results are not clean baseline v1 results.
"""
import argparse
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from analyze_light_val_thresholds import sha, dump, ious, nms, evaluate

GROUPS = ['all', 'interior', 'small', 'thin_aspect_ge3', 'light']
BINS = [round(.5+.05*i, 2) for i in range(11)]
ERRORS = ['dcx', 'dcy', 'dlogw', 'dlogh']
# Already published in MODEL_RECORDS (12-arm screen and full-val direction test).
RECORDED_AP = {'BASELINE': dict(mAP=.53983, AP75=.59327), 'SHIFT_X1': dict(mAP=.52022, AP75=.54917)}
RECORDED_INTERIOR = {'SHIFT_X1': dict(baseline_tp=1153, shifted_tp=1158, lost=19, recovered=24)}


def key(cut):
    return f'{cut:g}'


def tags(g, im, light):
    x, y, w, h = g['bbox']
    mx, my = 8*im['width']/2560, 8*im['height']/1440
    out = ['all']
    if x >= mx and y >= my and x+w <= im['width']-mx and y+h <= im['height']-my:
        out.append('interior')
    if g.get('area', w*h) < 1024:
        out.append('small')
    if max(w/max(h, 1e-8), h/max(w, 1e-8)) >= 3:
        out.append('thin_aspect_ge3')
    if g['category_id'] == light:
        out.append('light')
    return out


def describe(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return dict(n=0)
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return dict(n=int(len(v)), mean=float(v.mean()), p10=float(q[0]), p25=float(q[1]),
                median=float(q[2]), p75=float(q[3]), p90=float(q[4]))


def index(e, thr):
    return int(np.where(np.isclose(e.params.iouThrs, thr))[0][0])


def ap_table(e):
    """AP per IoU threshold (area all, maxDets 100), overall and per class."""
    pr = e.eval['precision'][:, :, :, 0, -1]
    mean = lambda p: float(p[p > -1].mean()) if np.any(p > -1) else None
    overall = [mean(pr[t]) for t in range(len(e.params.iouThrs))]
    assert np.isclose(overall[index(e, .5)], e.stats[1]) and np.isclose(overall[index(e, .75)], e.stats[2])
    assert np.isclose(np.mean(overall), e.stats[0])
    classes = {}
    for k, cat in enumerate(e.params.catIds):
        classes[e.cocoGt.cats[cat]['name']] = dict(AP50=mean(pr[index(e, .5), :, k]),
                                                    AP75=mean(pr[index(e, .75), :, k]), mAP=mean(pr[:, :, k]))
    return overall, classes


def matches(e, cut):
    """{.5|.75: {gt_id: (dt_id, IoU)}} from COCO one-to-one matching, score >= cut."""
    out = {}
    for thr in (.5, .75):
        j = index(e, thr)
        hit = {}
        for ev in e.evalImgs:
            if ev is None or ev['aRng'] != e.params.areaRng[0]:
                continue
            for k, d in enumerate(ev['dtIds']):
                g = int(ev['dtMatches'][j, k])
                if g <= 0 or ev['dtIgnore'][j, k] or ev['dtScores'][k] < cut:
                    continue
                iou = float(ious(e.cocoGt.anns[g]['bbox'], [e.cocoDt.anns[d]['bbox']])[0])
                assert iou >= thr-1e-6, (g, d, iou)
                hit[g] = (d, iou)
        out[thr] = hit
    return out


def error(g, d, im):
    """Signed box error in network-input pixels (2560x1440) and log size ratios."""
    gx, gy, gw, gh = g
    dx, dy, dw, dh = d
    return dict(dcx=(dx+dw/2-gx-gw/2)*2560/im['width'], dcy=(dy+dh/2-gy-gh/2)*1440/im['height'],
                dlogw=float(np.log(max(dw, 1e-6)/max(gw, 1e-6))), dlogh=float(np.log(max(dh, 1e-6)/max(gh, 1e-6))))


def decompose(ids, b, s):
    """Per-GT TP@.5/.75 transitions from baseline matches b to shifted matches s."""
    c = dict(gt=len(ids), baseline_tp50=0, shifted_tp50=0, lost50=0, recovered50=0, baseline_tp75=0,
             shifted_tp75=0, kept75=0, loosened=0, lost=0, tightened=0, appeared=0)
    for gid in ids:
        b50, b75, s50, s75 = gid in b[.5], gid in b[.75], gid in s[.5], gid in s[.75]
        c['baseline_tp50'] += b50
        c['shifted_tp50'] += s50
        c['lost50'] += b50 and not s50
        c['recovered50'] += s50 and not b50
        c['baseline_tp75'] += b75
        c['shifted_tp75'] += s75
        c['kept75'] += b75 and s75
        c['loosened'] += b75 and not s75 and s50
        c['lost'] += b75 and not s75 and not s50
        c['tightened'] += s75 and not b75 and b50
        c['appeared'] += s75 and not b75 and not b50
    assert c['shifted_tp75']-c['baseline_tp75'] == c['tightened']+c['appeared']-c['loosened']-c['lost']
    return c


def contribution(ap, arms, e):
    """Class AP change / number of classes with GT; sums exactly to the overall change."""
    base = ap['BASELINE'][1]
    k = sum(v['mAP'] is not None for v in base.values())
    out = {}
    for arm in arms:
        rows = {name: {m: (ap[arm][1][name][m]-r[m])/k for m in ('AP50', 'AP75', 'mAP')}
                for name, r in base.items() if r['mAP'] is not None}
        for m, thr in (('AP50', .5), ('AP75', .75)):
            j = index(e, thr)
            assert np.isclose(sum(x[m] for x in rows.values()), ap[arm][0][j]-ap['BASELINE'][0][j])
        assert np.isclose(sum(x['mAP'] for x in rows.values()), np.mean(ap[arm][0])-np.mean(ap['BASELINE'][0]))
        out[arm] = rows
    return out


def analyze(coco, evals, arms, cuts, light):
    gts = [g for g in coco.anns.values() if not g.get('iscrowd', 0) and not g.get('ignore', 0)]
    groups = {g['id']: tags(g, coco.imgs[g['image_id']], light) for g in gts}
    members = {name: [gid for gid, t in groups.items() if name in t] for name in GROUPS}
    classes = {coco.cats[k]['name']: [g['id'] for g in gts if g['category_id'] == k] for k in sorted(coco.cats)}
    classes = {name: ids for name, ids in classes.items() if ids}
    ap = {arm: ap_table(e) for arm, e in evals.items()}
    out = dict(iou_thresholds=[round(float(t), 2) for t in evals['BASELINE'].params.iouThrs],
               ap_by_iou={arm: v[0] for arm, v in ap.items()}, ap_by_class={arm: v[1] for arm, v in ap.items()},
               ap_contribution=contribution(ap, arms, evals['BASELINE']),
               n_images=len(coco.imgs), group_sizes={name: len(v) for name, v in members.items()}, cuts={})

    def err(e, gid, did):
        g = e.cocoGt.anns[gid]
        return error(g['bbox'], e.cocoDt.anns[did]['bbox'], coco.imgs[g['image_id']])

    for cut in cuts:
        m = {arm: matches(e, cut) for arm, e in evals.items()}
        row = dict(tp_iou={}, histogram={}, signed_error={}, paired={}, decomposition={}, per_class={})
        for arm, e in evals.items():
            row['tp_iou'][arm], row['histogram'][arm], row['signed_error'][arm] = {}, {}, {}
            for name, ids in members.items():
                hit = [(gid, *m[arm][.5][gid]) for gid in ids if gid in m[arm][.5]]
                v = np.asarray([x[2] for x in hit], dtype=float)
                row['tp_iou'][arm][name] = dict(describe(v), ge75=float(np.mean(v >= .75)) if len(v) else None,
                                                ge90=float(np.mean(v >= .9)) if len(v) else None)
                row['histogram'][arm][name] = dict(bins=BINS, counts=np.histogram(v, bins=BINS)[0].tolist())
                errs = [err(e, gid, did) for gid, did, _ in hit]
                row['signed_error'][arm][name] = {k: describe([x[k] for x in errs]) for k in ERRORS}
        b = m['BASELINE']
        for arm in arms:
            s = m[arm]
            row['paired'][arm], row['decomposition'][arm] = {}, {}
            for name, ids in members.items():
                both = [gid for gid in ids if gid in b[.5] and gid in s[.5]]
                bi = np.asarray([b[.5][gid][1] for gid in both], dtype=float)
                si = np.asarray([s[.5][gid][1] for gid in both], dtype=float)
                moved = [(err(evals[arm], gid, s[.5][gid][0]), err(evals['BASELINE'], gid, b[.5][gid][0])) for gid in both]
                row['paired'][arm][name] = dict(
                    delta_iou=describe(si-bi), delta_le_minus05=int(np.sum(si-bi <= -.05)),
                    crossed_below75=int(np.sum((bi >= .75) & (si < .75))), crossed_above75=int(np.sum((bi < .75) & (si >= .75))),
                    delta_error={k: describe([x[k]-y[k] for x, y in moved]) for k in ERRORS})
                row['decomposition'][arm][name] = decompose(ids, b, s)
            row['per_class'][arm] = {name: decompose(ids, b, s) for name, ids in classes.items()}
        out['cuts'][key(cut)] = row
    return out


def reproduction(out, evals, cut):
    rows = []
    for arm, ref in RECORDED_AP.items():
        if arm in evals:
            got = dict(mAP=float(evals[arm].stats[0]), AP75=float(evals[arm].stats[2]))
            rows.append(dict(check=arm+' mAP/AP75', recorded=ref, got=got,
                             ok=all(abs(got[k]-v) <= 6e-6 for k, v in ref.items())))
    if np.isclose(cut, .48):
        for arm, ref in RECORDED_INTERIOR.items():
            d = out['cuts'][key(cut)]['decomposition'].get(arm)
            if d:
                q = d['interior']
                got = dict(baseline_tp=q['baseline_tp50'], shifted_tp=q['shifted_tp50'], lost=q['lost50'], recovered=q['recovered50'])
                rows.append(dict(check=arm+' interior TP@.5 score>=.48', recorded=ref, got=got, ok=got == ref))
    return rows


def render(out, arms, cuts, checks):
    pct = lambda v: '-' if v is None else f'{100*v:.3f}'
    thr = out['iou_thresholds']
    base = out['ap_by_iou']['BASELINE']
    L = ['# +1px translation: AP vs IoU threshold and matched-TP IoU', '',
         'Frozen historical V36 from an existing translation bundle; not clean baseline v1 results. '
         f'{out["n_images"]} validation images, class-wise NMS 0.60, standard COCO (maxDets 100). TP = COCO one-to-one match at IoU .50 with score >= cut. '
         'Exploratory: these validation images also guided earlier interventions.', '',
         '## AP by IoU threshold (%)', '',
         '| arm | ' + ' | '.join(f'{t:.2f}' for t in thr) + ' | mAP |', '|---|' + '---:|'*(len(thr)+1)]
    for arm in ['BASELINE']+arms:
        v = out['ap_by_iou'][arm]
        L.append(f'| {arm} | ' + ' | '.join(pct(x) for x in v) + f' | {pct(float(np.mean(v)))} |')
    for arm in arms:
        v = out['ap_by_iou'][arm]
        L.append(f'| delta {arm} (pp) | ' + ' | '.join(f'{100*(x-y):+.3f}' for x, y in zip(v, base))
                 + f' | {100*(np.mean(v)-np.mean(base)):+.3f} |')
    cut = cuts[0]
    L += ['', f'## Per-class contribution (sorted by AP75 contribution; TP counts at score >= {cut:g})', '',
          'contrib = class AP change / number of classes with GT (COCO averages classes equally), so contributions sum exactly '
          'to the overall change; dAP and contrib in pp. cum columns accumulate down the table.', '',
          '| arm | class | GT | base AP75 | dAP50 | dAP75 | dmAP | AP75 contrib | mAP contrib | cum AP75 % | cum GT % | '
          'TP@.75 base->shifted | loosened | lost | tightened | appeared | TP@.5 lost/recovered |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|']
    for arm in arms:
        contrib = out['ap_contribution'][arm]
        per = out['cuts'][key(cut)]['per_class'][arm]
        total = sum(x['AP75'] for x in contrib.values())
        ngt = sum(c['gt'] for c in per.values())
        run_ap = run_gt = 0.
        for name in sorted(contrib, key=lambda n: contrib[n]['AP75']):
            x, c = contrib[name], per[name]
            r, v = out['ap_by_class']['BASELINE'][name], out['ap_by_class'][arm][name]
            run_ap += x['AP75']
            run_gt += c['gt']
            share = f'{100*run_ap/total:.1f}' if abs(total) > 1e-12 else '-'
            L.append(f"| {arm} | {name} | {c['gt']} | {pct(r['AP75'])} | {100*(v['AP50']-r['AP50']):+.3f} | "
                     f"{100*(v['AP75']-r['AP75']):+.3f} | {100*(v['mAP']-r['mAP']):+.3f} | {100*x['AP75']:+.3f} | "
                     f"{100*x['mAP']:+.3f} | {share} | {100*run_gt/ngt:.1f} | {c['baseline_tp75']}->{c['shifted_tp75']} | "
                     f"{c['loosened']} | {c['lost']} | {c['tightened']} | {c['appeared']} | {c['lost50']}/{c['recovered50']} |")
    L += ['', '## TP@.75 change decomposition', '',
          'loosened = TP@.75 in baseline, still TP@.5 but not TP@.75 after shift; lost = not even TP@.5 after shift. net = tightened + appeared - loosened - lost.', '',
          '| score >= | arm | group | GT | base TP@.75 | shifted TP@.75 | net | kept | loosened | lost | tightened | appeared | TP@.5 lost/recovered |',
          '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for cut in cuts:
        for arm in arms:
            for name, c in out['cuts'][key(cut)]['decomposition'][arm].items():
                if not c['gt']:
                    continue
                L.append(f"| {cut:g} | {arm} | {name} | {c['gt']} | {c['baseline_tp75']} | {c['shifted_tp75']} | "
                         f"{c['shifted_tp75']-c['baseline_tp75']:+d} | {c['kept75']} | {c['loosened']} | {c['lost']} | "
                         f"{c['tightened']} | {c['appeared']} | {c['lost50']}/{c['recovered50']} |")
    L += ['', '## Matched-TP IoU distribution', '',
          '| score >= | arm | group | TP | mean | p25 | median | p75 | >=.75 % | >=.90 % |', '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cut in cuts:
        for arm in ['BASELINE']+arms:
            for name, d in out['cuts'][key(cut)]['tp_iou'][arm].items():
                if not d['n']:
                    continue
                L.append(f"| {cut:g} | {arm} | {name} | {d['n']} | {d['mean']:.4f} | {d['p25']:.4f} | {d['median']:.4f} | "
                         f"{d['p75']:.4f} | {pct(d['ge75'])} | {pct(d['ge90'])} |")
    L += ['', '## Paired GTs (TP@.5 in both): IoU change, shifted minus baseline', '',
          '| score >= | arm | group | paired | mean dIoU | p10 | median | p90 | dIoU <= -.05 | crossed below .75 | crossed above .75 |',
          '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cut in cuts:
        for arm in arms:
            for name, p in out['cuts'][key(cut)]['paired'][arm].items():
                d = p['delta_iou']
                if not d['n']:
                    continue
                L.append(f"| {cut:g} | {arm} | {name} | {d['n']} | {d['mean']:+.4f} | {d['p10']:+.4f} | {d['median']:+.4f} | "
                         f"{d['p90']:+.4f} | {p['delta_le_minus05']} | {p['crossed_below75']} | {p['crossed_above75']} |")
    L += ['', '## Paired signed box error change, shifted minus baseline', '',
          'cx/cy in network-input pixels (2560x1440); w/h as log ratio. A nonzero mean dcx would indicate a residual inverse-mapping offset rather than model instability.', '',
          '| score >= | arm | group | paired | dcx mean | dcx median | dcy mean | dcy median | dlog w median | dlog h median |',
          '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cut in cuts:
        for arm in arms:
            for name, p in out['cuts'][key(cut)]['paired'][arm].items():
                d = p['delta_error']
                if not d['dcx']['n']:
                    continue
                L.append(f"| {cut:g} | {arm} | {name} | {d['dcx']['n']} | {d['dcx']['mean']:+.3f} | {d['dcx']['median']:+.3f} | "
                         f"{d['dcy']['mean']:+.3f} | {d['dcy']['median']:+.3f} | {d['dlogw']['median']:+.4f} | {d['dlogh']['median']:+.4f} |")
    cut = cuts[0]
    L += ['', f'## Matched-TP IoU histogram (group all, score >= {cut:g})', '',
          '| IoU bin | ' + ' | '.join(['BASELINE']+arms) + ' |', '|---|' + '---:|'*(len(arms)+1)]
    for i in range(len(BINS)-1):
        counts = [out['cuts'][key(cut)]['histogram'][arm]['all']['counts'][i] for arm in ['BASELINE']+arms]
        L.append(f'| {BINS[i]:.2f}-{BINS[i+1]:.2f} | ' + ' | '.join(str(x) for x in counts) + ' |')
    L += ['', '## Reproduction checks against MODEL_RECORDS', '', '| check | recorded | got | ok |', '|---|---|---|---|']
    for r in checks:
        L.append(f"| {r['check']} | {json.dumps(r['recorded'])} | {json.dumps(r['got'])} | {'PASS' if r['ok'] else 'FAIL'} |")
    if not checks:
        L.append('| (no recorded reference for these arms) | | | |')
    L += ['', 'Reading guide: a delta that grows with the IoU threshold while delta AP50 stays near zero points to localization; '
          '"loosened" counts GTs still found at IoU .5 but no longer at .75. Fixed-cut counts approximate, but do not equal, '
          "AP's score-ranked loss. Full per-group values are in results.json."]
    return '\n'.join(L)+'\n'


def self_test():
    """Synthetic fixture whose categories and IoUs are known in closed form."""
    cats = [dict(id=1, name='car'), dict(id=2, name='light')]
    gt = [dict(id=i+1, image_id=1, category_id=c, bbox=[100.+300*i, 100., 100., 100.], area=1e4, iscrowd=0)
          for i, c in enumerate([1, 1, 1, 2, 1, 1])]
    coco = COCO()
    coco.dataset = dict(images=[dict(id=1, width=2560, height=1440, file_name='x.jpg')], annotations=gt, categories=cats)
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    # Per GT: (x offset, score) for BASELINE and SHIFT; None = no box. Same-size boxes: IoU = (100-d)/(100+d).
    plan = [((5, .9), (6, .9)),    # kept at .75
            ((10, .9), (30, .9)),  # loosened .818 -> .538
            ((5, .9), (5, .2)),    # below cut .48 after shift; kept at cut 0
            ((30, .9), (10, .9)),  # tightened (light)
            (None, (10, .9)),      # appeared
            ((10, .9), None)]      # lost, no box
    preds = {'BASELINE': [], 'SHIFT': []}
    for g, pair in zip(gt, plan):
        for arm, p in zip(preds, pair):
            if p:
                x, y, w, h = g['bbox']
                preds[arm].append(dict(image_id=1, category_id=g['category_id'], bbox=[x+p[0], y, w, h], score=p[1]))
    evals = {arm: evaluate(coco, v) for arm, v in preds.items()}
    out = analyze(coco, evals, ['SHIFT'], [.48, 0.], 2)
    iou = lambda d: (100-d)/(100+d)
    base = dict(gt=6, baseline_tp50=5, baseline_tp75=4)
    assert out['cuts']['0.48']['decomposition']['SHIFT']['all'] == dict(
        base, shifted_tp50=4, lost50=2, recovered50=1, shifted_tp75=3, kept75=1, loosened=1, lost=2, tightened=1, appeared=1)
    assert out['cuts']['0']['decomposition']['SHIFT']['all'] == dict(
        base, shifted_tp50=5, lost50=1, recovered50=1, shifted_tp75=4, kept75=2, loosened=1, lost=1, tightened=1, appeared=1)
    assert out['cuts']['0.48']['decomposition']['SHIFT']['light'] == dict(
        gt=1, baseline_tp50=1, shifted_tp50=1, lost50=0, recovered50=0, baseline_tp75=0, shifted_tp75=1,
        kept75=0, loosened=0, lost=0, tightened=1, appeared=0)
    p = out['cuts']['0.48']['paired']['SHIFT']['all']
    delta = sorted([iou(6)-iou(5), iou(30)-iou(10), iou(10)-iou(30)])
    assert p['delta_iou']['n'] == 3 and np.isclose(p['delta_iou']['median'], delta[1])
    assert (p['delta_le_minus05'], p['crossed_below75'], p['crossed_above75']) == (1, 1, 1)
    assert np.isclose(p['delta_error']['dcx']['median'], 1.) and np.isclose(p['delta_error']['dcy']['mean'], 0.)
    t = out['cuts']['0.48']['tp_iou']['BASELINE']['all']
    assert t['n'] == 5 and np.isclose(t['median'], iou(10)) and np.isclose(t['ge75'], .8)
    assert sum(out['cuts']['0.48']['histogram']['BASELINE']['all']['counts']) == 5
    per = out['cuts']['0.48']['per_class']['SHIFT']
    assert per['car'] == dict(gt=5, baseline_tp50=4, shifted_tp50=3, lost50=2, recovered50=1, baseline_tp75=4,
                              shifted_tp75=2, kept75=1, loosened=1, lost=2, tightened=0, appeared=1)
    assert per['light'] == out['cuts']['0.48']['decomposition']['SHIFT']['light']
    contrib = out['ap_contribution']['SHIFT']
    assert set(contrib) == {'car', 'light'}
    assert np.isclose(sum(x['AP75'] for x in contrib.values()), out['ap_by_iou']['SHIFT'][5]-out['ap_by_iou']['BASELINE'][5])
    assert np.isclose(out['cuts']['0.48']['signed_error']['BASELINE']['interior']['dcx']['mean'], (5+10+5+30+10)/5)
    render(out, ['SHIFT'], [.48, 0.], [])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, help='audit dir with manifest.json and <ARM>_predictions.json (or its raw/ subdir)')
    p.add_argument('--annotations', type=Path, help='defaults to the manifest annotation_path; SHA is always checked')
    p.add_argument('--arms', nargs='+', default=['SHIFT_X1'])
    p.add_argument('--score-thr', type=float, default=.48)
    p.add_argument('--out', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        print('SELF_TEST_PASS')
        return
    if not (a.bundle and a.out):
        p.error('--bundle and --out are required')
    bundle = a.bundle if (a.bundle/'manifest.json').is_file() else a.bundle/'raw'
    m = json.loads((bundle/'manifest.json').read_text())
    r = m['runs'][0]
    assert r['status'] == 'complete'
    ann = a.annotations or Path(m['annotation_path'])
    assert sha(ann) == m['annotation_sha256']
    missing = [arm for arm in ['BASELINE']+a.arms if arm not in r['arms']]
    assert not missing, f'{missing} not in bundle arms {sorted(r["arms"])}'
    a.out.mkdir(parents=True, exist_ok=False)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann))
    light = next(k for k, v in coco.cats.items() if v['name'] == 'light')
    evals = {}
    for arm in ['BASELINE']+a.arms:
        f = bundle/(arm+'_predictions.json')
        assert sha(f) == r['arms'][arm]['prediction_sha256'], arm
        by = defaultdict(list)
        for v in json.loads(f.read_text()):
            by[v['image_id']].append(v)
        assert set(by) == set(coco.imgs) and all(len(v) == 300 for v in by.values()), arm
        evals[arm] = evaluate(coco, [v for batch in by.values() for v in nms(batch, .6)])
        print(arm, 'mAP %.5f AP50 %.5f AP75 %.5f' % tuple(evals[arm].stats[:3]), flush=True)
    cuts = sorted({a.score_thr, 0.}, reverse=True)
    out = analyze(coco, evals, a.arms, cuts, light)
    checks = reproduction(out, evals, a.score_thr)
    for c in checks:
        print(('PASS ' if c['ok'] else 'WARN reproduction mismatch: ')+c['check'], c['got'], flush=True)
    out.update(bundle=str(bundle.resolve()), bundle_manifest_sha256=sha(bundle/'manifest.json'), annotation_sha256=sha(ann),
               analyzer_sha256=sha(__file__), arms=a.arms, score_cuts=cuts, reproduction_checks=checks,
               contract='Frozen historical V36 translation bundle, not clean baseline v1. Exploratory validation analysis; '
                        'TP = COCO one-to-one match at IoU .5/.75 with score >= cut; NMS 0.60 as in earlier screens.')
    dump(a.out/'results.json', out)
    (a.out/'report.md').write_text(render(out, a.arms, cuts, checks))


if __name__ == '__main__':
    main()
