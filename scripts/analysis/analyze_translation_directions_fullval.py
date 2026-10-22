#!/usr/bin/env python3
"""Paired single-image interventions; all-query oracle scores are not detection recall."""
import argparse, contextlib, io, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from analyze_light_val_thresholds import sha, dump, ious, nms, evaluate, metrics, scan


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.bundle/'manifest.json').read_text());r=m['runs'][0]
    assert r['status']=='complete' and len(r['exact_forward_matches'])==400 and all(v['exact'] for v in r['exact_forward_matches'])
    assert sha(a.annotations)==m['annotation_sha256']
    assert sha(a.bundle/'intervention_outputs.npz')==r['intervention_outputs_sha256']
    with contextlib.redirect_stdout(io.StringIO()):c=COCO(str(a.annotations))
    light=next(k for k,v in c.cats.items() if v['name']=='light')
    z=np.load(a.bundle/'intervention_outputs.npz');ids=z['image_ids'].tolist();assert len(ids)==200 and set(ids)==set(c.imgs)
    gt=[g for g in c.anns.values() if not g.get('iscrowd',0) and not g.get('ignore',0)]
    pos={v:i for i,v in enumerate(ids)};results={};oracle={};matched={};selection={}
    for arm,info in r['arms'].items():
        f=a.bundle/(arm+'_predictions.json');assert sha(f)==info['prediction_sha256']
        raw=json.loads(f.read_text());assert len(raw)==60000
        by=defaultdict(list)
        for v in raw:by[v['image_id']].append(v)
        assert set(by)==set(ids) and all(len(v)==300 for v in by.values())
        preds=[v for batch in by.values() for v in nms(batch,.6)]
        e=evaluate(c,preds);s=scan(e,light)
        results[arm]=dict(metrics=metrics(e,light),thresholds=[v for v in s if v['threshold'] in [.3,.48] and v['iou']==.5])
        match=set()
        for ev in e.evalImgs:
            if ev is None or ev['aRng']!=e.params.areaRng[0]:continue
            for j,score in enumerate(ev['dtScores']):
                if score>=.48 and not ev['dtIgnore'][0,j] and ev['dtMatches'][0,j]>0:match.add(int(ev['dtMatches'][0,j]))
        matched[arm]=match;rows={}
        for g in gt:
            i=pos[g['image_id']];im=c.imgs[g['image_id']];b=z[arm+'_boxes'][i].astype(float).copy()
            b[:,0]-=b[:,2]/2;b[:,1]-=b[:,3]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']])
            ov=ious(g['bbox'],b);log=z[arm+'_logits'][i,:,g['category_id']].astype(float)
            scores=1/(1+np.exp(-np.clip(log,-80,80)));good=ov>=.5
            q=int(np.argmax(np.where(good,scores,-1))) if good.any() else -1
            rows[g['id']]=dict(geometry=bool(good.any()),score=float(scores[q]) if q>=0 else 0.,max_iou=float(ov.max()),query=q)
        oracle[arm]=rows
        print(arm,results[arm]['metrics'],flush=True)
    pairs=[];groups=defaultdict(list)
    for g in gt:
        im=c.imgs[g['image_id']];x,y,w,h=g['bbox'];mx=8*im['width']/2560;my=8*im['height']/1440
        tags=['all']
        if x>=mx and y>=my and x+w<=im['width']-mx and y+h<=im['height']-my:tags.append('interior')
        if g.get('area',w*h)<1024:tags.append('small')
        if max(w/max(h,1e-8),h/max(w,1e-8))>=3:tags.append('thin_aspect_ge3')
        if g['category_id']==light:tags.append('light')
        row=dict(ann_id=g['id'],image_id=g['image_id'],category_id=g['category_id'],groups=tags,arms={arm:dict(v[g['id']],detected=g['id'] in matched[arm]) for arm,v in oracle.items()})
        pairs.append(row)
        for t in tags:groups[t].append(row)
    summary={}
    for arm in oracle:
        if arm=='BASELINE':continue
        summary[arm]={}
        for name,rr in groups.items():
            counts=dict(gt=len(rr),baseline_detected=0,variant_detected=0,tp_lost=0,tp_recovered=0,baseline_oracle_above48=0,geometry_lost=0,score_cross_below48_geometry_kept=0,score_drop_ge20_geometry_kept=0)
            for row in rr:
                b=row['arms']['BASELINE'];v=row['arms'][arm]
                counts['baseline_detected']+=b['detected'];counts['variant_detected']+=v['detected']
                counts['tp_lost']+=b['detected'] and not v['detected'];counts['tp_recovered']+=v['detected'] and not b['detected']
                eligible=b['geometry'] and b['score']>=.48
                counts['baseline_oracle_above48']+=eligible
                counts['geometry_lost']+=eligible and not v['geometry']
                counts['score_cross_below48_geometry_kept']+=eligible and v['geometry'] and v['score']<.48
                counts['score_drop_ge20_geometry_kept']+=b['geometry'] and v['geometry'] and b['score']-v['score']>=.2
            summary[arm][name]=counts
    rescue={}
    stability={}
    for candidate in ['BASELINE']:
        shifted='SHIFT_X1' if candidate=='BASELINE' else candidate+'_SHIFT_X1'
        stability[candidate]={}
        for group,rows in groups.items():
            lost=sum(v['arms'][candidate]['detected'] and not v['arms'][shifted]['detected'] for v in rows)
            gained=sum(v['arms'][shifted]['detected'] and not v['arms'][candidate]['detected'] for v in rows)
            crosses=sum(v['arms'][candidate]['geometry'] and v['arms'][candidate]['score']>=.48 and v['arms'][shifted]['geometry'] and v['arms'][shifted]['score']<.48 for v in rows)
            stability[candidate][group]=dict(gt=len(rows),original_tp=sum(v['arms'][candidate]['detected'] for v in rows),shifted_tp=sum(v['arms'][shifted]['detected'] for v in rows),lost=lost,recovered=gained,geometry_kept_score_crosses=crosses)
    rescue=stability
    dump(a.out/'results.json',dict(manifest_sha256=sha(a.bundle/'manifest.json'),annotation_sha256=sha(a.annotations),analyzer_sha256=sha(__file__),metrics=results,query_selections=selection,paired_groups=summary,candidate_stability=rescue,contract='All-query GT-class scores are oracle diagnostics, not COCO detections. Interventions use a historical V36 checkpoint, not trained clean baseline v1. Threshold .48 and IoU .5; interior excludes 8 native pixels on each edge.'))
    dump(a.out/'paired_gt.json',pairs)
    lines=['# Single-image stability screening','', 'Historical V36 proxy; clean baseline v1 remains untrained. All arms use independent images, FP32, identical weights and validation inputs, class-wise NMS 0.60.','', '| Arm | mAP % | AP75 % | AP-small % | light AP % | light TP/FP @ .48 |','|---|---:|---:|---:|---:|---|']
    for arm,v in results.items():
        t=next(x for x in v['thresholds'] if x['scope']=='light' and x['threshold']==.48);q=v['metrics']
        lines.append(f"| {arm} | {100*q['mAP']:.3f} | {100*q['AP75']:.3f} | {100*q['AP_small']:.3f} | {100*q['light_AP']:.3f} | {t['tp']}/{t['fp']} |")
    lines+=['','## Interior paired detections at IoU .5, score .48','','| Arm | GT | Baseline TP | Variant TP | TP lost / recovered | Geometry kept but score crosses below .48 |','|---|---:|---:|---:|---|---:|']
    for arm,v in summary.items():
        q=v['interior'];lines.append(f"| {arm} | {q['gt']} | {q['baseline_detected']} | {q['variant_detected']} | {q['tp_lost']}/{q['tp_recovered']} | {q['score_cross_below48_geometry_kept']} |")
    lines+=['','## Paired original / shifted stability','','| Candidate | Interior original TP | Interior shifted TP | Lost/recovered | Geometry kept score crossing |','|---|---:|---:|---|---:|']
    for name,rr in stability.items():
        q=rr['interior'];lines.append(f"| {name} | {q['original_tp']} | {q['shifted_tp']} | {q['lost']}/{q['recovered']} | {q['geometry_kept_score_crosses']} |")
    lines+=['','Oracle crossings inspect all 300 boxes and all 12 logits, whereas actual TP uses postprocessing and one-to-one COCO matching. All arms are unmodified-model independent single-image translations; all candidates are exploratory frozen-weight validation screens, not trained baseline v1 results. Border and subgroup details are in results.json; every GT is retained in paired_gt.json.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
