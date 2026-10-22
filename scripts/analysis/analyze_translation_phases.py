#!/usr/bin/env python3
"""Score counterfactual activation patches on the prespecified 19-image cohort."""
import argparse,contextlib,io,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from analyze_light_val_thresholds import sha,dump,nms,evaluate,ious


def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--paired-gt',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--iou-audit',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 m=json.loads((a.bundle/'manifest.json').read_text());r=m['runs'][0];assert r['status']=='complete' and len(r['exact_forward_matches'])==38 and all(v['exact'] for v in r['exact_forward_matches']);assert sha(a.annotations)==m['annotation_sha256'];assert sha(a.paired_gt)==m['intervention_contract']['paired_gt_sha256'];assert sha(a.bundle/'intervention_outputs.npz')==r['intervention_outputs_sha256']
 original=json.loads(a.annotations.read_text());ids=set(r['evaluated_image_ids']);assert len(ids)==19
 with contextlib.redirect_stdout(io.StringIO()):
  c=COCO();c.dataset=dict(images=[v for v in original['images'] if v['id'] in ids],annotations=[v for v in original['annotations'] if v['image_id'] in ids],categories=original['categories'],info={});c.createIndex()
 gt=[v for v in c.anns.values() if not v.get('iscrowd',0) and not v.get('ignore',0)];old={v['ann_id']:v for v in json.loads(a.paired_gt.read_text())};lost=set(m['intervention_contract']['cohort_lost_gt_ids']);stable={v['id'] for v in gt if 'interior' in old[v['id']]['groups'] and old[v['id']]['arms']['BASELINE']['detected'] and old[v['id']]['arms']['SHIFT_X1']['detected']};assert len(lost)==19 and len(stable)==225
 z=np.load(a.bundle/'intervention_outputs.npz');pos={int(v):i for i,v in enumerate(z['image_ids'])};results={};matched={};details={}
 for arm,info in r['arms'].items():
  f=a.bundle/(arm+'_predictions.json');assert sha(f)==info['prediction_sha256'];raw=json.loads(f.read_text());assert len(raw)==5700
  by=defaultdict(list)
  for v in raw:by[v['image_id']].append(v)
  assert set(by)==ids and all(len(v)==300 for v in by.values())
  preds=[v for pp in by.values() for v in nms(pp,.6)];e=evaluate(c,preds);tp=set();fp=0
  for ev in e.evalImgs:
   if ev is None or ev['aRng']!=e.params.areaRng[0]:continue
   for j,score in enumerate(ev['dtScores']):
    if score<.48 or ev['dtIgnore'][0,j]:continue
    if ev['dtMatches'][0,j]>0:tp.add(int(ev['dtMatches'][0,j]))
    else:fp+=1
  matched[arm]=tp;details[arm]={}
  for g in gt:
   i=pos[g['image_id']];im=c.imgs[g['image_id']];b=z[arm+'_boxes'][i].astype(float).copy();b[:,:2]-=b[:,2:]/2;b*=np.array([im['width'],im['height'],im['width'],im['height']]);ov=ious(g['bbox'],b);ss=1/(1+np.exp(-np.clip(z[arm+'_logits'][i,:,g['category_id']].astype(float),-80,80)));good=ov>=.5
   details[arm][str(g['id'])]=dict(detected=g['id'] in tp,oracle_score=float(np.max(np.where(good,ss,0))),max_iou=float(ov.max()))
  results[arm]=dict(tp=len(tp),fp=fp,lost19_rescued=len(tp&lost),stable225_lost=len(stable-tp),lost19_rescued_ids=sorted(tp&lost),stable225_lost_ids=sorted(stable-tp))
 audit=json.loads(a.iou_audit.read_text());strong=set(audit['SHIFT_X1']['low_gt_score_even_at_iou30']);near=set(audit['SHIFT_X1']['high_score_near_miss_iou45_to50']);assert len(strong)==13 and len(near)==4
 for arm,d in results.items():
  ref=matched['SHIFT_X1'];v=matched[arm];d.update(strong13_rescued=len(v&strong),near4_rescued=len(v&near),strong13_ids=sorted(v&strong));d.update(tp_gained_vs_locked=len(v-ref),tp_lost_vs_locked=len(ref-v),lost19_rescue_new_vs_locked=sorted((v-ref)&lost),lost19_rescue_removed_vs_locked=sorted((ref-v)&lost))
 dump(a.out/'summary.json',dict(scope='Unmodified frozen model translation-phase sweep on 19 symptom-selected images; not deployable, not full validation mAP, no training.',gt=len(gt),lost_cohort=sorted(lost),stable_controls=sorted(stable),results=results,manifest_sha256=sha(a.bundle/'manifest.json'),script_sha256=sha(__file__)))
 dump(a.out/'gt_details.json',details)
 lines=['# Decoder activation intervention results','',f'19 symptom-selected validation images / {len(gt)} noncrowd GT. Actual class-wise NMS .60 / IoU .50 / threshold .48 COCO matching; not full-set AP. Each forward independently uses its own translated image; no hooks alter the model.','', '| Arm | TP | FP | Original 19 losses rescued | 225 stable controls lost | TP gained/lost vs +1x |','|---|---:|---:|---:|---:|---|']
 for arm,v in results.items():lines.append(f"| {arm} | {v['tp']} | {v['fp']} | {v['lost19_rescued']} | {v['stable225_lost']} | {v['tp_gained_vs_locked']}/{v['tp_lost_vs_locked']} |")
 lines+=['',m['intervention_contract']['phase_contract'],'','## Strong low-score13 vs near-IoU-boundary4','','| Arm | Strong13 rescued | Near4 rescued |','|---|---:|---:|']
 for arm,v in results.items():lines.append(f"| {arm} | {v['strong13_rescued']} | {v['near4_rescued']} |")
 (a.out/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(results,ensure_ascii=False))

if __name__=='__main__':main()
