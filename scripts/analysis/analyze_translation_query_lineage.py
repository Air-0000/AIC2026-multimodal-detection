#!/usr/bin/env python3
"""Separate encoder selection, query lineage, class switching and localization."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'work_dir/audits/layer_translation_trace_20261021/raw';out=ROOT/'outputs/reports/translation_query_lineage_20261021';out.mkdir(exist_ok=False)
z=np.load(base/'layer_queries.npz');m=json.loads((base/'manifest.json').read_text());assert m['runs'][0]['status']=='complete'
for name,key in [('layer_queries.npz','layer_queries_sha256'),('layer_gt_trace.json','layer_gt_trace_sha256')]:assert hashlib.sha256((base/name).read_bytes()).hexdigest()==m['runs'][0][key]
rows=json.loads((base/'layer_gt_trace.json').read_text());old={v['ann_id']:v for v in json.loads((ROOT/'outputs/reports/single_image_stability_20261021/analysis/paired_gt.json').read_text())};ann=json.loads((ROOT/'work_dir/deimv2_dataset/valid/annotations/instances_val.json').read_text());gt={v['id']:v for v in ann['annotations']};ims={v['id']:v for v in ann['images']}
def sigmoid(v):return 1/(1+np.exp(-np.clip(v.astype(float),-80,80)))
def iou(g,b):
 im=ims[g['image_id']];bb=b.astype(float).copy();bb[:,:2]-=bb[:,2:]/2;bb*=np.array([im['width'],im['height'],im['width'],im['height']]);x,y,w,h=g['bbox'];inter=np.maximum(0,np.minimum(x+w,bb[:,0]+bb[:,2])-np.maximum(x,bb[:,0]))*np.maximum(0,np.minimum(y+h,bb[:,1]+bb[:,3])-np.maximum(y,bb[:,1]));return inter/np.maximum(w*h+bb[:,2]*bb[:,3]-inter,1e-12)
records=[]
for row in rows:
 aid=row['ann_id'];im=row['image_id'];cat=row['category_id'];g=gt[aid];record=dict(ann_id=aid,image_id=im,category_id=cat,cohort='lost' if row['lost_cohort'] else 'stable' if 'interior' in old[aid]['groups'] and old[aid]['arms']['BASELINE']['detected'] and old[aid]['arms']['SHIFT_X1']['detected'] else 'other',arms={})
 bq=row['arms']['BASELINE']['endpoint_query'];bt=int(z[f'{im}_BASELINE_ids'][bq]);base_logits=z[f'{im}_BASELINE_logits'];bvec=base_logits[-1,bq];baseraw=z[f'{im}_BASELINE_raw'][-1,bq]
 for arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
  prefix=f'{im}_{arm}_';ids=z[prefix+'ids'];ll=z[prefix+'logits'];raw=z[prefix+'raw'];boxes=z[prefix+'boxes'];enc=z[prefix+'enc_logits'];same=np.flatnonzero(ids==bt);q=int(same[0]) if len(same) else None;layers=[]
  for j in range(6):
   ov=iou(g,boxes[j]);scores=sigmoid(ll[j,:,cat]);valid=ov>=.5;best=int(np.argmax(np.where(valid,scores,-1))) if valid.any() else int(ov.argmax());v=ll[j,best];rr=raw[j,best]
   d=dict(layer=j+1,best_query=best,best_token=int(ids[best]),valid_queries=int(valid.sum()),gt_score=float(scores[best]) if valid.any() else 0.,best_iou=float(ov[best]),argmax_class=int(v.argmax()),max_class_score=float(sigmoid(v).max()),gt_logit=float(v[cat]),max_wrong_logit=float(np.max(np.delete(v,cat))),common_logit_shift=float(np.median(v-bvec)),class_specific_shift=float((v-bvec)[cat]-np.median(v-bvec)),encoder_gt_score=float(sigmoid(enc[best])[cat]))
   if q is not None:
    vec=ll[j,q];d['same_token']=dict(query=q,iou=float(ov[q]),gt_score=float(scores[q]),argmax_class=int(vec.argmax()),gt_logit=float(vec[cat]),max_class_score=float(sigmoid(vec).max()),encoder_gt_score=float(sigmoid(enc[q])[cat]))
   layers.append(d)
  record['arms'][arm]=dict(baseline_token_present=q is not None,layers=layers)
 records.append(record)
summary={}
for cohort in ['lost','stable']:
 rr=[v for v in records if v['cohort']==cohort];summary[cohort]={}
 for arm in ['SHIFT_X1','SHIFT_X1_LOCKED']:
  count=dict(n=len(rr),baseline_token_missing=0,same_token_final_iou_ge50=0,same_token_final_class_changed=0,best_valid_final_class_changed=0,all_valid_max_class_score_below48=0,encoder_same_token_gt_score_drop_ge20=0,encoder_same_token_gt_score_drop_ge05=0)
  enc_deltas=[];native_deltas=[];class_specific=[];common=[];same_losses=[]
  for r in rr:
   a=r['arms'][arm];b=r['arms']['BASELINE'];v=a['layers'][-1];bb=b['layers'][-1];cat=r['category_id'];count['baseline_token_missing']+=not a['baseline_token_present'];count['best_valid_final_class_changed']+=v['argmax_class']!=cat
   prefix=f"{r['image_id']}_{arm}_";ov=iou(gt[r['ann_id']],z[prefix+'boxes'][-1]);valid=ov>=.5
   count['all_valid_max_class_score_below48']+=bool(valid.any()) and bool((sigmoid(z[prefix+'logits'][-1][valid]).max()<.48))
   class_specific.append(v['class_specific_shift']);common.append(v['common_logit_shift'])
   if 'same_token' in v:
    s=v['same_token'];bs=bb['same_token'];count['same_token_final_iou_ge50']+=s['iou']>=.5;count['same_token_final_class_changed']+=s['argmax_class']!=cat
    delta=bs['encoder_gt_score']-s['encoder_gt_score'];enc_deltas.append(delta);count['encoder_same_token_gt_score_drop_ge20']+=delta>=.2;count['encoder_same_token_gt_score_drop_ge05']+=delta>=.05
    native_deltas.append(bs['gt_score']-s['gt_score']);same_losses.append(dict(ann_id=r['ann_id'],encoder_delta=delta,final_delta=bs['gt_score']-s['gt_score'],final_iou=s['iou']))
  count.update(median_encoder_same_token_score_drop=float(np.median(enc_deltas)),median_final_same_token_score_drop=float(np.median(native_deltas)),median_best_valid_common_logit_shift=float(np.median(common)),median_best_valid_gt_specific_shift=float(np.median(class_specific)),same_token_details=same_losses)
  summary[cohort][arm]=count
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(out/'query_lineages.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
for k,v in summary.items():
 for arm,d in v.items():print(k,arm,{k:v for k,v in d.items() if k!='same_token_details'})
