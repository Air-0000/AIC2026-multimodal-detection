#!/usr/bin/env python3
"""Frozen native matcher and TADL derivative replay; CPU logits only, no model training."""
import sys,json,hashlib,contextlib,io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
framework=Path('/home/ailab40/aic_light_val_20261021/framework_v3');sys.path[:0]=[str(framework),str(framework/'scripts_lib')]
from engine.deim.matcher import HungarianMatcher
from engine.deim.deim_criterion import DEIMCriterion
from engine.core.yaml_utils import load_config
root=Path('/home/ailab40/aic_light_val_20261021');out=root/'matcher_replay_20261021';out.mkdir(exist_ok=False)
trace=root/'layer_translation_trace_20261021';z=np.load(trace/'layer_queries.npz');rows=json.loads((trace/'layer_gt_trace.json').read_text());annpath=Path('/home/ailab/ssd/bigdata/hejianzhong/AIC_V29/work_dir/deimv2_dataset/valid/annotations/instances_val.json');ann=json.loads(annpath.read_text());ims={v['id']:v for v in ann['images']};config=load_config(str(trace/'V36.yml'),cfg={});cc=config['DEIMCriterion'];mc=cc['matcher'];assert config['use_focal_loss'] and cc['losses']==['tadl','boxes','local']
torch.set_num_threads(2)
matchers={name:HungarianMatcher(weight_dict=weights,use_focal_loss=True,alpha=mc['alpha'],gamma=mc['gamma']) for name,weights in [('native',mc['weight_dict']),('geometry_only',dict(mc['weight_dict'],cost_class=0.)),('class_only',dict(cost_class=mc['weight_dict']['cost_class'],cost_bbox=0.,cost_giou=0.))]}
assert not cc['use_ndsl'];fake=SimpleNamespace(tadl_min_iou=cc['tadl_min_iou'],tadl_beta=cc['tadl_beta'],cls_neg_scale=cc['cls_neg_scale'],num_classes=12,use_ndsl=False,gamma=cc['gamma'],alpha=cc['alpha'],_get_src_permutation_idx=lambda idx:(torch.cat([torch.full_like(s,i) for i,(s,t) in enumerate(idx)]),torch.cat([s for s,t in idx])))
results=[]
for iid in sorted({r['image_id'] for r in rows}):
 gt=[v for v in ann['annotations'] if v['image_id']==iid and not v.get('iscrowd',0) and not v.get('ignore',0)];im=ims[iid];boxes=np.array([g['bbox'] for g in gt],dtype=np.float32);boxes[:,:2]+=boxes[:,2:]/2;boxes/=np.array([im['width'],im['height'],im['width'],im['height']],dtype=np.float32);targets=[dict(boxes=torch.from_numpy(boxes),labels=torch.tensor([g['category_id'] for g in gt]))];rby={r['ann_id']:r for r in rows if r['image_id']==iid}
 cache={}
 for arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
  ids=z[f'{iid}_{arm}_ids']
  for layer in range(6):
   logits=torch.from_numpy(z[f'{iid}_{arm}_logits'][layer].copy()).unsqueeze(0).requires_grad_(True);outputs=dict(pred_logits=logits,pred_boxes=torch.from_numpy(z[f'{iid}_{arm}_boxes'][layer].copy()).unsqueeze(0));matches={k:matcher(outputs,targets)['indices'] for k,matcher in matchers.items()}
   loss=DEIMCriterion.loss_labels_tadl(fake,outputs,targets,matches['native'],len(gt))['loss_tadl'];grad=torch.autograd.grad(loss,logits)[0][0]
   for name,idx in matches.items():
    lookup={int(t):int(q) for q,t in zip(idx[0][0],idx[0][1])}
    for gi,g in enumerate(gt):
     rr=rby[g['id']];q=lookup[gi];key=(layer,name,g['id'])
     if arm=='BASELINE':cache[key]=int(ids[q])
     baseline_token=rr['arms']['BASELINE']['baseline_source_token'];same=np.flatnonzero(ids==baseline_token);sq=int(same[0]) if len(same) else None
     native_lookup={int(q):int(t) for q,t in zip(matches['native'][0][0],matches['native'][0][1])}
     results.append(dict(ann_id=g['id'],image_id=iid,lost=rr['lost_cohort'],arm=arm,layer=layer+1,matcher=name,matched_query=q,matched_token=int(ids[q]),baseline_matching_token=cache[key],matched_gt_score=float(logits[0,q,g['category_id']].sigmoid()),baseline_endpoint_token_present=sq is not None,baseline_endpoint_token_matched_to_same_gt=sq is not None and native_lookup.get(sq)==gi,baseline_endpoint_gt_logit_derivative=float(grad[sq,g['category_id']]) if sq is not None else None))
 print('MATCH_IMAGE',iid,flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'records.json').write_text(json.dumps(results,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(status='complete',scope='Inference-output matcher replay only, not historical training assignments. CPU derivative wrt saved logits only, no model/optimizer/backward through model/weight updates.',images=19,records=len(results),native_matcher_sha256=sha(framework/'engine/deim/matcher.py'),native_criterion_sha256=sha(framework/'engine/deim/deim_criterion.py'),config_sha256=sha(trace/'V36.yml'),npz_sha256=sha(trace/'layer_queries.npz'),annotation_sha256=sha(annpath),script_sha256=sha(__file__),records_sha256=sha(out/'records.json')),indent=2)+'\n');print('ALL_COMPLETE',flush=True)
