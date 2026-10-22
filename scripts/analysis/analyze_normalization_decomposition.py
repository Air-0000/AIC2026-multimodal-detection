#!/usr/bin/env python3
"""Reconstruct measured Norm3 inputs/outputs; separate centering, scale, gamma, beta."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/reports/normalization_screen_20261022';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=ROOT/'work_dir/audits/translation_cause_tests_20261021/affine/manifest.json'
b=ROOT/'work_dir/audits/cross_query_component_trace_20261021/raw/manifest.json'
c=b.parent/'query_attention_trace.npz'
m=json.loads(a.read_text());t=json.loads(b.read_text());assert m['runs'][0]['checkpoint_sha256']==t['runs'][0]['checkpoint_sha256'];assert sha(c)==t['runs'][0]['query_attention_trace_sha256']
z=np.load(c);rows=[]
def cos(x,y):return float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-15))
for j,w in enumerate(m['norm_contract']['norm_weights']):
 gamma=np.asarray(w['gamma']);beta=np.asarray(w['beta']);eps=w['eps']
 vals={}
 for arm in ['SHIFT_X1_LOCKED','OTHERS_H']:
  x=z[f'{arm}_L{j}_component_norm3_input'].astype(float);actual=z[f'{arm}_L{j}_component_norm3'].astype(float);center=x-x.mean();unit=center/np.sqrt(x.var()+eps);scaled=unit*gamma;affine=scaled+beta
  assert np.allclose(affine,actual,rtol=3e-4,atol=3e-4)
  vals[arm]={'input':x,'centered':center,'unit':unit,'gamma':scaled,'affine':affine,'actual':actual}
 p,q=vals.values()
 rows.append(dict(layer=j+1,cosine={k:cos(p[k],q[k]) for k in p},std={arm:float(v['input'].std()) for arm,v in vals.items()},var_to_epsilon={arm:float(v['input'].var()/eps) for arm,v in vals.items()},gamma_abs_max=float(np.abs(gamma).max()),gamma_abs_median=float(np.median(np.abs(gamma))),gamma_abs_min=float(np.abs(gamma).min()),beta_norm=float(np.linalg.norm(beta)),output_l2={arm:float(np.linalg.norm(v['actual'])) for arm,v in vals.items()}))
d={'scope':'Offline reconstruction of measured same-q77 trace, locked shifted vs OTHERS_H. No forward intervention. Scalar standard-deviation division cannot change pairwise cosine by itself.','sources':{str(p.relative_to(ROOT)):sha(p) for p in [a,b,c]},'rows':rows}
d['absolute_distance_note']={}
for component in ['norm3_input','norm3']:
 x=z['SHIFT_X1_LOCKED_L1_component_'+component].astype(float);y=z['OTHERS_H_L1_component_'+component].astype(float)
 d['absolute_distance_note'][component+'_difference_l2']=float(np.linalg.norm(x-y))
d['absolute_distance_note']['interpretation']='Cosine divergence increases while absolute Euclidean difference decreases; these are distinct measurements, not evidence of blanket numerical explosion.'
(OUT/'norm3_decomposition.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(rows[1],indent=2))
# Broader descriptive check on the old failure-selected 19-image trace cohort.
src=ROOT/'work_dir/audits/translation_cause_tests_20261021/affine/norm_records.json'
records=json.loads(src.read_text());assert sha(src)==m['runs'][0]['norm_records_sha256'];groups={}
for group in ['lost','stable']:
 selected=[x for x in records if x[group] and 'SHIFT_X1_LOCKED' in x['arms']];layers=[]
 for j in range(6):
  rr=[x['arms']['SHIFT_X1_LOCKED'][j] for x in selected]
  layers.append(dict(layer=j+1,n=len(rr),pre_cos_median=float(np.median([x['pre_norm_cosine'] for x in rr])),post_cos_median=float(np.median([x['post_norm_cosine'] for x in rr])),cos_drop_median=float(np.median([x['pre_norm_cosine']-x['post_norm_cosine'] for x in rr])),centered_cos_median=float(np.median([x['centered_pre_norm_cosine'] for x in rr])),no_beta_cos_median=float(np.median([x['affine_without_beta_cosine'] for x in rr])),std_min=float(np.min([x['residual_std'] for x in rr])),var_to_eps_min=float(np.min([x['var_to_epsilon'] for x in rr]))))
 groups[group]=layers
(OUT/'historical_norm_cohort.json').write_text(json.dumps(dict(scope='Saved historical 19 selected images; original baseline vs locked +1x token. Descriptive, not a full-validation cohort and not an intervention.',source=str(src.relative_to(ROOT)),sha256=sha(src),groups=groups),indent=2)+'\n')
