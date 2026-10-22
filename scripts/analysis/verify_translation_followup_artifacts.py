#!/usr/bin/env python3
"""Verify completed frozen-weight follow-up bundles and write integrity manifests."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
names=['channel_interventions','stem_preblur','gamma16_screen','translation_phases','encoder_level_interventions','encoder_alignment_interventions','translation_directions_fullval','translation_negative_phases','translation_padding_controls','local_context_interventions','local_random_controls','box_ring_interventions','query_memory_interventions','memory_layer_interventions','relu_component_interventions','reference_alignment_interventions','reference_component_interventions','single_query_geometry_interventions','other_query_height_interventions','cross_query_attention_trace','cross_query_component_trace']
reference=json.loads((ROOT/'work_dir/audits/light_val_20261021/results_v3/manifest.json').read_text());reference_run=next(r for r in reference['runs'] if r['name']=='V36');checks=[]
for stem in names:
 name=stem+'_20261021';base=ROOT/'work_dir/audits'/name;bundle=base/'raw';m=json.loads((bundle/'manifest.json').read_text());r=m['runs'][0];assert r['status']=='complete';assert r['checkpoint_sha256']==reference_run['checkpoint_sha256'];assert r['head_classes']==12 and r['strict']
 for key in ['annotation_sha256','inputs_sha256','source_sha256','library_sha256','common_val_dataloader']:assert m[key]==reference[key],(name,key)
 assert sha(bundle/'val_inputs.json')==m['inputs_sha256'];assert all(v['exact'] for v in r['exact_forward_matches']);assert len(r['exact_forward_matches']) in [3,38,57,400]
 for key in ['intervention_outputs','candidate_outputs']:
  if key+'_sha256' in r:assert sha(bundle/(key+'.npz'))==r[key+'_sha256']
 if 'query_attention_trace_sha256' in r:assert sha(bundle/'query_attention_trace.npz')==r['query_attention_trace_sha256']
 for arm,v in r['arms'].items():assert sha(bundle/(arm+'_predictions.json'))==v['prediction_sha256']
 assert any(sha(f)==m['script_sha256'] for f in (base/'source').glob('*.py'))
 extra={k:len(v) for k,v in r.items() if k.startswith('exact_') and k!='exact_forward_matches'}
 for key in extra:assert all(v['exact'] for v in r[key])
 report=ROOT/'outputs/reports'/(name+'_attempt2' if stem in ['encoder_level_interventions','cross_query_attention_trace'] else name)
 files=[f for f in base.rglob('*') if f.is_file() and f.name!='artifact_sha256.json']+[f for f in report.rglob('*') if f.is_file()];hashes={str(f.relative_to(ROOT)):sha(f) for f in files};dump(base/'artifact_sha256.json',hashes);assert all(sha(ROOT/f)==h for f,h in hashes.items());checks.append(dict(audit=name,files=len(hashes),arms=len(r['arms']),exact_controls=len(r['exact_forward_matches']),extra_exact_controls=extra))
clean=json.loads((ROOT/'work_dir/audits/aic_clean_baseline_v1_20261021/source_final_manifest.json').read_text());assert len(clean)==124 and all(sha(ROOT/f)==h for f,h in clean.items())
out=ROOT/'outputs/reports/translation_cause_followup_20261021';dump(out/'artifact_verification.json',dict(scope='Completed frozen historical V36 inference diagnostics; not training or clean-v1 evaluation.',baseline124_unchanged=True,reference_manifest_sha256=sha(ROOT/'work_dir/audits/light_val_20261021/results_v3/manifest.json'),verifier_sha256=sha(Path(__file__)),experiments=checks,total_verified_files=sum(v['files'] for v in checks)))
files=[f for f in out.rglob('*') if f.is_file() and f.name!='artifact_sha256.json']+[ROOT/'work_dir/audits'/(stem+'_20261021')/'artifact_sha256.json' for stem in names]+[ROOT/'logs/single_query_geometry_interventions_failed_attempt1_20261021.log',ROOT/'logs/encoder_level_interventions_analysis_20261021.log',ROOT/'logs/cross_query_attention_trace_analysis_failed_attempt1_20261021.log']
dump(out/'artifact_sha256.json',{str(f.relative_to(ROOT)):sha(f) for f in files});print(json.dumps(dict(experiments=len(checks),files=sum(v['files'] for v in checks),baseline124_unchanged=True)))
