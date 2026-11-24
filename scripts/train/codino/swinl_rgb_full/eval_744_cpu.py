"""CPU probes of the frozen 744 residual branch; validation images only, resumable JSON."""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmdet.apis import init_detector, inference_detector
import aic_744_nightgate as ng

p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=1);p.add_argument('--benchmark',action='store_true');p.add_argument('--night-only',action='store_true');p.add_argument('--short-side',type=int);p.add_argument('--out',required=True);p.add_argument('--root',default='/root/744_gate');p.add_argument('--checkpoint');p.add_argument('--images');p.add_argument('--ir');p.add_argument('--depth');p.add_argument('--ann');p.add_argument('--threads',type=int,default=1);p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
r=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cfg=Config.fromfile(str(r/'744_gate.py'))
cfg.test_dataloader.dataset.pipeline=[t for t in cfg.test_dataloader.dataset.pipeline if t['type']!='LoadAnnotations']
cfg.test_dataloader.dataset.pipeline[0].update(ir_dir=a.ir or '/root/autodl-tmp/AIC/dataset/train/infrared',depth_dir=a.depth or '/root/autodl-tmp/AIC/dataset/train/depth')
cfg.test_dataloader.dataset.pipeline[0]=ConfigDict(type="LoadImageFromNDArray")
cfg.test_pipeline=cfg.test_dataloader.dataset.pipeline
if a.short_side:
 for t in cfg.test_dataloader.dataset.pipeline:
  if t['type']=='Resize':t['scale']=(a.short_side*2,a.short_side)
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
register_all_modules(init_default_scope=True)
model=MODELS.build(cfg.model)
ck=torch.load(a.checkpoint or str(r/'train/epoch_4.pth'),map_location='cpu',mmap=True,weights_only=False)
model.load_state_dict(ck['state_dict'],strict=True,assign=True)
model.cfg=cfg
model.dataset_meta=ck.get('meta',{}).get('dataset_meta',{})
model.to(a.device)
model.eval()
if a.device=='cpu':
 from cpu_attention_chunks import enable_query_chunks
 enable_query_chunks()
print('DEVICE_MODEL_LOADED',a.device,flush=True)
loader=ng.Load744GateInputs(ir_dir=a.ir or '/root/autodl-tmp/AIC/dataset/train/infrared',depth_dir=a.depth or '/root/autodl-tmp/AIC/dataset/train/depth')
images=sorted(json.load(open(a.ann or r/'data/val.json'))['images'],key=lambda x:x['file_name'])[:a.limit]
branch_forward=model.data_preprocessor.fuse.forward

def set_branch(on):
 # The disabled counterfactual is exactly the zero residual of the zeroed net,
 # without spending CPU time on its five zero convolutions.
 model.data_preprocessor.fuse.forward = branch_forward if on else lambda x: x.new_zeros((1,*x.shape[-2:]))
def predict(x,im):
 start=time.monotonic()
 with torch.inference_mode():v=inference_detector(model,x).pred_instances
 v=v[torch.argsort(v.scores,descending=True,stable=True)[:100]]
 rows=[]
 for box,score,label in zip(v.bboxes.tolist(),v.scores.tolist(),v.labels.tolist()):
  x1,y1,x2,y2=box
  x1,x2=np.clip([x1,x2],0,im['width']);y1,y2=np.clip([y1,y2],0,im['height'])
  if x2<=x1 or y2<=y1:continue
  rows.append(dict(image_id=im['id'],category_id=label,bbox=[x1,y1,x2-x1,y2-y1],score=score))
 return rows,float(model.data_preprocessor.last_gates[0]),time.monotonic()-start
for i,im in enumerate(images):
 target=out/(Path(im['file_name']).stem+'.json')
 if target.exists():continue
 x=loader(dict(img_path=str(Path(a.images or r/'data/visible')/im['file_name'])))['img']
 scenarios=[('normal',x,True)]
 if not a.benchmark:
  scenarios.append(('normal_base',x,False))
  np.random.seed(20260918+i)
  dark=ng.RandomNight744(prob=1.,level_range=(.01,.01),photon=30.,read=.02)(dict(img=x.copy()))['img']
  auxoff=dark.copy();auxoff[...,3:6]=0
  scenarios += [('night01',dark,True),('night01_base',dark,False),('night01_branch_auxoff',auxoff,True)]
  if a.night_only:scenarios=[x for x in scenarios if x[0] in ('night01','night01_base')]
 result={}
 for name,arr,on in scenarios:
  set_branch(on);rows,gate,seconds=predict(arr,im)
  result[name]=dict(predictions=rows,gate=gate,seconds=seconds)
  print(json.dumps(dict(image=i+1,name=name,gate=gate,seconds=seconds,boxes=len(rows))),flush=True)
 if not a.benchmark and not a.night_only:
  assert result['normal']['predictions']==result['normal_base']['predictions'],'normal output changed'
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(target)
print('CPU_PROBES_COMPLETE',flush=True)

if not a.benchmark:
 from pycocotools.coco import COCO
 from pycocotools.cocoeval import COCOeval
 gt=COCO(a.ann or str(r/'data/val.json'))
 records=[json.loads(x.read_text()) for x in out.glob('*.json') if x.stem!='metrics']
 keys=set.intersection(*(set(x) for x in records)) if records else set()
 metrics={}
 for key in sorted(keys):
  rows=[p for x in records for p in x[key]['predictions']]
  if not rows:continue
  dt=gt.loadRes(rows);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=[x['id'] for x in images];ev.evaluate();ev.accumulate();ev.summarize()
  metrics[key]=dict(AP=float(ev.stats[0]),AP50=float(ev.stats[1]),AP75=float(ev.stats[2]),images=len(images))
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
