"""Fail closed before full training: gate bypass, prediction parity and frozen-state smoke."""
import copy, json, hashlib
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
import aic_744_nightgate as ng

root=Path('/root/744_gate'); cfg=Config.fromfile(str(root/'744_gate.py'))
ck=cfg.load_from
basecfg=Config.fromfile('/autodl-fs/data/pretrained_models/teammate_s0918/codino_aic_full.py')
basecfg.model.backbone.init_cfg=None
for c in (cfg,basecfg):
    c.test_dataloader.dataset.pipeline=[t for t in c.test_dataloader.dataset.pipeline if t['type']!='LoadAnnotations']
    c.test_pipeline=c.test_dataloader.dataset.pipeline
loader=ng.Load744GateInputs(ir_dir='/root/autodl-tmp/AIC/dataset/train/infrared',depth_dir='/root/autodl-tmp/AIC/dataset/train/depth')
ims=json.load(open(root/'data/val.json'))['images'][:3]
reference={'00000253.png': '631730883a0b8c75515fdadc8e730e2a0cd2ae72751a2703fa4bbf36a58105a2', '00000257.png': '23432938ff60ed42d85b74cc3366a3502183c2574fe9c0a0f3d6ab49dfc49079', '00000271.png': '01a18597d8b96f6c777be78ea2c25549b95310667bd9d38b6bebc5406f11cd41', '00000273.png': '417bccee7cdc584c5008864f060449350b60c4964fc9265be4f9281d74a57f8c', '00000275.png': 'e2f2c8d9ab4b70925291fb0e78eaa9af77218b5a7255f3ec1744e11d7f3659c2', '00000282.png': '6e05bf7e692b30e7b076da1324757eaec44c1e6fc3ea78df6e70dedf332f0ef4', '00000291.png': 'f3c20691875961821286a843c23228cb99477264660c37f01bff9f3c75a35051', '00000293.png': '060948f29bc0f5e64b118a52c527d90874426bec306e2a4534e8620f61f0c1c1', '000002_081_00000001.png': '32db4cb4f6aa159fe63abb54a90c9dd26c418712a80e5765149c7b089c8a6303', '000002_081_00000047.png': 'd55c6039f5b82e9d9253d1914a4a02abbc77d47bf670ffaf38211ff3c7e282e7', '000002_081_00000185.png': '1970a51917386d0df7a884f7f12bda331997224494ed8a1cf660efcef73020bc', '000003_014_00000001.png': '1ca14791d735391d1ad1a6955940cee68d6d49fd623b6cb08cb2f043bb34a788'}
for im in json.load(open(root/'data/val.json'))['images']:
    key=Path(im['file_name']).stem+'.png'
    if key not in reference: continue
    arr=loader(dict(img_path=str(root/'data/visible'/im['file_name'])))['img'][...,:3][...,::-1].copy()
    assert hashlib.sha256(arr.tobytes()).hexdigest()==reference[key], ('744 pixel mismatch',key)
print('PASS original fused pixel hashes',flush=True)
base=init_detector(basecfg,ck,device='cuda:0')
gate=init_detector(cfg,ck,device='cuda:0')
assert all(not p.requires_grad for n,p in gate.named_parameters() if not n.startswith('data_preprocessor.fuse.'))
print('TRAINABLE',[(n,p.numel()) for n,p in gate.named_parameters() if p.requires_grad],flush=True)
for im in ims:
    path=str(root/'data/visible'/im['file_name'])
    x=loader(dict(img_path=path))['img']
    t=torch.from_numpy(x.transpose(2,0,1).copy()).cuda()
    result=gate.data_preprocessor.fuse_one(t,False)
    assert torch.equal(result,t[:3]) and gate.data_preprocessor.last_gates[-1]==0.
    b=inference_detector(base,x[...,:3].copy()).pred_instances
    g=inference_detector(gate,path).pred_instances
    assert all(torch.equal(getattr(b,k),getattr(g,k)) for k in ('bboxes','scores','labels')),'prediction parity failed'
print('PASS normal fused-input bypass and exact predictions on 3 images',flush=True)
del base,gate;torch.cuda.empty_cache()
s=copy.deepcopy(cfg);s.work_dir=str(root/'smoke');s.train_dataloader.dataset.indices=24
s.train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=1,val_interval=2)
s.param_scheduler=[dict(type='ConstantLR',factor=1.,by_epoch=False,begin=0,end=24)]
for h in s.custom_hooks:
    if h['type']=='AicIterRecorderHook': h['out_file']=str(root/'smoke/iter_log.jsonl')
Runner.from_cfg(s).train()
new=torch.load(root/'smoke/epoch_1.pth',map_location='cpu',weights_only=False)['state_dict']
old=torch.load(ck,map_location='cpu',weights_only=False)['state_dict']
rest={k:v for k,v in new.items() if not k.startswith('data_preprocessor.fuse.')}
assert set(rest)==set(old),(set(rest)^set(old))
assert all(torch.equal(rest[k],v) for k,v in old.items()),'frozen detector changed'
assert new['data_preprocessor.fuse.net.8.weight'].abs().sum()>0,'no fusion update'
(root/'CONTRACT_PASS').write_text('Exact normal predictions; all detector tensors unchanged after 24 training steps; branch updated.\n')
print('CONTRACT_PASS',flush=True)
