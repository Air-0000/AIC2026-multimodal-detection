"""Build the frozen s0918 branch experiment; no baseline weights are updated."""
import argparse
from pathlib import Path
from mmengine.config import Config

p=argparse.ArgumentParser()
p.add_argument('--root', required=True)
a=p.parse_args(); root=Path(a.root)
cfg=Config.fromfile('/autodl-fs/data/pretrained_models/teammate_s0918/codino_aic_full.py')
cfg.custom_imports['imports'] += ['aic_744_nightgate']
cfg.model.type='AICFrozenCoDETR'
cfg.model.trainable=('data_preprocessor.fuse.',)
cfg.model.data_preprocessor.update(type='DataPreprocessor744',gate=dict(lo=.02,hi=.05),hidden=32,layers=5,erode_px=5,feather_px=8)
# Do not download the unrelated backbone initialiser: load_from contains the full detector.
cfg.model.backbone.init_cfg=None
cfg.load_from='/autodl-fs/data/pretrained_models/teammate_s0918/epoch_12.pth'
loader=dict(type='Load744GateInputs',ir_dir='/root/autodl-tmp/AIC/dataset/train/infrared',depth_dir='/root/autodl-tmp/AIC/dataset/train/depth')
for key,ann in [('train_dataloader','train.json'),('val_dataloader','val.json'),('test_dataloader','val.json')]:
    ds=cfg[key].dataset
    ds.data_root=str(root/'data')+'/'
    ds.ann_file=ann
    ds.data_prefix=dict(img='visible/')
    ds.pipeline=[dict(loader) if t['type']=='LoadImageFromFile' else t for t in ds.pipeline]
    if key=='train_dataloader':
        at=next(i for i,t in enumerate(ds.pipeline) if t['type']=='LoadAnnotations')
        ds.pipeline.insert(at+1,dict(type='RandomNight744',prob=1.,level_range=(.003,.08),photon=70.,read=.012))
    cfg[key].num_workers=2
    cfg[key].persistent_workers=True
cfg.test_pipeline=cfg.test_dataloader.dataset.pipeline
for key in ['val_evaluator','test_evaluator']:
    cfg[key].ann_file=str(root/'data/val.json')
cfg.optim_wrapper.optimizer.lr=.001
cfg.optim_wrapper.type='OptimWrapper'
cfg.optim_wrapper.pop('dtype',None)
cfg.optim_wrapper.pop('loss_scale',None)
cfg.custom_hooks.append(dict(type='Finite744GradientHook'))
cfg.val_cfg=None
cfg.val_dataloader=None
cfg.val_evaluator=None
cfg.train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=4,val_interval=5)
cfg.param_scheduler=[dict(type='LinearLR',start_factor=.001,by_epoch=False,begin=0,end=200),dict(type='CosineAnnealingLR',eta_min=.00005,begin=0,end=4,T_max=4,by_epoch=True,convert_to_iter_based=True)]
cfg.work_dir=str(root/'train')
for h in cfg.custom_hooks:
    if h['type']=='AicIterRecorderHook':h['out_file']=str(root/'train/iter_log.jsonl')
cfg.default_hooks.checkpoint.update(save_optimizer=True,save_param_scheduler=True,max_keep_ckpts=2)
cfg.dump(str(root/'744_gate.py'))
