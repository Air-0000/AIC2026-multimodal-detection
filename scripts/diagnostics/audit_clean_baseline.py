#!/usr/bin/env python3
"""Read-only corrected-baseline acceptance: provenance, COCO load, inference/reload.

No optimizer, backward, training epoch or scored checkpoint evaluation is run.
Use a fresh --out; all full-data IO and GPU verification belong on the server.
"""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--reuse-input-audit',type=Path,
                        help='Reuse decoded metadata only; rehash every input against this successful report')
    args=parser.parse_args()
    cached_inputs={}
    if args.reuse_input_audit:
        reference_path=args.reuse_input_audit.resolve()
        ref=json.loads(reference_path.read_text())
        cached_file=reference_path.parent/'inputs.json'
        assert ref['status']=='passed_untrained_baseline_contract' and sha(cached_file)==ref['inputs_sha256']
        for row in json.loads(cached_file.read_text()):
            for field,meta in row['files'].items():cached_inputs[(row['split'],row['image_id'],field)]=meta
    framework=args.framework.resolve();config=args.config.resolve();out=args.out.resolve()
    if str(framework).startswith('/home/constant/'):
        parser.error('Full-data and GPU acceptance is remote-only')
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical!=str(args.gpu):parser.error('CUDA_VISIBLE_DEVICES must select exactly --gpu')
    active=subprocess.check_output(['nvidia-smi','-i',physical,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if active.strip():raise RuntimeError('GPU already occupied: '+active)
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(framework)
    sys.path[:0]=[str(framework),str(framework.parents[1]/'scripts/lib')]
    import numpy as np
    from PIL import Image
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver._solver import BaseSolver
    from engine.misc.numerical_contract import require_finite
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if torch.cuda.get_device_capability(0)==(7,0):torch.backends.cudnn.enabled=False
    cfg=YAMLConfig(str(config));c=cfg.yaml_cfg
    assert c['clean_baseline'] and c['stop_on_nonfinite'] and c['strict_resume']
    assert c['num_classes']==c['DFINETransformer']['num_classes']==c['PostProcessor']['num_classes']==12
    assert c['DFINETransformer']['p3_quota_ratio']==.35 and not c['use_amp']
    for flag in ('use_sdqs','use_distinct_queries','use_anisotropic_routing','use_spatial_attn_bias','use_layer_ensemble','use_offset_clamp'):
        assert c['DFINETransformer'][flag] is False
    assert c['HGNetv2MultiModal']['disable_fallback']
    report=dict(status='running',scope='Corrected baseline code/input/load/inference acceptance only; no training and no mAP claim',
                torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_gpu=args.gpu,
                config_sha256=sha(config),script_sha256=sha(__file__),source_sha256={str(f.relative_to(framework)):sha(f) for f in sorted((framework/'engine').rglob('*.py'))},
                train_py_sha256=sha(framework/'train.py'),seed=cfg.seed,tf32=False,cudnn_enabled=torch.backends.cudnn.enabled)
    def save(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save()
    inputs=[];annotations={};split_ids={};depth_modes=Counter();val_modes={}
    for split,key,expected in [('train','train_dataloader',1800),('valid','val_dataloader',200)]:
        ds=c[key]['dataset'];annpath=Path(ds['ann_file']);ann=json.loads(annpath.read_text())
        assert len(ann['images'])==expected and len({x['id'] for x in ann['images']})==expected
        assert [x['name'] for x in sorted(ann['categories'],key=lambda x:x['id'])]==['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
        assert sorted(x['id'] for x in ann['categories'])==list(range(12))
        annotations[split]=dict(path=str(annpath.resolve()),sha256=sha(annpath),images=expected,annotations=len(ann['annotations']))
        split_ids[split]={x['file_name'] for x in ann['images']}
        for image in ann['images']:
            item=dict(split=split,image_id=image['id'],file_name=image['file_name'],files={})
            stem=Path(image['file_name']).stem
            for field in ('img_folder','ir_folder','raw_ir_folder','depth_folder'):
                folder=Path(ds[field])
                names=[image['file_name']] if field=='img_folder' else [stem+'.png',image['file_name']]
                if field=='depth_folder':names=[stem+e for e in ('.png','.jpg','.jpeg')]
                path=next((folder/name for name in names if (folder/name).is_file()),None)
                if path is None:raise FileNotFoundError((folder,image['file_name']))
                content_sha=sha(path)
                cached=cached_inputs.get((split,image['id'],field))
                if cached:
                    assert cached['path']==str(path.resolve()) and cached['sha256']==content_sha
                    mode,size=cached['mode'],cached['size']
                else:
                    with Image.open(path) as im:
                        im.load();mode=im.mode;size=list(im.size)
                item['files'][field]=dict(path=str(path.resolve()),sha256=content_sha,mode=mode,size=size)
                if field=='depth_folder':
                    depth_modes[mode]+=1
                    if split=='valid':val_modes[image['id']]=mode
            inputs.append(item)
            if len(inputs)%200==0:print('INPUT_PROGRESS',len(inputs),'images',flush=True)
        print('INPUTS',split,len(ann['images']),flush=True)
    assert not split_ids['train']&split_ids['valid']
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    report.update(annotations=annotations,input_files=sum(len(x['files']) for x in inputs),inputs_sha256=sha(out/'inputs.json'),depth_modes=dict(depth_modes),depth_semantics='Historical mixed formats retained: I/I;16 /19999, others /255; no claim all images are calibrated metric depth')
    report['coco_checkpoint_sha256']=sha(c['tuning'])
    if report['coco_checkpoint_sha256']!='39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b':raise ValueError('Unexpected COCO starting checkpoint')
    report['teacher_checkpoint_sha256']=sha(c['DEIM']['dinov3_teacher_path'])
    save()
    dist_utils.setup_seed(cfg.seed)
    model=cfg.model.cpu().eval();solver=BaseSolver(cfg);solver.model=model
    solver.load_tuning_state(c['tuning'])
    report['initialization']=solver.tuning_load_report
    rgb=model.backbone.rgb_net.stem.stem1.conv.weight.detach();ir=model.backbone.ir_stem.stem[0].weight.detach()
    mean=rgb.mean(1,keepdim=True);lap=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).view(1,1,3,3)*.5
    expected=torch.cat([mean.expand(-1,2,-1,-1),(mean*.5+lap*.5).expand(-1,2,-1,-1)])
    torch.testing.assert_close(ir,expected,atol=0,rtol=0)
    # Mode switching only; no forward/backward in train mode.
    model.train();assert not model.dinov3_distiller.teacher.training
    assert model.dinov3_distiller.proj_head.training
    model.eval()
    inference_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith(('dinov3_distiller.','co_aux_head.'))}
    torch.save({'model':inference_state,'source':'COCO-initialized corrected baseline; untrained'},out/'initial_inference_state.pth')
    config2=copy.deepcopy(c);config2['DEIM'].update(use_dinov3_distill=False,use_co_aux=False);config2['tuning']=None
    import yaml
    cfgpath=out/'inference_only.yml';cfgpath.write_text(yaml.safe_dump(config2,sort_keys=False))
    reload_model=YAMLConfig(str(cfgpath)).model.cpu().eval()
    reloaded=torch.load(out/'initial_inference_state.pth',map_location='cpu',weights_only=False)['model']
    reload_model.load_state_dict(reloaded,strict=True)
    report['strict_inference_state_keys']=len(inference_state)
    del inference_state,reloaded;gc.collect()
    model.cuda();reload_model.cuda();post=cfg.postprocessor.cuda().eval()
    selected=[]
    for mode in ('I;16','I','RGB','L'):
        candidates=sorted(k for k,v in val_modes.items() if v==mode)
        if candidates:selected.append(candidates[0])
    selected=set(selected[:3]);assert selected
    matches=[]
    for samples,targets in cfg.val_dataloader:
        image_id=int(targets[0]['image_id'].item())
        if image_id not in selected:continue
        assert tuple(samples.shape)==(1,6,1440,2560)
        samples=samples.cuda();sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
        with torch.inference_mode():
            first=model(samples);second=reload_model(samples)
            require_finite(first.items(),'corrected model forward');require_finite(second.items(),'strict reload forward')
            delta={key:float((first[key]-second[key]).abs().max()) for key in ('pred_logits','pred_boxes')}
            for key in delta:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
            predictions=post(second,sizes,samples)
            for result in predictions:
                require_finite(result.items(),'postprocess');assert bool(((result['labels']>=0)&(result['labels']<12)).all())
        matches.append(dict(image_id=image_id,depth_mode=val_modes[image_id],max_abs_difference=delta))
        print('FORWARD_RELOAD',matches[-1],flush=True)
        if len(matches)==len(selected):break
    assert len(matches)==len(selected)
    report.update(status='passed_untrained_baseline_contract',inference_state_sha256=sha(out/'initial_inference_state.pth'),reload_checks=matches,new_training_started=False)
    save();print('BASELINE_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
