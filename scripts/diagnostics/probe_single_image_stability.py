#!/usr/bin/env python3
"""Independent-image V36 structural stress tests; validation only, no training."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-manifest', type=Path, required=True)
    p.add_argument('--run', nargs=3, action='append', required=True, metavar=('NAME', 'CONFIG', 'CKPT'))
    args = p.parse_args()
    assert len(args.run) == 1
    reference = json.loads(args.reference_manifest.resolve().read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    # CUDA_VISIBLE_DEVICES must identify a single currently unused physical GPU.
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not physical.isdigit():
        raise ValueError('Select exactly one physical GPU')
    query = subprocess.check_output(['nvidia-smi', '-i', physical,
                                    '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if query.strip():
        raise RuntimeError('Selected GPU has existing compute processes: ' + query)
    framework = args.framework.resolve()
    root = args.data_root.resolve()
    sys.path[:0] = [str(framework), str(framework / 'scripts_lib')]
    os.chdir(framework)
    import numpy as np
    import torch
    import yaml
    from engine.core import YAMLConfig
    from engine.core.yaml_utils import load_config
    # The dataset historically catches import failures and silently skips IR NUC.
    # Verify this required dependency before creating any prediction rows.
    from destripe_1d import stripe_non_uniformity_correction_1d
    stripe_non_uniformity_correction_1d(np.zeros((32,32), dtype=np.uint8))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    def fresh(path):
        return copy.deepcopy(load_config(str(path), cfg={}))

    ann_path = root / 'work_dir/deimv2_dataset/valid/annotations/instances_val.json'
    ann = json.loads(ann_path.read_text())
    assert len(ann['images']) == 200 and len({x['id'] for x in ann['images']}) == 200
    folders = dict(img_folder=root / 'dataset/train/visible',
                   ir_folder=root / 'dataset/train/infrared_aligned',
                   raw_ir_folder=root / 'work_dir/train_lbg_ir',
                   depth_folder=root / 'dataset/train/depth')
    inputs = []
    for image in ann['images']:
        item = dict(image_id=image['id'], file_name=image['file_name'], files={})
        stem = Path(image['file_name']).stem
        for key, folder in folders.items():
            candidates = [folder / image['file_name']] if key == 'img_folder' else [folder / (stem + '.png'), folder / image['file_name']]
            if key == 'depth_folder':
                candidates = [folder / (stem + e) for e in ('.png', '.jpg', '.jpeg')]
            f = next(f for f in candidates if f.is_file())
            item['files'][key] = dict(path=str(f), sha256=sha(f))
        inputs.append(item)
    dump(args.out / 'val_inputs.json', inputs)
    common = fresh(framework / args.run[0][1])['val_dataloader']
    common.update(total_batch_size=1, num_workers=2, drop_last=False, shuffle=False)
    common.pop('persistent_workers', None)
    ds = common['dataset']
    ds.update({k: str(v) for k, v in folders.items()})
    ds.update(ann_file=str(ann_path), n_channels=6, depth_read='raw16', depth_mode='raw',
              remap_mscoco_category=False)
    ds['transforms']['ops'] = [
        dict(type='Resize', size=[1440, 2560]),
        dict(type='Normalize', mean=[.485,.456,.406,.5,.5,.5], std=[.229,.224,.225,.5,.5,.5]),
        dict(type='ConvertBoxes', fmt='cxcywh', normalize=True)]
    common['collate_fn'] = dict(type='BatchImageCollateFunction', stop_epoch=0,
                               base_size=2560, base_size_repeat=None)
    manifest = dict(annotation_path=str(ann_path), annotation_sha256=sha(ann_path),
                    inputs_sha256=sha(args.out / 'val_inputs.json'), common_val_dataloader=common,
                    source_sha256={str(f.relative_to(framework)):sha(f) for f in sorted((framework/'engine').rglob('*.py'))},
                    library_sha256={str(f.relative_to(framework)):sha(f) for f in sorted((framework/'scripts_lib').glob('*.py'))},
                    script_sha256=sha(__file__), precision='FP32; TF32 disabled', size=[1440,2560],
                    selection='ema.module', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    candidate_scope='PostProcessor top-300 class/query pairs before NMS; no score cutoff; not raw logits',
                    scope='Common ailab40 inputs; checkpoint comparison, not causal ablation; no test data', runs=[])
    manifest['reference_manifest_sha256'] = sha(args.reference_manifest.resolve())
    assert manifest['annotation_sha256'] == reference['annotation_sha256']
    assert manifest['inputs_sha256'] == reference['inputs_sha256']
    assert manifest['source_sha256'] == reference['source_sha256']
    assert manifest['library_sha256'] == reference['library_sha256']
    assert common == reference['common_val_dataloader']
    manifest['competition_contract'] = 'Independent-image inference only; no temporal model/state, no submission or ensemble. Locked selection is diagnostic only.'
    dump(args.out / 'manifest.json', manifest)
    for name, conf, ckpt in args.run:
        start = time.time()
        c = fresh(framework / conf)
        trained = c['train_dataloader']['dataset']
        if trained.get('depth_read','raw16') != 'raw16' or trained.get('depth_mode','raw') != 'raw' or Path(trained['depth_folder']).name != 'depth':
            raise ValueError('Training depth contract differs: ' + name)
        c.pop('__include__', None)
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        state = checkpoint['ema']['module']
        head_classes = state['decoder.enc_score_head.weight'].shape[0]
        assert head_classes == 12
        ref_run = next(r for r in reference['runs'] if r['name'] == 'V36')
        assert sha(ckpt) == ref_run['checkpoint_sha256']
        c.update(num_classes=head_classes, remap_mscoco_category=False, val_dataloader=copy.deepcopy(common), eval_spatial_size=[1440,2560])
        c['DEIM'].update(use_dinov3_distill=False, use_co_aux=False, use_internal_tta=False, use_internal_sahi=False)
        c[c['DEIM']['backbone']]['pretrained'] = False
        # The historical SDQS flags were unused; preserve effective historical query quota.
        c['DFINETransformer'].update(num_classes=head_classes, use_sdqs=False, use_distinct_queries=False, p3_quota_ratio=.35)
        c['PostProcessor'] = dict(num_classes=head_classes, num_top_queries=300, use_nms=False, depth_cond_enable=False, use_containment_fusion=False)
        cfgpath = args.out / (name + '.yml')
        cfgpath.write_text(yaml.safe_dump(c, sort_keys=False))
        with patch('engine.core.yaml_config.load_config', fresh):
            cfg = YAMLConfig(str(cfgpath))
        model = cfg.model.float().eval()
        omitted = [k for k in state if k.startswith(('dinov3_distiller.', 'co_aux_head.'))]
        clean = {k.removeprefix('module.'):v for k,v in state.items() if k not in omitted}
        model.load_state_dict(clean, strict=True)
        row = dict(name=name, config=conf, checkpoint=ckpt, checkpoint_sha256=sha(ckpt),
                   epoch=checkpoint.get('last_epoch', checkpoint.get('epoch')), head_classes=head_classes, strict=True, omitted_training_keys=omitted,
                   overrides=dict(use_sdqs=False, use_distinct_queries=False, p3_quota_ratio=.35),
                   status='running', evaluated_image_ids=[])
        manifest['runs'].append(row)
        dump(args.out/'manifest.json', manifest)
        del clean, state, checkpoint
        gc.collect()
        model.cuda()
        post = cfg.postprocessor.cuda().eval()
        arms = ['BASELINE', 'SHIFT_X1', 'SHIFT_X4', 'SHIFT_Y1',
                'SHIFT_X1_LOCKED', 'SHIFT_X4_LOCKED', 'SA5_SELF_ONLY', 'SA5_HALF',
                'P3_50', 'FUSION_HALF', 'FUSION_ZERO', 'IR_SHIFT_X4']
        manifest['arms'] = arms
        predictions = {arm: [] for arm in arms}
        query_boxes = {arm: [] for arm in arms}
        query_logits = {arm: [] for arm in arms}
        selections = {arm: [] for arm in arms}
        seen = []
        runtime = {'arm':'BASELINE', 'baseline_ids':None}
        original_select = model.decoder._select_topk
        def select(memory, logits, anchors, topk, spatial_shapes=None):
            assert spatial_shapes is not None and memory.shape[0] == 1
            scores = torch.nan_to_num(logits.detach(), nan=-10., posinf=10., neginf=-10.).max(-1).values
            p3_len = spatial_shapes[0][0] * spatial_shapes[0][1]
            p3_k = min(int(topk * (.50 if runtime['arm']=='P3_50' else .35)), p3_len)
            p3_ids = scores[:,:p3_len].topk(p3_k, dim=-1).indices
            masked = scores.clone();masked.scatter_(1,p3_ids,-1e4)
            indices = torch.cat([p3_ids, masked.topk(topk-p3_k, dim=-1).indices],dim=-1)
            if runtime['arm'].endswith('LOCKED'):
                indices = runtime['baseline_ids']
            selected_anchors = anchors.gather(1, indices.unsqueeze(-1).expand(-1,-1,4))
            selected_anchors = torch.nan_to_num(selected_anchors,nan=0.,posinf=5.,neginf=-5.)
            if runtime['arm']=='BASELINE':
                runtime['baseline_ids'] = indices.detach().clone()
            runtime['ids'] = indices.detach().cpu().numpy()
            runtime['spatial_shapes'] = spatial_shapes
            if runtime['arm'] not in ('P3_50','SHIFT_X1_LOCKED','SHIFT_X4_LOCKED'):
                result = original_select(memory,logits,anchors,topk,spatial_shapes)
                assert torch.equal(result[2],selected_anchors), 'Selection observer must exactly match native selection'
                return result
            selected_memory = memory.gather(1,indices.unsqueeze(-1).expand(-1,-1,memory.shape[-1]))
            return selected_memory, None, selected_anchors
        model.decoder._select_topk = select
        def self_pre(module, positional, kwargs):
            if runtime['arm']!='SA5_SELF_ONLY':return None
            assert kwargs.get('attn_mask') is None
            length=positional[0].shape[1]
            mask=~torch.eye(length,dtype=torch.bool,device=positional[0].device)
            kwargs=dict(kwargs,attn_mask=mask)
            return positional,kwargs
        def self_post(module,positional,result):
            if runtime['arm']=='SA5_HALF':return (result[0]*.5,result[1])
            return None
        sa = model.decoder.decoder.layers[4].self_attn
        hooks=[sa.register_forward_pre_hook(self_pre,with_kwargs=True),sa.register_forward_hook(self_post)]
        def fusion_hook(module,positional,result):
            if runtime['arm']=='FUSION_HALF':return result*.5
            if runtime['arm']=='FUSION_ZERO':return result*0.
            return None
        hooks.extend(module.register_forward_hook(fusion_hook) for module in model.backbone.ir_res)
        def translate(x,dx=0,dy=0):
            shifted=torch.zeros_like(x)
            shifted[:,:,dy:,dx:] = x[:,:,:x.shape[-2]-dy,:x.shape[-1]-dx]
            return shifted
        for i, (samples, targets) in enumerate(cfg.val_dataloader):
            assert tuple(samples.shape) == (1,6,1440,2560)
            sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
            image_id = int(targets[0]['image_id'].item()); seen.append(image_id)
            source = samples.cuda()
            runtime['baseline_ids'] = None
            for arm in arms:
                runtime['arm'] = arm
                dx = 4 if arm.startswith('SHIFT_X4') else 1 if arm.startswith('SHIFT_X1') else 0
                dy = 1 if arm=='SHIFT_Y1' else 0
                variant = translate(source,dx,dy) if dx or dy else source
                if arm=='IR_SHIFT_X4':
                    variant=source.clone();variant[:,3:5]=translate(source[:,3:5],4,0)
                with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                    outputs=model(variant)
                    boxes=outputs['pred_boxes'].clone()
                    boxes[:,:,0]-=dx/2560.;boxes[:,:,1]-=dy/1440.
                    corrected=dict(pred_logits=outputs['pred_logits'],pred_boxes=boxes)
                    result=post(corrected,sizes)[0]
                assert np.isfinite(boxes.cpu().numpy()).all() and torch.isfinite(outputs['pred_logits']).all()
                query_boxes[arm].append(boxes[0].cpu().numpy())
                query_logits[arm].append(outputs['pred_logits'][0].cpu().numpy())
                selections[arm].append(runtime['ids'][0])
                for lab, box, score in zip(result['labels'].cpu().tolist(),result['boxes'].cpu().tolist(),result['scores'].cpu().tolist()):
                    x1,y1,x2,y2=box
                    predictions[arm].append(dict(image_id=image_id,category_id=lab,bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)],score=score))
                del outputs,result,corrected,boxes,variant
            if (i+1)%10==0:print('STABILITY',i+1,'/200',round(time.time()-start,1),'sec',flush=True)
        assert len(seen)==200 and set(seen)=={im['id'] for im in ann['images']}
        for hook in hooks:hook.remove()
        row['arms'] = {}
        for arm in arms:
            f=args.out/(arm+'_predictions.json');dump(f,predictions[arm])
            row['arms'][arm]=dict(prediction_sha256=sha(f),prediction_count=len(predictions[arm]))
        arrays = dict(image_ids=np.asarray(seen), spatial_shapes=np.asarray(runtime['spatial_shapes']))
        for arm in arms:
            arrays[arm+'_boxes']=np.asarray(query_boxes[arm],dtype=np.float32)
            arrays[arm+'_logits']=np.asarray(query_logits[arm],dtype=np.float32)
            arrays[arm+'_selected_ids']=np.asarray(selections[arm],dtype=np.int64)
        np.savez_compressed(args.out/'decoder_outputs.npz',**arrays)
        row.update(status='complete',evaluated_image_ids=seen,seconds=time.time()-start,
                   decoder_outputs_sha256=sha(args.out/'decoder_outputs.npz'),
                   baseline_reproduced_old_prediction_sha=row['arms']['BASELINE']['prediction_sha256']==ref_run['prediction_sha256'])
        manifest['padding_contract']='All-channel translations pad normalized tensors with 0; inverse-map predictions to original coordinates without clipping. Border GT are reported separately.'
        manifest['diagnostic_contract']='Decoder NPZ contains all 300 boxes and all 12 logits, distinct from PostProcessor top300 class/query pairs. Every image resets locked query state.'
        dump(args.out/'manifest.json',manifest)
        del model, post, cfg, samples
        gc.collect()
        torch.cuda.empty_cache()
    print('ALL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
