#!/usr/bin/env python3
"""Layerwise localization of independent-image translation sensitivity; no training."""
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
    p.add_argument('--paired-gt',type=Path,required=True)
    p.add_argument('--reference-npz',type=Path,required=True)
    args = p.parse_args()
    assert len(args.run) == 1
    paired = json.loads(args.paired_gt.resolve().read_text())
    npz_path = args.reference_npz.resolve()
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
    import torch.nn.functional as F
    import engine.deim.dfine_decoder as decoder_module
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
        # Cohort selected before tracing: ALL 19 interior GT lost after +1x.
        lost=[r for r in paired if 'interior' in r['groups'] and r['arms']['BASELINE']['detected'] and not r['arms']['SHIFT_X1']['detected']]
        assert len(lost)==19
        chosen_images={r['image_id'] for r in lost}
        lost_ids={r['ann_id'] for r in lost}
        reference_npz=np.load(npz_path)
        ref_positions={int(v):i for i,v in enumerate(reference_npz['image_ids'])}
        gt_by_image={i:[g for g in ann['annotations'] if g['image_id']==i and not g.get('iscrowd',0) and not g.get('ignore',0)] for i in chosen_images}
        images_by_id={im['id']:im for im in ann['images']}
        from collections import defaultdict
        manifest['trace_cohort']=dict(selection='All 19 interior baseline TP lost under SHIFT_X1 in previous 200-image screen; other GT in these same 19 images retained as descriptive controls; post-selection, not held-out.',lost_gt_ids=sorted(lost_ids),image_ids=sorted(chosen_images),paired_gt_sha256=sha(args.paired_gt),reference_npz_sha256=sha(npz_path))
        runtime={};hooks=[];all_results=[];all_queries={};exact=[]
        spatial=[(180,320),(90,160),(45,80)]
        def pool_roi(t,name):
            # Fixed 7x7 object-relative sampling; align_corners=False pixel-center formula.
            if t.ndim!=4:return
            n=len(runtime['gt'])
            if not n:return
            im=images_by_id[runtime['image_id']]
            grids=[]
            for g in runtime['gt']:
                x,y,w,h=g['bbox'];u=(torch.arange(7,device=t.device,dtype=t.dtype)+.5)/7
                xx=(x+u*w)/im['width']+runtime['dx']/2560.
                yy=(y+u*h)/im['height']
                gy,gx=torch.meshgrid(yy*2-1,xx*2-1,indexing='ij')
                grids.append(torch.stack([gx,gy],dim=-1))
            grid=torch.stack(grids)
            # Avoid expanding dense feature into N copies.
            patches=F.grid_sample(t,grid.reshape(1,n*7,7,2),mode='bilinear',padding_mode='zeros',align_corners=False)
            runtime['dense'][name]=patches[0].reshape(t.shape[1],n,7,7).permute(1,0,2,3).detach().cpu().numpy()
        def dense_hook(name):
            def hook(module,ins,out):
                if isinstance(out,(list,tuple)):
                    for j,t in enumerate(out):
                        if torch.is_tensor(t):pool_roi(t,name+'.'+str(j))
                elif torch.is_tensor(out):pool_roi(out,name)
            return hook
        hooks.append(model.backbone.rgb_net.stem.register_forward_hook(dense_hook('rgb.stem')))
        for j,module in enumerate(model.backbone.rgb_net.stages):hooks.append(module.register_forward_hook(dense_hook('rgb.stage'+str(j))))
        hooks.append(model.backbone.register_forward_hook(dense_hook('backbone.out')))
        hooks.append(model.encoder.register_forward_hook(dense_hook('encoder.out')))
        for j,module in enumerate(model.backbone.ir_res):hooks.append(module.register_forward_hook(dense_hook('fusion.res'+str(j))))
        def enc_memory_hook(module,ins,out):
            off=0
            for j,(h,w) in enumerate(spatial):
                pool_roi(out[:,off:off+h*w].transpose(1,2).reshape(1,-1,h,w),'enc.memory.'+str(j));off+=h*w
            assert off==out.shape[1]
        hooks.append(model.decoder.enc_output.register_forward_hook(enc_memory_hook))
        def feature_hook(name):
            def hook(module,ins,out):
                t=out[0] if isinstance(out,tuple) else out
                runtime['features'][name]=t.detach().clone()
            return hook
        dec=model.decoder.decoder
        hooks.append(dec.register_forward_pre_hook(lambda module,ins:runtime['features'].update(query_input=ins[0].detach().clone())))
        for j,layer in enumerate(dec.layers):
            for sub in ['self_attn','norm1','cross_attn','gateway','norm3']:
                hooks.append(getattr(layer,sub).register_forward_hook(feature_hook('d'+str(j+1)+'.'+sub)))
            hooks.append(layer.register_forward_hook(feature_hook('d'+str(j+1)+'.out')))
        def corners_hook(j):
            def hook(module,ins,out):
                corners=out+runtime.get('previous_corners',0)
                runtime['previous_corners']=corners.detach().clone()
                feat=runtime['features']['d'+str(j+1)+'.out']
                raw=model.decoder.dec_score_head[j](feat)
                corrected=dec.lqe_layers[j](raw,corners)
                runtime['layer_logits'].append(corrected.detach().clone())
                runtime['raw_logits'].append(raw.detach().clone())
            return hook
        for j,module in enumerate(model.decoder.dec_bbox_head):hooks.append(module.register_forward_hook(corners_hook(j)))
        original_distance=decoder_module.distance2bbox
        def distance(*a,**kw):
            result=original_distance(*a,**kw);runtime['layer_boxes'].append(result.detach().clone());return result
        decoder_module.distance2bbox=distance
        original_select=model.decoder._select_topk
        def select(memory,logits,anchors,topk,spatial_shapes=None):
            scores=torch.nan_to_num(logits.detach(),nan=-10.,posinf=10.,neginf=-10.).max(-1).values
            n=spatial_shapes[0][0]*spatial_shapes[0][1];k=min(int(topk*.35),n)
            p=scores[:,:n].topk(k,dim=-1).indices;masked=scores.clone();masked.scatter_(1,p,-1e4)
            ids=torch.cat([p,masked.topk(topk-k,dim=-1).indices],dim=-1)
            if runtime['arm']=='SHIFT_X1_LOCKED':ids=runtime['base_ids']
            if runtime['arm']=='BASELINE':runtime['base_ids']=ids.detach().clone()
            runtime['ids']=ids.detach().clone();runtime['enc_logits']=logits.gather(1,ids.unsqueeze(-1).expand(-1,-1,12)).detach().clone()
            if runtime['arm']!='SHIFT_X1_LOCKED':
                result=original_select(memory,logits,anchors,topk,spatial_shapes)
                expected=torch.nan_to_num(anchors.gather(1,ids.unsqueeze(-1).expand(-1,-1,4)),nan=0.,posinf=5.,neginf=-5.)
                assert torch.equal(result[2],expected)
                return result
            aa=torch.nan_to_num(anchors.gather(1,ids.unsqueeze(-1).expand(-1,-1,4)),nan=0.,posinf=5.,neginf=-5.)
            return memory.gather(1,ids.unsqueeze(-1).expand(-1,-1,memory.shape[-1])),None,aa
        model.decoder._select_topk=select
        def overlap(g,b,im):
            bb=b.copy();bb[:,0]-=bb[:,2]/2;bb[:,1]-=bb[:,3]/2;bb*=np.array([im['width'],im['height'],im['width'],im['height']])
            x,y,w,h=g['bbox'];inter=np.maximum(0,np.minimum(x+w,bb[:,0]+bb[:,2])-np.maximum(x,bb[:,0]))*np.maximum(0,np.minimum(y+h,bb[:,1]+bb[:,3])-np.maximum(y,bb[:,1]))
            return inter/np.maximum(w*h+bb[:,2]*bb[:,3]-inter,1e-12)
        def comparison(b,v):
            x=np.asarray(b,dtype=float).ravel();y=np.asarray(v,dtype=float).ravel()
            return dict(cosine=float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12)),relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-12)))
        for samples,targets in cfg.val_dataloader:
            image_id=int(targets[0]['image_id'].item())
            if image_id not in chosen_images:continue
            assert tuple(samples.shape)==(1,6,1440,2560)
            runtime.clear();runtime.update(image_id=image_id,gt=gt_by_image[image_id]);source=samples.cuda();arm_data={}
            for arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
                dx=0 if arm=='BASELINE' else 1
                runtime.update(arm=arm,dx=dx,features={},dense={},layer_logits=[],raw_logits=[],layer_boxes=[])
                runtime.pop('previous_corners',None)
                variant=source if not dx else torch.cat([torch.zeros_like(source[:,:,:,:1]),source[:,:,:,:-1]],dim=-1)
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):outputs=model(variant)
                corrected=outputs['pred_boxes'].clone();corrected[:,:,0]-=dx/2560.
                ri=ref_positions[image_id]
                assert np.array_equal(outputs['pred_logits'][0].cpu().numpy(),reference_npz[arm+'_logits'][ri]),(image_id,arm,'logits changed by tracing')
                assert np.array_equal(corrected[0].cpu().numpy(),reference_npz[arm+'_boxes'][ri]),(image_id,arm,'boxes changed by tracing')
                assert np.array_equal(runtime['ids'][0].cpu().numpy(),reference_npz[arm+'_selected_ids'][ri])
                assert len(runtime['layer_logits'])==len(runtime['layer_boxes'])==6
                assert torch.equal(runtime['layer_logits'][-1],outputs['pred_logits'])
                exact.append(dict(image_id=image_id,arm=arm,exact_output_match=True))
                data=dict(dense=runtime['dense'],features={k:v[0].cpu().numpy() for k,v in runtime['features'].items()},ids=runtime['ids'][0].cpu().numpy(),enc_logits=runtime['enc_logits'][0].cpu().numpy(),raw=np.stack([v[0].cpu().numpy() for v in runtime['raw_logits']]),logits=np.stack([v[0].cpu().numpy() for v in runtime['layer_logits']]),boxes=np.stack([v[0].cpu().numpy() for v in runtime['layer_boxes']]))
                data['boxes'][:,:,0]-=dx/2560.
                # Same final classifier readout on sublayer features: diagnostic projection, not intermediate calibrated detection.
                with torch.inference_mode():data['fixed_readout']={k:model.decoder.dec_score_head[-1](v).detach()[0].cpu().numpy() for k,v in runtime['features'].items() if v.shape[-1]==model.decoder.dec_score_head[-1].in_features}
                arm_data[arm]=data
                for key in ['ids','enc_logits','raw','logits','boxes']:all_queries[str(image_id)+'_'+arm+'_'+key]=data[key]
            for gi,g in enumerate(runtime['gt']):
                record=dict(ann_id=g['id'],image_id=image_id,category_id=g['category_id'],lost_cohort=g['id'] in lost_ids,arms={},dense_pairs={})
                im=images_by_id[image_id];cat=g['category_id'];base=arm_data['BASELINE']
                for arm,d in arm_data.items():
                    probs=1/(1+np.exp(-np.clip(d['logits'][:,:,cat].astype(float),-80,80)))
                    ovs=np.stack([overlap(g,b,im) for b in d['boxes']]);valid=ovs>=.5
                    q=int(np.argmax(np.where(valid[-1],probs[-1],-1))) if valid[-1].any() else int(ovs[-1].argmax())
                    bprobs=1/(1+np.exp(-np.clip(base['logits'][-1,:,cat].astype(float),-80,80)));bov=overlap(g,base['boxes'][-1],im)
                    bq=int(np.argmax(np.where(bov>=.5,bprobs,-1))) if (bov>=.5).any() else int(bov.argmax())
                    token=int(base['ids'][bq]);same=np.flatnonzero(d['ids']==token);sq=int(same[0]) if len(same) else None
                    native=[]
                    for j in range(6):
                        native.append(dict(layer=j+1,best_valid_score=float(np.max(np.where(valid[j],probs[j],0))),max_iou=float(ovs[j].max()),endpoint_query_score=float(probs[j,q]),endpoint_query_iou=float(ovs[j,q]),endpoint_raw_logit=float(d['raw'][j,q,cat]),endpoint_lqe_delta=float(d['logits'][j,q,cat]-d['raw'][j,q,cat]),same_token_query_score=float(probs[j,sq]) if sq is not None else None))
                    features={}
                    for name,v in d['features'].items():
                        bb=base['features'][name]
                        item=dict(endpoint_pair=comparison(bb[bq],v[q]),endpoint_fixed_logit=float(d['fixed_readout'][name][q,cat]) if name in d['fixed_readout'] else None)
                        if sq is not None:item['same_token_pair']=comparison(bb[bq],v[sq]);item['same_token_fixed_logit']=float(d['fixed_readout'][name][sq,cat]) if name in d['fixed_readout'] else None
                        features[name]=item
                    record['arms'][arm]=dict(endpoint_query=q,baseline_endpoint_query=bq,source_token=int(d['ids'][q]),baseline_source_token=token,same_token_query=sq,encoder_endpoint_logit=float(d['enc_logits'][q,cat]),layers=native,features=features)
                    if arm!='BASELINE':record['dense_pairs'][arm]={name:comparison(base['dense'][name][gi],v[gi]) for name,v in d['dense'].items()}
                all_results.append(record)
            print('TRACE_IMAGE',image_id,'completed',len(exact)//3,'/19',flush=True)
            del arm_data,outputs,variant,source,samples;gc.collect()
        for hook in hooks:hook.remove()
        decoder_module.distance2bbox=original_distance
        assert len(exact)==57 and {v['image_id'] for v in exact}==chosen_images
        np.savez_compressed(args.out/'layer_queries.npz',**all_queries)
        dump(args.out/'layer_gt_trace.json',all_results)
        row.update(status='complete',evaluated_image_ids=sorted(chosen_images),exact_forward_matches=exact,layer_queries_sha256=sha(args.out/'layer_queries.npz'),layer_gt_trace_sha256=sha(args.out/'layer_gt_trace.json'),seconds=time.time()-start)
        manifest['trace_contract']='Hooks must exactly reproduce previous full-screen logits, corrected boxes and selected tokens on all 57 forwards. Native heads/LQE differ across layers. Fixed final classifier readouts on intermediate features are diagnostic projections, not calibrated probabilities. Dense 7x7 GT ROI samples align to translated object coordinates; differences are not causal attribution.'
        dump(args.out/'manifest.json',manifest)
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
