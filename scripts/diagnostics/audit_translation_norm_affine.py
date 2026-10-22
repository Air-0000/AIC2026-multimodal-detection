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
        pairedrows={v['ann_id']:v for v in paired}
        trace_rows=json.loads((args.reference_npz.resolve().parent.parent.parent/'nonexistent').read_text()) if False else None
        # Fixed old baseline final query and token for every GT in the selected images.
        lost=[r for r in paired if 'interior' in r['groups'] and r['arms']['BASELINE']['detected'] and not r['arms']['SHIFT_X1']['detected']]
        chosen={v['image_id'] for v in lost};ref=np.load(npz_path);pos={int(v):i for i,v in enumerate(ref['image_ids'])};runtime={};hooks=[];records=[];checks=[]
        def capture(j,name):
            def hook(module,ins,out):runtime['features'][(j,name)]=out.detach().clone()
            return hook
        def norm_pre(j):
            def hook(module,ins):runtime['features'][(j,'residual')]=ins[0].detach().clone()
            return hook
        for j,layer in enumerate(model.decoder.decoder.layers):
            hooks.append(layer.gateway.register_forward_hook(capture(j,'gateway')));hooks.append(layer.linear2.register_forward_hook(capture(j,'delta')));hooks.append(layer.norm3.register_forward_pre_hook(norm_pre(j)));hooks.append(layer.norm3.register_forward_hook(capture(j,'normalized')))
        original_select=model.decoder._select_topk
        def select(memory,logits,anchors,topk,spatial_shapes=None):
            ss=torch.nan_to_num(logits.detach(),nan=-10.,posinf=10.,neginf=-10.).max(-1).values;n=spatial_shapes[0][0]*spatial_shapes[0][1];k=min(int(topk*.35),n);p=ss[:,:n].topk(k,dim=-1).indices;masked=ss.clone();masked.scatter_(1,p,-1e4);ids=torch.cat([p,masked.topk(topk-k,dim=-1).indices],dim=-1)
            if runtime['arm']=='SHIFT_X1_LOCKED':ids=runtime['base_ids']
            if runtime['arm']=='BASELINE':runtime['base_ids']=ids.detach().clone()
            runtime['ids']=ids;aa=torch.nan_to_num(anchors.gather(1,ids.unsqueeze(-1).expand(-1,-1,4)),nan=0.,posinf=5.,neginf=-5.)
            if runtime['arm']!='SHIFT_X1_LOCKED':
                result=original_select(memory,logits,anchors,topk,spatial_shapes);assert torch.equal(result[2],aa);return result
            return memory.gather(1,ids.unsqueeze(-1).expand(-1,-1,memory.shape[-1])),None,aa
        model.decoder._select_topk=select
        weights=[]
        for j,layer in enumerate(model.decoder.decoder.layers):
            weights.append(dict(layer=j+1,eps=layer.norm3.eps,gamma_abs_mean=float(layer.norm3.weight.abs().mean()),gamma_abs_max=float(layer.norm3.weight.abs().max()),gamma=layer.norm3.weight.detach().cpu().tolist(),beta=layer.norm3.bias.detach().cpu().tolist()))
        for samples,targets in cfg.val_dataloader:
            iid=int(targets[0]['image_id'].item())
            if iid not in chosen:continue
            runtime.clear();source=samples.cuda();collected={}
            for arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
                runtime.update(arm=arm,features={});dx=0 if arm=='BASELINE' else 1;x=source if not dx else torch.cat([torch.zeros_like(source[:,:,:,:1]),source[:,:,:,:-1]],-1)
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):outputs=model(x)
                boxes=outputs['pred_boxes'].clone();boxes[:,:,0]-=dx/2560.;i=pos[iid]
                assert np.array_equal(outputs['pred_logits'][0].cpu().numpy(),ref[arm+'_logits'][i]);assert np.array_equal(boxes[0].cpu().numpy(),ref[arm+'_boxes'][i]);checks.append(dict(image_id=iid,arm=arm,exact=True))
                collected[arm]=dict(ids=runtime['ids'][0].cpu().numpy(),features={k:v[0].cpu().numpy().astype(float) for k,v in runtime['features'].items()})
            for g in [v for v in paired if v['image_id']==iid]:
                i=pos[iid];q=g['arms']['BASELINE']['query']
                if q<0:continue
                token=int(collected['BASELINE']['ids'][q]);gtrow=dict(ann_id=g['ann_id'],image_id=iid,lost='interior' in g['groups'] and g['arms']['BASELINE']['detected'] and not g['arms']['SHIFT_X1']['detected'],stable='interior' in g['groups'] and g['arms']['BASELINE']['detected'] and g['arms']['SHIFT_X1']['detected'],arms={})
                for arm,data in collected.items():
                    indices=np.flatnonzero(data['ids']==token)
                    if not len(indices):continue
                    sq=int(indices[0]);layers=[]
                    for j in range(6):
                        f=data['features'];base=collected['BASELINE']['features'];rr=f[(j,'residual')][sq];gw=f[(j,'gateway')][sq];dd=f[(j,'delta')][sq];yy=f[(j,'normalized')][sq];br=base[(j,'residual')][q];by=base[(j,'normalized')][q]
                        def cos(a,b):return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
                        ww=np.asarray(weights[j]['gamma']);bias=np.asarray(weights[j]['beta']);eps=weights[j]['eps'];a=(rr-rr.mean())/np.sqrt(rr.var()+eps)*ww;ba=(br-br.mean())/np.sqrt(br.var()+eps)*ww
                        assert np.allclose(a+bias,yy,rtol=2e-4,atol=2e-4)
                        layers.append(dict(affine_without_beta_cosine=cos(ba,a),affine_cancellation_ratio=float(np.linalg.norm(yy)/max(np.linalg.norm(a)+np.linalg.norm(bias),1e-12)),baseline_affine_cancellation_ratio=float(np.linalg.norm(by)/max(np.linalg.norm(ba)+np.linalg.norm(bias),1e-12)),relative_input_delta=float(np.linalg.norm(br-rr)/max(np.linalg.norm(br),1e-12)),relative_output_delta=float(np.linalg.norm(by-yy)/max(np.linalg.norm(by),1e-12)),layer=j+1,residual_std=float(rr.std()),baseline_residual_std=float(br.std()),residual_norm=float(np.linalg.norm(rr)),gateway_norm=float(np.linalg.norm(gw)),delta_norm=float(np.linalg.norm(dd)),cancellation_ratio=float(np.linalg.norm(rr)/max(np.linalg.norm(gw)+np.linalg.norm(dd),1e-12)),baseline_cancellation_ratio=float(np.linalg.norm(br)/max(np.linalg.norm(base[(j,'gateway')][q])+np.linalg.norm(base[(j,'delta')][q]),1e-12)),pre_norm_cosine=cos(br,rr),post_norm_cosine=cos(by,yy),centered_pre_norm_cosine=cos(br-br.mean(),rr-rr.mean()),diff_amplification=float(np.linalg.norm(by-yy)/max(np.linalg.norm(br-rr),1e-12)),max_abs_residual=float(np.max(np.abs(rr))),var_to_epsilon=float(rr.var()/weights[j]['eps'])))
                    gtrow['arms'][arm]=layers
                records.append(gtrow)
            print('NORM_IMAGE',iid,len(checks)//3,'/19',flush=True)
            del source,collected,outputs;gc.collect()
        assert len(checks)==57
        for h in hooks:h.remove()
        dump(args.out/'norm_records.json',records)
        row.update(status='complete',evaluated_image_ids=sorted(chosen),exact_forward_matches=checks,norm_records_sha256=sha(args.out/'norm_records.json'),seconds=time.time()-start)
        manifest['norm_contract']=dict(scope='No model changes; same baseline final object query by source token; FFN residual cancellation/LayerNorm epsilon and clamp audit. Not training or historical gradients.',norm_weights=weights,paired_gt_sha256=sha(args.paired_gt),reference_npz_sha256=sha(npz_path))
        dump(args.out/'manifest.json',manifest)
    print('ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
