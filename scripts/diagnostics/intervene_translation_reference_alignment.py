#!/usr/bin/env python3
"""Paired activation interventions for translation losses; validation-only counterfactuals."""
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
    p.add_argument('--box-reference-npz',type=Path,required=True)
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
        from collections import defaultdict
        lost=[r for r in paired if 'interior' in r['groups'] and r['arms']['BASELINE']['detected'] and not r['arms']['SHIFT_X1']['detected']]
        assert len(lost)==19
        chosen={r['image_id'] for r in lost};lost_ids={r['ann_id'] for r in lost}
        ref=np.load(npz_path);refpos={int(v):i for i,v in enumerate(ref['image_ids'])}
        interventions={a:(0,'BRANCH') for a in ['BOX_REF','BOX_INIT','BOX_REF_ALIGNED','BOX_INIT_ALIGNED','ALL_ALL']}
        gt_by_id={g['id']:g for g in ann['annotations']};im_by_id={g['id']:g for g in ann['images']};lost_by_image={r['image_id']:r['ann_id'] for r in lost};assert len(lost_by_image)==19
        boxref=np.load(args.box_reference_npz.resolve());boxpos={int(v):i for i,v in enumerate(boxref['image_ids'])};branch_checks=[];mask_records=[]
        arms=['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']+list(interventions)
        runtime={};hooks=[];predictions={a:[] for a in arms};arrays={a:defaultdict(list) for a in arms};seen=[];checks=[]
        def active(j,kind):
            target=interventions.get(runtime['arm']);return target is not None and target[1]==kind and (target[0]==j or target[0]==-1)
        def branch_patch(module,ins):
            target,refs,memory,shapes=ins[:4];arm=runtime['arm'];runtime['seen_refs']=refs.detach().clone()
            if arm=='BASELINE':
                runtime['cache']['decoder_inputs']=(target.detach().clone(),refs.detach().clone(),memory.detach().clone());return None
            if arm not in interventions:return None
            bt,br,bm=runtime['cache']['decoder_inputs']
            if arm=='ALL_ALL':
                runtime['seen_refs']=br.detach().clone();return (bt.clone(),br.clone(),bm.clone(),*ins[3:])
            g=gt_by_id[lost_by_image[runtime['image_id']]];im=im_by_id[runtime['image_id']];x,y,bw,bh=g['bbox'];masks=[]
            for h,w in shapes:
                mask=torch.zeros((h,w),device=memory.device,dtype=torch.bool);x0=max(0,int(np.floor(x*w/im['width'])));x1=min(w,int(np.ceil((x+bw)*w/im['width'])));y0=max(0,int(np.floor(y*h/im['height'])));y1=min(h,int(np.ceil((y+bh)*h/im['height'])));mask[y0:y1,x0:x1]=True;masks.append(mask.flatten())
            mm=torch.cat(masks)[None,:];qm=mm.gather(1,runtime['base_ids']);mask_records.append(dict(image_id=runtime['image_id'],arm=arm,memory_positions=int(mm.sum()),query_positions=int(qm.sum())))
            if arm in ['BOX_INIT','BOX_INIT_ALIGNED']:target=torch.where(qm[:,:,None],bt,target)
            replacement=br
            if arm.endswith('_ALIGNED'):
                replacement=br.clone();cx=(br[:,:,0].sigmoid()+1/2560.).clamp(1e-5,1-1e-5);replacement[:,:,0]=torch.log(cx/(1-cx))
            refs=torch.where(qm[:,:,None],replacement,refs);runtime['seen_refs']=refs.detach().clone()
            if arm in ['BOX_MEMORY','BOX_JOINT']:memory=torch.where(mm[:,:,None],bm,memory)
            return (target,refs,memory,*ins[3:])
        hooks.append(model.decoder.decoder.register_forward_pre_hook(branch_patch))
        original_select=model.decoder._select_topk
        def select(memory,logits,anchors,topk,spatial_shapes=None):
            ss=torch.nan_to_num(logits.detach(),nan=-10.,posinf=10.,neginf=-10.).max(-1).values;n=spatial_shapes[0][0]*spatial_shapes[0][1];k=min(int(topk*.35),n)
            p=ss[:,:n].topk(k,dim=-1).indices;masked=ss.clone();masked.scatter_(1,p,-1e4);ids=torch.cat([p,masked.topk(topk-k,dim=-1).indices],dim=-1)
            if runtime['arm'] not in ['BASELINE','SHIFT_X1']:ids=runtime['base_ids']
            if runtime['arm']=='BASELINE':runtime['base_ids']=ids.detach().clone()
            runtime['ids']=ids
            aa=torch.nan_to_num(anchors.gather(1,ids.unsqueeze(-1).expand(-1,-1,4)),nan=0.,posinf=5.,neginf=-5.)
            if runtime['arm'] in ['BASELINE','SHIFT_X1']:
                result=original_select(memory,logits,anchors,topk,spatial_shapes);assert torch.equal(result[2],aa);return result
            return memory.gather(1,ids.unsqueeze(-1).expand(-1,-1,memory.shape[-1])),None,aa
        model.decoder._select_topk=select
        manifest['intervention_contract']=dict(scope='19 symptom-selected validation images; all patch arms compared with SHIFT_X1_LOCKED. Baseline activations are forbidden oracle inputs for deployment; no temporal inference or training.',arms=arms,cohort_lost_gt_ids=sorted(lost_ids),paired_gt_sha256=sha(args.paired_gt),reference_npz_sha256=sha(npz_path),encoder_patch='Reference-coordinate confound audit: fixed baseline queryIDs; originalGTBOX querymask. OriginalBOX_REF/INIT mustexactlyreplay precedingquery-memory bundle38forwards. ALIGNED arms restorebaselineinitialbboxwithcenterx sigmoidshifted+1/2560 thenlogit; other3componentskeptbitwise. Originalimage querycontent restoredonly INITarms. CurrentmemoryunchangedexceptALL_ALLpositivecontrol. Saveinitialreferences eachforward. OracleGT/activations forbiddenfordeployment; not atestofcoordinateequivariance ofallfeatures; no training.',patch_scope='all 300 corresponding original token IDs, not annotation-selected queries')
        manifest['intervention_contract']['box_reference_npz_sha256']=sha(args.box_reference_npz.resolve())
        for samples,targets in cfg.val_dataloader:
            image_id=int(targets[0]['image_id'].item())
            if image_id not in chosen:continue
            runtime.clear();runtime.update(cache={},image_id=image_id);source=samples.cuda();shift=torch.cat([torch.zeros_like(source[:,:,:,:1]),source[:,:,:,:-1]],dim=-1);sizes=torch.stack([t['orig_size'] for t in targets]).cuda();seen.append(image_id)
            for arm in arms:
                runtime['arm']=arm;dx=0 if arm=='BASELINE' else 1
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                    outputs=model(source if not dx else shift);boxes=outputs['pred_boxes'].clone();boxes[:,:,0]-=dx/2560.;result=post(dict(pred_logits=outputs['pred_logits'],pred_boxes=boxes),sizes)[0]
                assert torch.isfinite(boxes).all() and torch.isfinite(outputs['pred_logits']).all()
                if arm in ['BASELINE','SHIFT_X1','SHIFT_X1_LOCKED']:
                    i=refpos[image_id];assert np.array_equal(outputs['pred_logits'][0].cpu().numpy(),ref[arm+'_logits'][i]);assert np.array_equal(boxes[0].cpu().numpy(),ref[arm+'_boxes'][i]);checks.append(dict(image_id=image_id,arm=arm,exact=True))
                if arm in ['BOX_REF','BOX_INIT']:
                    bp=boxpos[image_id];assert np.array_equal(outputs['pred_logits'][0].cpu().numpy(),boxref[arm+'_logits'][bp]);assert np.array_equal(boxes[0].cpu().numpy(),boxref[arm+'_boxes'][bp]);branch_checks.append(dict(image_id=image_id,arm=arm,exact=True))
                if arm=='ALL_ALL':
                    i=refpos[image_id];assert np.array_equal(outputs['pred_logits'][0].cpu().numpy(),ref['BASELINE_logits'][i]);assert np.array_equal(outputs['pred_boxes'][0].cpu().numpy(),ref['BASELINE_boxes'][i])
                arrays[arm]['init_ref_unact'].append(runtime['seen_refs'][0].cpu().numpy());arrays[arm]['boxes'].append(boxes[0].cpu().numpy());arrays[arm]['logits'].append(outputs['pred_logits'][0].cpu().numpy());arrays[arm]['ids'].append(runtime['ids'][0].cpu().numpy())
                for lab,b,score in zip(result['labels'].cpu().tolist(),result['boxes'].cpu().tolist(),result['scores'].cpu().tolist()):
                    x,y,x2,y2=b;predictions[arm].append(dict(image_id=image_id,category_id=lab,bbox=[x,y,max(0,x2-x),max(0,y2-y)],score=score))
                del outputs,result,boxes
            print('PATCH_IMAGE',image_id,len(seen),'/19',flush=True)
            del samples,source,shift;gc.collect()
        assert len(seen)==19 and set(seen)==chosen and len(checks)==57
        for h in hooks:h.remove()
        row['arms']={};save={'image_ids':np.array(seen)}
        for arm in arms:
            f=args.out/(arm+'_predictions.json');dump(f,predictions[arm]);row['arms'][arm]=dict(prediction_sha256=sha(f),count=len(predictions[arm]))
            for k,v in arrays[arm].items():save[arm+'_'+k]=np.array(v)
        np.savez_compressed(args.out/'intervention_outputs.npz',**save)
        assert len(branch_checks)==38
        row['exact_box_joint_matches']=branch_checks
        row.update(status='complete',evaluated_image_ids=seen,exact_forward_matches=checks,intervention_outputs_sha256=sha(args.out/'intervention_outputs.npz'),seconds=time.time()-start)
        dump(args.out/'manifest.json',manifest)
        dump(args.out/'mask_records.json',mask_records)
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
