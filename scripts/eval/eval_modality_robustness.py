#!/usr/bin/env python3
"""Modality-robustness profile of one checkpoint on the labelled validation set (rules §七(二)).

Conditions, all single-image independent inference with the config's own strict loader:
  clean            unchanged inputs
  night{lvl}       pre-generated dark RGB (data_aux/aic_valid_rgb_night{lvl}, brightness x lvl); IR/depth unchanged
  rgb_black        RGB raw pixels 0
  ir_off           both IR channels raw 0 (dead thermal sensor)           6-channel models only
  depth_off        depth raw 0 (invalid everywhere, as on sensor dropout) 6-channel models only
  ir_depth_off     both of the above                                      6-channel models only
"*_off" and rgb_black are applied to the normalized tensor as (0 - mean) / std of the config's Normalize op.

Writes PostProcessor top-300 class/query pairs per condition (no NMS, no score cut), then scores every
condition against clean with compare_detectors_paired (NMS 0.60, per-image cap 100, paired image bootstrap).
Inference only: no training, no TTA, no test data.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
sys.path[:0] = [str(FW), str(ROOT/'scripts/lib'), str(ROOT/'scripts/analysis')]
NIGHT = ['50', '25', '12', '05', '01', '005', '002', '001', '0005']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='+', help='subset of conditions; default: all that apply')
    p.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE', help='val dataset overrides, e.g. raw_ir_folder=/path')
    p.add_argument('--p3-quota', type=float, help='override DFINETransformer.p3_quota_ratio (historical runs used 0.35)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--boot', type=int, default=1000)
    p.add_argument('--expect-clean-map', type=float, help='assert clean standard-COCO mAP (no cap) equals this to 1e-6')
    a = p.parse_args()
    config, ckpt = a.config.resolve(), a.checkpoint.resolve()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    os.chdir(FW)  # relative data and checkpoint paths in configs resolve against vendor/DEIMv2
    import numpy as np
    import torch
    import yaml
    # Bind the reference pycocotools before the engine imports: engine/data/dataset/coco_dataset.py calls
    # faster_coco_eval.init_as_pycocotools(), which replaces pycocotools for every later import.
    from pycocotools.coco import COCO
    import analyze_light_val_thresholds as lvt
    from analyze_light_val_thresholds import nms, evaluate
    from compare_detectors_paired import Tables, compare, render
    from engine.core import YAMLConfig
    from engine.core.yaml_utils import load_config
    assert lvt.COCOeval.__module__ == 'pycocotools.cocoeval' and COCO.__module__ == 'pycocotools.coco', \
        (lvt.COCOeval.__module__, COCO.__module__)
    # The non-strict loader silently skips IR destriping if this import fails; fail here instead.
    from destripe_1d import stripe_non_uniformity_correction_1d
    stripe_non_uniformity_correction_1d(np.zeros((32, 32), dtype=np.uint8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    c = copy.deepcopy(load_config(str(config), cfg={}))
    c.pop('__include__', None)
    c['DEIM'].update(use_dinov3_distill=False, use_co_aux=False, use_internal_tta=False, use_internal_sahi=False)
    c[c['DEIM']['backbone']]['pretrained'] = False
    c['DFINETransformer'].update(use_sdqs=False, use_distinct_queries=False)
    if a.p3_quota is not None:
        c['DFINETransformer']['p3_quota_ratio'] = a.p3_quota
    c['PostProcessor'] = dict(num_classes=c['num_classes'], num_top_queries=300, use_nms=False,
                              depth_cond_enable=False, use_containment_fusion=False)
    val = c['val_dataloader']
    val.update(total_batch_size=1, num_workers=2, shuffle=False, drop_last=False)
    val.pop('persistent_workers', None)
    ds = val['dataset']
    for kv in a.set:
        k, v = kv.split('=', 1)
        ds[k] = yaml.safe_load(v)
    n_ch = ds.get('n_channels', 3)
    norm = next(op for op in ds['transforms']['ops'] if op['type'] == 'Normalize')
    zero = [-m/s for m, s in zip(norm['mean'], norm['std'])]   # normalized value of raw pixel 0, per channel
    assert len(zero) == n_ch, (len(zero), n_ch)
    offs = {'rgb_black': [0, 1, 2]}
    if n_ch == 6:
        offs.update(ir_off=[3, 4], depth_off=[5], ir_depth_off=[3, 4, 5])
    elif n_ch != 3:
        raise ValueError(f'unsupported channel layout n_channels={n_ch}')
    folders = {f'night{lvl}': ROOT/f'data_aux/aic_valid_rgb_night{lvl}' for lvl in NIGHT}
    available = ['clean'] + [k for k, f in folders.items() if f.is_dir()] + list(offs)
    arms = a.arms or available
    unknown = [x for x in arms if x not in available]
    if unknown or arms[0] != 'clean':
        raise ValueError(f'arms must start with clean and come from {available}; got {arms}')

    def cfg_for(arm):
        d = copy.deepcopy(c)
        if arm in folders:
            d['val_dataloader']['dataset']['img_folder'] = str(folders[arm])
        path = a.out/f'{arm}.yml'
        path.write_text(yaml.safe_dump(d, sort_keys=False))
        with contextlib.redirect_stdout(io.StringIO()):
            return YAMLConfig(str(path))

    base = cfg_for('clean')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = state['ema']['module'] if 'ema' in state else state.get('model', state)
    omitted = sorted(k for k in state if k.startswith(('dinov3_distiller.', 'co_aux_head.')))
    state = {k.removeprefix('module.'): v for k, v in state.items() if k not in omitted}
    heads = state['decoder.enc_score_head.weight'].shape[0]
    if heads != c['num_classes']:
        raise ValueError(f'checkpoint has {heads} classes, config {c["num_classes"]}; class-head adaptation is out of scope')
    with contextlib.redirect_stdout(io.StringIO()):
        model = base.model.float().eval()
        model.load_state_dict(state, strict=True)
        post = base.postprocessor.eval()
    model.to(a.device)
    post.to(a.device)
    del state

    ann = Path(ds['ann_file']).resolve()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann))
    manifest = dict(config=str(config), config_sha256=sha(config), checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt),
                    annotation=str(ann), annotation_sha256=sha(ann), script_sha256=sha(__file__), arms=arms,
                    omitted_training_keys=len(omitted), n_channels=n_ch, p3_quota_ratio=c['DFINETransformer'].get('p3_quota_ratio'),
                    dataset_overrides=a.set, device=str(a.device), torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0) if str(a.device).startswith('cuda') else None,
                    precision='FP32; TF32 off; no TTA', candidates='PostProcessor top-300 class/query pairs, no NMS, no score cut',
                    off_semantics='raw pixel 0 on the listed channels, written as (0-mean)/std after Normalize',
                    runs={})
    dump(a.out/'manifest.json', manifest)
    files = {}
    for arm in arms:
        start = time.time()
        loader = (base if arm == 'clean' else cfg_for(arm)).val_dataloader
        rows, seen = [], set()
        for samples, targets in loader:
            x = samples.to(a.device)
            for ch in offs.get(arm, []):
                x[:, ch] = zero[ch]
            sizes = torch.stack([t['orig_size'] for t in targets]).to(a.device)
            with torch.inference_mode(), torch.autocast(device_type=str(a.device).split(':')[0], enabled=False):
                out = model(x)
                assert torch.isfinite(out['pred_logits']).all() and torch.isfinite(out['pred_boxes']).all(), arm
                res = post(dict(pred_logits=out['pred_logits'], pred_boxes=out['pred_boxes']), sizes)
            for t, r in zip(targets, res):
                iid = int(t['image_id'].item())
                seen.add(iid)
                for lab, b, s in zip(r['labels'].tolist(), r['boxes'].tolist(), r['scores'].tolist()):
                    rows.append(dict(image_id=iid, category_id=int(lab), bbox=[b[0], b[1], max(0., b[2]-b[0]), max(0., b[3]-b[1])], score=float(s)))
        assert seen == set(coco.imgs), (arm, len(seen))
        f = a.out/f'{arm}_predictions.json'
        dump(f, rows)
        files[arm] = rows
        manifest['runs'][arm] = dict(predictions_sha256=sha(f), count=len(rows), seconds=round(time.time()-start, 1),
                                     img_folder=str(folders[arm]) if arm in folders else ds['img_folder'], zeroed_channels=offs.get(arm, []))
        dump(a.out/'manifest.json', manifest)
        print(f'{arm}: {len(rows)} candidates in {time.time()-start:.0f}s', flush=True)

    # Standard COCO (NMS 0.60, no cap) for continuity with MODEL_RECORDS, then paired official-rule deltas vs clean.
    summary = {}
    for arm, rows in files.items():
        by = {}
        for r in rows:
            by.setdefault(r['image_id'], []).append(r)
        e = evaluate(coco, [r for batch in by.values() for r in nms(batch, .6)])
        one = np.ones((1, len(coco.imgs)))
        t = Tables(e)
        w = t.summarize(t.precision(one), one)
        summary[arm] = dict(standard=dict(mAP=float(e.stats[0]), AP50=float(e.stats[1]), AP75=float(e.stats[2]),
                                          AP_small=float(e.stats[3]), wAP=float(w['wAP'][0])))
    if a.expect_clean_map is not None:
        got = summary['clean']['standard']['mAP']
        assert abs(got-a.expect_clean_map) <= 1e-6, f'clean mAP {got} != expected {a.expect_clean_map}'
        manifest['reproduction'] = dict(expected_clean_map=a.expect_clean_map, got=got, ok=True)
    args = argparse.Namespace(nms=.6, cap=100, boot=a.boot, seed=0, score_thr=.48)
    for arm in arms[1:]:
        out = compare(coco, files['clean'], files[arm], args)
        out['n_images'] = len(coco.imgs)
        summary[arm]['vs_clean'] = {m: out['stats'][m] for m in ('mAP', 'AP75', 'wAP')}
        summary[arm]['paired_gt'] = out['paired_gt']
        (a.out/f'{arm}_vs_clean.md').write_text(render(out, ['clean', arm], args))
    manifest['summary'] = summary
    dump(a.out/'manifest.json', manifest)

    pp = lambda v: f'{100*v:.2f}'
    ci = lambda s: f"{100*s['delta']:+.2f} [{100*s['ci_delta'][0]:+.2f}, {100*s['ci_delta'][1]:+.2f}]"
    L = ['# Modality robustness profile', '',
         f'Config `{config.name}`, checkpoint `{ckpt.name}` (SHA256 {manifest["checkpoint_sha256"][:12]}), {len(coco.imgs)} labelled validation images, '
         f'{n_ch}-channel input. Standard columns: class-wise NMS 0.60, COCO maxDets 100 (comparable with MODEL_RECORDS). '
         f'Delta columns: official rules (NMS 0.60, <=100 boxes per image), paired image bootstrap n={a.boot}, 95% CI, versus clean.', '',
         '| condition | mAP | AP75 | AP_small | wAP | dmAP vs clean [CI] | dwAP vs clean [CI] | TP@.5 lost/gained |', '|---|---:|---:|---:|---:|---|---|---|']
    for arm in arms:
        s = summary[arm]['standard']
        v = summary[arm].get('vs_clean')
        g = summary[arm].get('paired_gt')
        moved = f"{g['lost50']}/{g['recovered50']}" if g else '-'
        L.append(f"| {arm} | {pp(s['mAP'])} | {pp(s['AP75'])} | {pp(s['AP_small'])} | {pp(s['wAP'])} | "
                 f"{ci(v['mAP']) if v else '-'} | {ci(v['wAP']) if v else '-'} | {moved} |")
    L += ['', 'night{lvl}: pre-generated dark RGB at brightness x lvl with IR/depth unchanged. *_off and rgb_black: raw pixel 0 on those '
          'channels. A robust multimodal model should lose less than an RGB-only control under night* and rgb_black, and degrade '
          'gracefully under ir_off/depth_off. Per-condition paired details: *_vs_clean.md.']
    if 'reproduction' in manifest:
        L.append(f"\nReproduction: clean standard mAP {manifest['reproduction']['got']:.10f} matches the expected value.")
    (a.out/'report.md').write_text('\n'.join(L)+'\n')
    print((a.out/'report.md').read_text())


if __name__ == '__main__':
    main()
