#!/usr/bin/env python3
"""Single-model inference to official submission files, through the training data loader.

The image list comes from --images: a folder of RGB images (test sets), or a COCO json whose
'images' are used and 'annotations' ignored (labelled checks). Every channel is read by the config's
own val-set CocoDetectionMultiModal in strict mode, so preprocessing is the training code path and a
missing IR / LBG IR / depth file is an error rather than a silent fallback. Detections come from the
config's PostProcessor (single forward, no TTA).

Output: one <stem>.txt per image, lines "<class> <cx> <cy> <w> <h> <conf>" normalized to [0, 1],
CRLF, descending confidence, at most --max-dets lines, empty file when nothing passes --score-thr.
With --zip: archive with the txt files at its root, scripts/eval/check_submission.py (folder mode),
and registration in the visualization workstation (AGENTS.md rule 11) unless --no-workstation.

Example (test set, 6-channel model):
  python scripts/inference/predict.py --config vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml \
    --checkpoint CKPT --images dataset/test/AIC2026_PHASE_2_1000/visible \
    --rgb-dir dataset/test/AIC2026_PHASE_2_1000/visible --ir-dir work_dir/test_aligned_ir \
    --lbg-dir work_dir/test_lbg_ir --depth-dir dataset/test/AIC2026_PHASE_2_1000/depth \
    --score-thr 0.45 --out-dir work_dir/submissions/x/txt --zip work_dir/submissions/x/submission.zip
"""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
CLASS_NAMES = ['person', 'boat', 'animal', 'seat', 'sign', 'bicycle', 'car', 'ball', 'light', 'garbage can', 'uav', 'tricycle']
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def stub_annotations(images, rgb_dir):
    """COCO dict with images and no annotations; ids are kept from a json source."""
    from PIL import Image
    images, rgb_dir = Path(images), Path(rgb_dir)
    if images.is_dir():
        names = sorted(f for f in os.listdir(images) if f.lower().endswith(IMAGE_EXTS))
        entries = []
        for i, f in enumerate(names):
            with Image.open(images/f) as im:
                entries.append(dict(id=i, file_name=f, width=im.width, height=im.height))
    else:
        data = json.loads(images.read_text())
        entries = [dict(id=im['id'], file_name=im['file_name'], width=im['width'], height=im['height'])
                   for im in sorted(data['images'], key=lambda im: im['id'])]
    missing = [e['file_name'] for e in entries if not (rgb_dir/e['file_name']).is_file()]
    if not entries or missing:
        raise FileNotFoundError(f'{len(entries)} images listed, {len(missing)} missing in {rgb_dir}: {missing[:5]}')
    return dict(images=entries, annotations=[], categories=[dict(id=i, name=n) for i, n in enumerate(CLASS_NAMES)])


def load_config(config, overrides=None):
    """Expanded config dict with training-only branches off; overrides: dict of top-level updates."""
    sys.path[:0] = [p for p in (str(FW), str(ROOT/'scripts/lib')) if p not in sys.path]
    from engine.core.yaml_utils import load_config as _load
    config = Path(config).resolve()  # the CLI path is relative to the caller's working directory
    cwd = os.getcwd()
    os.chdir(FW)  # relative paths inside configs resolve against vendor/DEIMv2
    try:
        c = copy.deepcopy(_load(str(config), cfg={}))
    finally:
        os.chdir(cwd)
    c.pop('__include__', None)
    c['DEIM'].update(use_dinov3_distill=False, use_co_aux=False, use_internal_tta=False, use_internal_sahi=False)
    c[c['DEIM']['backbone']]['pretrained'] = False
    c['DFINETransformer'].update(use_sdqs=False, use_distinct_queries=False)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


def configure_inputs(c, ann_file, rgb_dir, ir_dir=None, lbg_dir=None, depth_dir=None, batch=1):
    ds = c['val_dataloader']['dataset']
    if ds.get('type') != 'CocoDetectionMultiModal':
        raise ValueError(f"val dataset type {ds.get('type')} is not CocoDetectionMultiModal")
    aux = dict(ir_folder=ir_dir, raw_ir_folder=lbg_dir, depth_folder=depth_dir)
    if ds.get('modality') == 'rgb':
        given = [k for k, v in aux.items() if v]
        if given:
            raise ValueError(f'RGB-only config: do not pass {given}')
    else:
        if ds.get('n_channels') != 6 or not all(aux.values()):
            raise ValueError('6-channel config needs --ir-dir, --lbg-dir and --depth-dir')
        ds.update({k: str(Path(v).resolve()) for k, v in aux.items()})
    ds.update(img_folder=str(Path(rgb_dir).resolve()), ann_file=str(Path(ann_file).resolve()), strict_inputs=True)
    val = c['val_dataloader']
    val.update(total_batch_size=batch, num_workers=2, shuffle=False, drop_last=False)
    val.pop('persistent_workers', None)
    return c


def build(c, workdir, name='predict'):
    """YAMLConfig built from the expanded dict (written next to the outputs for provenance)."""
    import yaml
    from engine.core import YAMLConfig
    path = Path(workdir)/f'{name}_effective.yml'
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    cwd = os.getcwd()
    os.chdir(FW)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return YAMLConfig(str(path))
    finally:
        os.chdir(cwd)


def load_model(cfg, checkpoint, device):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state['ema']['module'] if 'ema' in state else state.get('model', state)
    state = {k.removeprefix('module.'): v for k, v in state.items() if not k.startswith(('dinov3_distiller.', 'co_aux_head.'))}
    heads = state['decoder.enc_score_head.weight'].shape[0]
    if heads != len(CLASS_NAMES):
        raise ValueError(f'checkpoint has {heads} classes; expected {len(CLASS_NAMES)}')
    with contextlib.redirect_stdout(io.StringIO()):
        model = cfg.model.float().eval()
        model.load_state_dict(state, strict=True)
        post = cfg.postprocessor.eval()
    return model.to(device), post.to(device)


def run(cfg, model, post, device):
    """{file_name: (width, height, [(label, score, x1, y1, x2, y2), ...])} in original pixels."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader = cfg.val_dataloader
    coco = loader.dataset.coco
    out = {}
    for samples, targets in loader:
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        with torch.inference_mode(), torch.autocast(device_type=str(device).split(':')[0], enabled=False):
            o = model(samples.to(device))
            if not (torch.isfinite(o['pred_logits']).all() and torch.isfinite(o['pred_boxes']).all()):
                raise FloatingPointError('non-finite model output')
            res = post(o, sizes)
        for t, r in zip(targets, res):
            name = coco.imgs[int(t['image_id'].item())]['file_name']
            w, h = (int(v) for v in t['orig_size'].tolist())
            dets = [(int(lab), float(s), *map(float, b)) for lab, s, b in zip(r['labels'].tolist(), r['scores'].tolist(), r['boxes'].tolist())]
            out[name] = (w, h, dets)
    if len(out) != len(coco.imgs):
        raise RuntimeError(f'{len(out)} predictions for {len(coco.imgs)} images')
    return out


def lines_for(w, h, dets, score_thr, max_dets):
    rows = []
    for lab, s, x1, y1, x2, y2 in sorted(dets, key=lambda d: -d[1]):
        if s < score_thr or len(rows) == max_dets:
            continue
        x1, x2 = min(max(x1, 0.), w), min(max(x2, 0.), w)
        y1, y2 = min(max(y1, 0.), h), min(max(y2, 0.), h)
        if x2 <= x1 or y2 <= y1 or not 0 <= lab < len(CLASS_NAMES):
            continue
        rows.append(f'{lab} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f} {min(max(s, 0.), 1.):.6f}')
    return rows


def write_submission(preds, out_dir, score_thr, max_dets):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, (w, h, dets) in sorted(preds.items()):
        rows = lines_for(w, h, dets, score_thr, max_dets)
        (out_dir/(Path(name).stem+'.txt')).write_bytes(''.join(r+'\r\n' for r in rows).encode())
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True, help='RGB image folder, or COCO json listing the images')
    p.add_argument('--rgb-dir', type=Path, required=True)
    p.add_argument('--ir-dir', type=Path)
    p.add_argument('--lbg-dir', type=Path)
    p.add_argument('--depth-dir', type=Path)
    p.add_argument('--score-thr', type=float, required=True, help='choose on labelled validation data, never from test-set box counts')
    p.add_argument('--max-dets', type=int, default=100)
    p.add_argument('--out-dir', type=Path, required=True, help='txt output folder (must not exist)')
    p.add_argument('--zip', type=Path)
    p.add_argument('--p3-quota', type=float, help='override DFINETransformer.p3_quota_ratio (historical runs used 0.35)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--no-workstation', action='store_true', help='skip workstation registration (e.g. reviewer machines)')
    p.add_argument('--workstation-id')
    p.add_argument('--workstation-name')
    a = p.parse_args()
    a.out_dir = a.out_dir.resolve()
    work = a.out_dir.parent/(a.out_dir.name+'_meta')
    work.mkdir(parents=True, exist_ok=False)
    ann = work/'images.json'
    ann.write_text(json.dumps(stub_annotations(a.images, a.rgb_dir)))
    overrides = {'DFINETransformer': {'p3_quota_ratio': a.p3_quota}} if a.p3_quota is not None else None
    c = configure_inputs(load_config(a.config, overrides), ann, a.rgb_dir, a.ir_dir, a.lbg_dir, a.depth_dir, a.batch)
    cfg = build(c, work)
    model, post = load_model(cfg, a.checkpoint.resolve(), a.device)
    preds = run(cfg, model, post, a.device)
    write_submission(preds, a.out_dir, a.score_thr, a.max_dets)
    total = sum(len(lines_for(w, h, d, a.score_thr, a.max_dets)) for w, h, d in preds.values())
    print(f'{len(preds)} images, {total} boxes written to {a.out_dir}')
    if a.images.is_dir():
        check = subprocess.run([sys.executable, str(ROOT/'scripts/eval/check_submission.py'), str(a.out_dir), str(a.images.resolve())])
        if check.returncode:
            sys.exit('check_submission.py failed')
    else:
        print('image list from json: skipped check_submission.py coverage check (it expects a folder of test images)')
    if a.zip:
        a.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a.zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(a.out_dir.glob('*.txt')):
                zf.write(f, arcname=f.name)
        print(f'zip: {a.zip}')
        if not a.no_workstation:
            sys.path.insert(0, str(ROOT))
            try:
                from scripts.viz.import_submission_to_workstation import auto_import_submission
                auto_import_submission(str(a.zip), model_id=a.workstation_id, name=a.workstation_name)
            except Exception as e:  # the submission itself is complete; registration is local tooling
                print(f'WARNING: workstation registration failed ({type(e).__name__}: {e}); rerun scripts/viz/import_submission_to_workstation.py --zip {a.zip}')


if __name__ == '__main__':
    main()
