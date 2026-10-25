#!/usr/bin/env python3
"""Export every decoder query (300 boxes x all class logits) for each validation image, optionally under shifts.

Inference only: no training, no TTA, no test data. Loads the checkpoint the way scripts/inference/predict.py does
(EMA weights, strict, training-only teacher / Co-ATSS keys left out) and reads every channel through the config's
own strict validation loader. For each image and each requested shift (dx, dy in network-input pixels) it runs one
FP32 forward and stores the decoder's final pred_logits and pred_boxes. A shift moves the normalized input tensor and
fills the vacated border with 0, then the predicted centres are moved back, as in the earlier V36 translation probes;
predictions are not clipped. Feed the result to scripts/analysis/analyze_error_modes.py.

Output (OUT must not exist): decoder_outputs.npz with image_ids, orig_sizes ([w, h] per image), input_hw and, per
shift arm 'dx{dx}_dy{dy}', <arm>_boxes (N x Q x 4, normalized cxcywh in original-image terms) and <arm>_logits
(N x Q x C), plus manifest.json (config, checkpoint, annotation and script SHA256, effective config, shifts).

Example (repo root, v1 best checkpoint on its held-out validation split, original and right-shift by 1 px):
  python scripts/diagnostics/export_query_outputs.py --config vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml \
    --checkpoint CKPT --shifts 0,0 1,0 0,1 --out work_dir/audits/query_outputs_v1_best
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT/'vendor/DEIMv2'
sys.path[:0] = [str(FW), str(ROOT/'scripts/lib'), str(ROOT/'scripts/inference')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def translate(x, dx, dy):
    """Shift a [B, C, H, W] tensor by (dx, dy) pixels, filling the vacated border with 0."""
    import torch
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., yd, xd] = x[..., ys, xs]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shifts', nargs='+', default=['0,0'], metavar='DX,DY', help='network-input pixels; first is the reference')
    p.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE', help='val dataset overrides, e.g. ann_file=/path')
    p.add_argument('--p3-quota', type=float, help='override DFINETransformer.p3_quota_ratio')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    shifts = [tuple(int(v) for v in s.split(',')) for s in a.shifts]
    if len(set(shifts)) != len(shifts):
        p.error('duplicate shifts')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config, checkpoint = a.config.resolve(), a.checkpoint.resolve()

    import numpy as np
    import torch
    import yaml
    import predict as P
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    c = P.load_config(config)
    if a.p3_quota is not None:
        c['DFINETransformer']['p3_quota_ratio'] = a.p3_quota
    ds = c['val_dataloader']['dataset']
    for kv in a.set:
        k, v = kv.split('=', 1)
        ds[k] = yaml.safe_load(v)
    c['val_dataloader'].update(total_batch_size=1, num_workers=2, shuffle=False, drop_last=False)
    c['val_dataloader'].pop('persistent_workers', None)
    cfg = P.build(c, out, 'export')
    model, _ = P.load_model(cfg, checkpoint, a.device)
    ann = Path(ds['ann_file'])
    ann = ann if ann.is_absolute() else (FW/ann).resolve()

    cwd = os.getcwd()
    os.chdir(FW)  # relative dataset paths in the config resolve against vendor/DEIMv2
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            loader = cfg.val_dataloader
        arms = [f'dx{dx}_dy{dy}' for dx, dy in shifts]
        boxes, logits = {k: [] for k in arms}, {k: [] for k in arms}
        image_ids, sizes, input_hw = [], [], None
        start = time.time()
        for i, (samples, targets) in enumerate(loader):
            if samples.shape[0] != 1:
                raise RuntimeError('batch size must be 1')
            source = samples.to(a.device)
            input_hw = tuple(source.shape[-2:])
            image_ids.append(int(targets[0]['image_id'].item()))
            sizes.append([int(v) for v in targets[0]['orig_size'].tolist()])
            for arm, (dx, dy) in zip(arms, shifts):
                variant = translate(source, dx, dy) if dx or dy else source
                with torch.inference_mode(), torch.autocast(device_type=a.device.split(':')[0], enabled=False):
                    o = model(variant)
                b = o['pred_boxes'][0].float().clone()
                b[:, 0] -= dx / input_hw[1]
                b[:, 1] -= dy / input_hw[0]
                lg = o['pred_logits'][0].float()
                if not (torch.isfinite(b).all() and torch.isfinite(lg).all()):
                    raise RuntimeError(f'non-finite output, image {image_ids[-1]} arm {arm}')
                boxes[arm].append(b.cpu().numpy())
                logits[arm].append(lg.cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f'{i + 1} images, {time.time() - start:.0f} s', flush=True)
    finally:
        os.chdir(cwd)

    arrays = dict(image_ids=np.asarray(image_ids, np.int64), orig_sizes=np.asarray(sizes, np.int64),
                  input_hw=np.asarray(input_hw, np.int64))
    for arm in arms:
        arrays[f'{arm}_boxes'] = np.stack(boxes[arm]).astype(np.float32)
        arrays[f'{arm}_logits'] = np.stack(logits[arm]).astype(np.float32)
    np.savez_compressed(out/'decoder_outputs.npz', **arrays)
    manifest = dict(config=str(config), config_sha256=sha(config), checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint), annotation=str(ann), annotation_sha256=sha(ann),
                    script_sha256=sha(__file__), effective_config=str(out/'export_effective.yml'), arms=arms,
                    shifts=[list(s) for s in shifts], images=len(image_ids), input_hw=list(input_hw),
                    p3_quota_ratio=c['DFINETransformer'].get('p3_quota_ratio'), dataset_overrides=a.set,
                    device=str(a.device), torch=torch.__version__, seconds=round(time.time() - start, 1),
                    decoder_outputs_sha256=sha(out/'decoder_outputs.npz'),
                    contract='pred_boxes are normalized cxcywh of the network input, shift-corrected and not clipped; '
                             'pred_logits are the final decoder logits (after the quality head), before sigmoid.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    print(f'{len(image_ids)} images x {len(arms)} arms -> {out}')


if __name__ == '__main__':
    main()
