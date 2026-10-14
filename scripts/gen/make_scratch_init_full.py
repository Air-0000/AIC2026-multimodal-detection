"""Build a *fully clean* RDTTrack warm-start checkpoint:

    RGB   stream <- official DINOv3 ViT-L/16   (3ch, unchanged)
    IR    stream <- official DINOv3 ViT-S+/16  (patch_embed 3ch -> 2ch)
    Depth stream <- official DINOv3 ViT-S+/16  (patch_embed 3ch -> 1ch)

Nothing else is written: encoder / decoder / convs / aux_proj / sta /
prompt / dt_ort stay randomly initialised.

The 3ch -> C channel stem adaptation follows the convention already used in
`multimodal_backbones.py` (`PureIR`: mean(dim=1).repeat(1, C, 1, 1);
`TriModal`: cat(original, mean) when enlarging).

Loads are verified with strict=True against real DinoVisionTransformer
instances, so a silent key/shape mismatch cannot slip through.
"""
import os
import sys
import torch

AIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(AIC_ROOT, 'vendor', 'DEIMv2')
# nightify_aic lives in scripts/lib and is imported by engine.data.transforms
sys.path.insert(0, os.path.join(AIC_ROOT, 'scripts', 'lib'))
sys.path.insert(0, REPO)

from engine.backbone.dinov3.vision_transformer import DinoVisionTransformer  # noqa: E402

CK = os.path.join(REPO, 'ckpts', 'dinov3')
VITL = os.path.join(CK, 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
VITSP = os.path.join(CK, 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
OUT = os.path.join(CK, 'vitl16_rdttrack_scratch_full.pth')


def strip(ck):
    """Reduce any checkpoint layout to a plain stem-level state dict."""
    if isinstance(ck, dict):
        for k in ('model', 'ema', 'teacher', 'state_dict'):
            if k in ck and isinstance(ck[k], dict):
                return strip(ck[k])
    out = {}
    for k, v in ck.items():
        if k.startswith('backbone.dinov3.'):
            out[k[len('backbone.dinov3.'):]] = v
        elif isinstance(k, str) and not k.startswith(('backbone.', 'encoder.', 'decoder.')):
            out[k] = v
    return out


def adapt_stem(sd, channels):
    """Official checkpoints are always 3-channel; map them to `channels`."""
    import copy
    sd = {k: v.clone() for k, v in sd.items()}
    w = sd.get('patch_embed.proj.weight')
    if w is None or w.dim() != 4 or w.shape[1] == channels:
        return sd
    src = w.shape[1]
    mean_w = w.mean(dim=1, keepdim=True)
    if channels == src:
        pass
    elif channels < src:
        sd['patch_embed.proj.weight'] = mean_w.repeat(1, channels, 1, 1)
    else:  # enlarge: keep the real RGB filters, pad with the mean
        sd['patch_embed.proj.weight'] = torch.cat(
            [w, mean_w.repeat(1, channels - src, 1, 1)], dim=1)
    return sd


def verify(name, channels, sd, tag):
    """strict=True load into a real model instance -- catches every mismatch."""
    m = DinoVisionTransformer(name=name, in_chans=channels)
    missing, unexpected = m.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in m.parameters())
    ok = (not missing) and (not unexpected)
    print(f"  [{tag}] {name} in_chans={channels}: params={n/1e6:.2f}M  "
          f"missing={len(missing)} unexpected={len(unexpected)} -> "
          f"{'OK' if ok else 'FAIL'}")
    if missing:
        print('     missing:', missing[:6])
    if unexpected:
        print('     unexpected:', unexpected[:6])
    return ok


def main():
    print('== loading official checkpoints ==')
    sd_l = strip(torch.load(VITL, map_location='cpu'))
    sd_sp = strip(torch.load(VITSP, map_location='cpu'))
    print(f'  ViT-L   : {len(sd_l)} keys  '
          f'{sum(v.numel() for v in sd_l.values())/1e6:.2f}M')
    print(f'  ViT-S+  : {len(sd_sp)} keys  '
          f'{sum(v.numel() for v in sd_sp.values())/1e6:.2f}M')

    print('== strict verification ==')
    ok = True
    ok &= verify('dinov3_vitl16', 3, adapt_stem(sd_l, 3), 'rgb')
    ok &= verify('dinov3_vits16plus', 2, adapt_stem(sd_sp, 2), 'ir')
    ok &= verify('dinov3_vits16plus', 1, adapt_stem(sd_sp, 1), 'depth')
    if not ok:
        print('!! verification failed, refusing to write', OUT)
        return

    out = {}
    for k, v in adapt_stem(sd_l, 3).items():
        out[f'backbone.rgb_dinov3.{k}'] = v
    for k, v in adapt_stem(sd_sp, 2).items():
        out[f'backbone.ir_dinov3.{k}'] = v
    for k, v in adapt_stem(sd_sp, 1).items():
        out[f'backbone.depth_dinov3.{k}'] = v

    # load_tuning_state() expects the {'model': state_dict} container
    torch.save({'model': out}, OUT)
    tot = sum(v.numel() for v in out.values())
    print(f'== wrote {OUT} ==')
    print(f'  total keys={len(out)}  params={tot/1e6:.2f}M')
    for p in ('backbone.rgb_dinov3.', 'backbone.ir_dinov3.',
              'backbone.depth_dinov3.', 'encoder.', 'decoder.'):
        n = sum(v.numel() for k, v in out.items() if k.startswith(p))
        print(f'  {p:26s} {n/1e6:8.2f}M')


if __name__ == '__main__':
    main()
