#!/usr/bin/env python
"""Visualize fusion-attention internals of HGNetv2MultiModal (D-FINE mm).

What it dumps per injection stage (CSSA = CSD CVPR 2023):
  * alpha  [C]  per-channel soft switch  1=trust RGB, 0=trust IR
  * sa     [H,W] CBAM-style spatial attention (where fusion emphasizes)
  * IR-shift sensitivity: sa map correlation + centroid shift when the IR
    input is rolled by --shift px (measures whether spatial attention is
    position-locked to IR content).

Usage:
  python scripts/viz/cssa_attention_viz.py [--ckpt path/to/model.pth]
Without --ckpt the untrained (zero-init) model is visualized as baseline;
re-run with the dcnmm checkpoint after training for the learned version.
"""

import argparse
import base64
import io
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'vendor', 'DEIMv2'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'lib'))

from engine.backbone.hgnetv2_mm import HGNetv2MultiModal  # noqa: E402


def load_gray(path, size):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return cv2.resize(im, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def load_rgb(path, size):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.resize(im, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def to_jet(x, size):
    x = np.clip(x, 0, 1)
    x = cv2.resize(x, size)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_JET)


def overlay(rgb, heat, alpha=0.45):
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def centroid(m):
    h, w = m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    s = m.sum() + 1e-9
    return (float((xs * m).sum() / s), float((ys * m).sum() / s))


class CSSAProbe:
    """Capture alpha / sa from every CSSA module via forward hooks."""

    def __init__(self, model):
        self.rec = {}
        self.handles = []
        for k, m in model.named_modules():
            if m.__class__.__name__ == 'CSSA':
                self.handles.append(m.register_forward_hook(self._make(k, m)))

    def _make(self, key, mod):
        def hook(module, inp, out):
            rgb, ir = inp
            with torch.no_grad():
                wr = mod.eca_rgb(rgb)
                wi = mod.eca_ir(ir)
                alpha_t = wr / (wr + wi + 1e-6)
                switched = alpha_t * rgb + (1.0 - alpha_t) * ir
                alpha = alpha_t.flatten().cpu().numpy()
                maxp = switched.amax(dim=1, keepdim=True)
                avgp = switched.mean(dim=1, keepdim=True)
                sa = mod.sigmoid(mod.sa_conv(torch.cat([maxp, avgp], 1)))
            self.rec[key] = {'alpha': alpha, 'sa': sa[0, 0].cpu().numpy()}
        return hook

    def clear(self):
        self.rec.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def build_sample(stem, size, night_gain=0.05, ir_shift=0):
    rgb = load_rgb(f'{ROOT}/work_dir/deimv2_dataset/train/images/{stem}.png', size)
    if rgb is None:
        rgb = load_rgb(f'{ROOT}/work_dir/deimv2_dataset/train/images/{stem}.jpg', size)
    heat = load_gray(f'{ROOT}/work_dir/deleak/images/{stem}.png', size)
    if heat is None:
        heat = load_gray(f'{ROOT}/work_dir/deleak/images/{stem}.jpg', size)
    ir = load_gray(f'{ROOT}/work_dir/dataset/train/infrared/images/{stem}.png', size)
    if ir is None:
        ir = load_gray(f'{ROOT}/work_dir/dataset/train/infrared/images/{stem}.jpg', size)
    assert rgb is not None and heat is not None and ir is not None, stem
    if ir_shift:
        heat = np.roll(heat, ir_shift, axis=1)
        ir = np.roll(ir, ir_shift, axis=1)
    rgb_v = torch.from_numpy(rgb.transpose(2, 0, 1))
    if night_gain < 1.0:
        rgb_v = rgb_v * night_gain
    x = torch.cat([rgb_v,
                   torch.from_numpy(heat)[None],
                   torch.from_numpy(ir)[None],
                   torch.zeros(1, *size)], dim=0)[None]
    return x, rgb, heat, ir


def b64(img_bgr):
    ok, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='trained model .pth (optional)')
    ap.add_argument('--stem', default='000008_033_00000001')
    ap.add_argument('--size', default='640,640')
    ap.add_argument('--shift', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(ROOT, 'work_dir', 'visualizations/fusion_viz'))
    args = ap.parse_args()

    size = tuple(int(v) for v in args.size.split(','))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    model = HGNetv2MultiModal(name='B5', ir_in_chans=2, inject_stages=[2, 3],
                              fusion='cssa', pretrained=False, ir_drop_prob=0.0)
    model.eval()
    tag = 'untrained-baseline'
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)
        bb = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        missing, unexpected = model.load_state_dict(bb, strict=False)
        print(f'loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}')
        tag = 'trained'
    probe = CSSAProbe(model)

    variants = [('day', 1.0, 0), ('night', 0.05, 0), ('night+IRshift', 0.05, args.shift)]
    rows = []
    report = [f'# CSSA attention probe — {tag} — stem {args.stem} @ {size[0]}px']

    with torch.no_grad():
        for vname, gain, shift in variants:
            x, rgb, heat, ir = build_sample(args.stem, size, gain, shift)
            probe.clear()
            model(x)
            for skey in sorted(probe.rec):
                a = probe.rec[skey]['alpha']
                sa = probe.rec[skey]['sa']
                rows.append((vname, skey, a, sa, rgb, heat, ir))
                report.append(f'{vname} {skey}: alpha mean={a.mean():.3f} '
                              f'sd={a.std():.3f} | sa mean={sa.mean():.3f} '
                              f'p99={np.percentile(sa, 99):.3f}')

    # IR-shift sensitivity: compare night vs night+IRshift sa maps
    sens = {}
    for skey in sorted(set(r[1] for r in rows)):
        sa0 = [r[3] for r in rows if r[0] == 'night' and r[1] == skey][0]
        sa1 = [r[3] for r in rows if r[0] == 'night+IRshift' and r[1] == skey][0]
        corr = float(np.corrcoef(sa0.flatten(), sa1.flatten())[0, 1])
        c0, c1 = centroid(sa0), centroid(sa1)
        sens[skey] = (corr, c1[0] - c0[0], c1[1] - c0[1])
        report.append(f'shift-test {skey}: sa corr={corr:.3f} '
                      f'centroid dx={c1[0]-c0[0]:.1f}px dy={c1[1]-c0[1]:.1f}px')

    # ---- render ----
    cards = []
    for vname, skey, a, sa, rgb, heat, ir in rows:
        strip = np.tile(np.clip(a, 0, 1), (48, 1))
        strip_c = cv2.applyColorMap((strip * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.putText(strip_c, f'alpha ch-switch  mean {a.mean():.2f}', (6, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sa_c = to_jet(sa, size)
        ov = overlay(rgb * 255, sa_c, 0.55)
        cards.append(f'''
<div class="card"><h3>{vname} · {skey}</h3>
<div class="row"><figure><img src="data:image/png;base64,{b64((rgb*255).astype(np.uint8))}"/><figcaption>RGB in</figcaption></figure>
<figure><img src="data:image/png;base64,{b64(sa_c)}"/><figcaption>sa spatial attn</figcaption></figure>
<figure><img src="data:image/png;base64,{b64(ov)}"/><figcaption>sa × RGB overlay</figcaption></figure></div>
<figure class="wide"><img src="data:image/png;base64,{b64(strip_c)}"/><figcaption>alpha: per-channel trust (red=RGB, blue=IR)</figcaption></figure>
</div>''')

    sens_html = ''.join(
        f'<li>{k}: corr={v[0]:.3f}, centroid moved ({v[1]:+.1f}, {v[2]:+.1f}) px</li>'
        for k, v in sens.items())
    rep_html = '<br>'.join(report)

    html = ('<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
            'body{font-family:sans-serif;max-width:1080px;margin:24px auto;padding:0 16px;color:#222}'
            'h1{font-size:20px}h2{font-size:16px;margin-top:28px}h3{font-size:14px;margin:14px 0 6px}'
            '.row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;text-align:center}'
            'figure img{max-width:320px;border:1px solid #ddd;border-radius:6px}'
            'figure.wide img{width:100%;max-width:1000px}'
            'figcaption{font-size:12px;color:#666;margin-top:3px}'
            '.card{border:1px solid #e3e3e3;border-radius:10px;padding:10px 14px;margin:14px 0}'
            '.note{background:#f6f2e8;border-left:4px solid #c8a13a;padding:10px 14px;border-radius:6px;font-size:13px}'
            'code{background:#f2f2f2;padding:1px 5px;border-radius:4px;font-size:12px}'
            '</style></head><body>'
            f'<h1>CSSA 融合内部可视化 — {tag}</h1>'
            '<div class="note"><b>怎么读这两张图</b><br>'
            '<b>alpha 条带</b>（逐通道软开关）：每个通道在 RGB(红) 和 IR(蓝) 之间做软切换；'
            '训练到位后预期是——day 样本整体偏红、night 样本 IR 通道偏蓝。<br>'
            '<b>sa 空间注意力</b>：每个位置 0~1 的重要性；预期热目标（鹿、人）处高、天空/路面低。'
            f'<br><br><b>IR 平移敏感性测试</b>（IR 输入整体挪 {args.shift}px）：<ul>{sens_html}</ul>'
            'sa 是从逐位置特征池化出来的，若 corr 接近 1 且质心跟着 IR 走，说明空间注意力<b>位置锁定在 IR 内容上</b>——'
            '错位时它会忠实地把权重投到错位的位置上；这正是交给下游检测损失去学的部分。</div>'
            f'<h2>原始读数</h2><div style="font-size:12px;line-height:1.7">{rep_html}</div>'
            + ''.join(cards) +
            '<h2>重跑方式</h2><p style="font-size:13px">dcnmm 训练完成后：<code>python scripts/viz/'
            'cssa_attention_viz.py --ckpt &lt;best.pth&gt;</code> 同脚本出 trained 版，逐图对比即知是否按预期。</p>'
            '</body></html>')

    out_html = os.path.join(args.out, f'cssa_attention_{tag}.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print('WROTE', out_html)


if __name__ == '__main__':
    main()
