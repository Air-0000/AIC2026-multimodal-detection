"""Frozen s0918 detector with the original pixel fusion plus a gated residual.
Nine channels: original fused BGR, thermal/window/depth, original BGR.
Normal-light inference returns the fused uint8 image without arithmetic.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from mmdet.registry import MODELS, TRANSFORMS
from aic_nightgate import (LoadAICGateInputs, AICNightGateDataPreprocessor,
                          AICRandomNightBGR, LUMA_RGB, image_gate, soft_window, _night_bgr)

@TRANSFORMS.register_module()
class Load744GateInputs(LoadAICGateInputs):
    def transform(self, results):
        results = super().transform(results)
        six = results['img']
        rgb = np.asarray(Image.open(results['img_path']).convert('RGB'), dtype=np.float32) / 255.0
        y = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
        stem = __import__('pathlib').Path(results['img_path']).stem
        ir = np.asarray(Image.open(self._ir[stem]).convert('L'), dtype=np.float32) / 255.
        raw = cv2.imread(str(self._depth[stem]), cv2.IMREAD_UNCHANGED)
        if ir.shape != y.shape or raw.shape[:2] != y.shape:
            raise ValueError('744 requires original aligned image sizes')
        divisor = 255. if raw.dtype == np.uint8 else 20000.
        raw = raw.astype(np.float32)
        if raw.ndim == 3:
            raw = raw.mean(2)
        valid = np.isfinite(raw) & (raw > 0)
        depth = np.zeros_like(raw)
        depth[valid] = np.clip(raw[valid] / divisor, 0, 1)
        if valid.mean() < .001:
            valid[:] = False
            depth[:] = 0
        fused = rgb.copy()
        if np.isfinite(ir).all() and np.any(ir > 0):
            fused[..., 0] += .15 * (ir - y)
        fused[..., 1] += .10 * valid.astype(np.float32) * (depth - y)
        fused = np.rint(np.clip(fused, 0, 1) * 255).astype(np.uint8)[..., ::-1]
        results['img'] = np.ascontiguousarray(np.concatenate([fused, six[..., 3:], six[..., :3]], -1))
        return results

@TRANSFORMS.register_module()
class RandomNight744(AICRandomNightBGR):
    def transform(self, results):
        if np.random.rand() < self.prob:
            lo, hi = self.level_range
            level = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            seed = np.random.randint(0, 2**31-1)
            x = results['img'].copy()
            # Corrupt only visible channels; preserve the original IR/depth contribution.
            original = x[..., 6:9].copy()
            dark = _night_bgr(original, level, np.random.default_rng(seed), self.photon, self.read)
            delta = dark.astype(np.float32) - original.astype(np.float32)
            x[..., :3] = np.rint(np.clip(x[..., :3].astype(np.float32) + delta, 0, 255)).astype(np.uint8)
            x[..., 6:9] = dark
            results['img'] = x
        return results

@MODELS.register_module()
class DataPreprocessor744(AICNightGateDataPreprocessor):
    def fuse_one(self, x, training):
        if x.ndim != 3 or x.shape[0] != 9:
            raise ValueError(f'expected 9 channels, got {tuple(x.shape)}')
        baseline = x[:3]
        rgb = x[6:9].flip(0).float() / 255.
        y = sum(w * rgb[i] for i, w in enumerate(LUMA_RGB))
        g = image_gate(y, self.lo, self.hi)
        self.last_gates.append(float(g))
        if not training and float(g) == 0:
            return baseline
        t, w, d = (x[i:i+1].float()/255. for i in (3,4,5))
        s = soft_window(w[None], self.erode_px, self.feather_px)[0]
        feats = torch.cat([y[None],t,w,d,(d>0).to(d.dtype),g.expand_as(y)[None]],0)
        return baseline.float() + 255. * g * s * self.fuse(feats)


from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class Finite744GradientHook(Hook):
    def before_train(self, runner):
        self.handles = []
        for name, p in runner.model.named_parameters():
            if p.requires_grad:
                def check(grad, name=name):
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('non-finite branch gradient: ' + name)
                    return grad
                self.handles.append(p.register_hook(check))

    def after_train(self, runner):
        for handle in self.handles:
            handle.remove()
