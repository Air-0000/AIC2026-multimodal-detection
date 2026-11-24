"""Night-gated multimodal branch in front of the frozen pure-RGB Co-DINO Swin-L (2026-10-30, user's plan; small-model
version G0, 原始 codino_fused 项目记录原 §五十八；本项目（AIC DEIMv2 仓 master 分支）见 docs/MODEL_RECORDS.md §五十). Runs in the teammate's env (onedl-mmdetection 3.5.1); imported through
custom_imports ('aic_nightgate') after 'projects.CO-DETR.codetr', with scripts/train/codino/aic_mm on PYTHONPATH for
ir_signal and night (both import nothing else from aic_mm).

User: train pure RGB first, then fine-tune with a brightness gate that fires in training but not at inference, only to
meet the three-modality rule while keeping exactly the pure-RGB ability.

  input   LoadAICGateInputs: LoadImageFromFile's BGR image (the same decode as the pure-RGB base) + thermal T (the
          organisers' 0.15·RGB layer removed, ir_signal.decompose_thermal), thermal window W, raw depth D as the
          teammate reads it (16-bit / 20000 or 8-bit / 255; 0 = invalid). uint8 HxWx6, so mmdet's flip / resize / crop
          move all modalities together (6-channel resizes leave the first three channels bit-identical, checked
          2026-10-30 on the training and test scales).
  fusion  AICNightGateDataPreprocessor, per image before the usual BGR→RGB / normalise / pad:
              rgb' = rgb + G·S·net(Y, T, W, D, V, G)     (one luma shift added to R, G and B)
          G: image-level illumination gate from the mean luma of the whole image: 1 at or below lo, 0 at or above hi
          (user's choice 2026-10-30: lo 0.02, hi 0.05; the darkest Phase 3 test image is 0.065, all training images
          ≥ 0.266). S: the thermal window eroded and feathered (G0's soft window). V: depth valid. net: 5 conv layers,
          hidden 32, last layer zero, so training starts from the plain RGB.
          Eval mode with G = 0: the RGB tensor passes through untouched and the net is not called, so the detector
          input — and its output — are exactly the pure-RGB base's. Training: the shift is computed for every image
          (G = 0 adds an exact zero), so the loss always has a graph.
  detector AICFrozenCoDETR: every detector parameter frozen (requires_grad False; no EMA in the recipe, so they stay
          bit-identical); only data_preprocessor.fuse trains. The backbone stays in eval mode (Swin drop path 0.3 off,
          as at inference); the heads stay in training mode because DINO builds its denoising queries only then.
  training AICRandomNightBGR: midboost's night scene (night.night_rgb: illumination field, cool white balance, shot noise
          photon 70, read noise 0.012) at levels log-uniform in [0.003, 0.08] on the BGR channels; T, W, D untouched.
          The thermal is decomposed from the day RGB in the loader, so it carries no day luminance.
  probes  AICNightProbeBGR (deterministic per image, midboost's evaluation noise photon 30 / read 0.02) and
          AICZeroAux (thermal and depth channels zeroed) for the night01 / night005 IR-on / IR-off evaluation.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.transforms import BaseTransform, LoadImageFromFile
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS, TRANSFORMS

from ir_signal import decompose_thermal
from night import night_rgb, probe_seed

importlib.import_module('projects.CO-DETR.codetr')
CoDETR = MODELS.get('CoDETR')

LUMA_RGB = (0.299, 0.587, 0.114)
N_AUX = 3  # thermal, thermal window, raw depth


def _index(folder: Path) -> dict[str, Path]:
    return {Path(n).stem: folder / n for n in os.listdir(folder)}


def teammate_depth(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """The teammate's depth reading (tri_modal_coco_dataset 'auto'): 8-bit / 255, otherwise / 20000, clipped to [0, 1];
    0 where invalid, and all 0 below 0.1 % valid pixels. uint8 x255."""
    h, w = shape
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    is8 = d.dtype == np.uint8
    d = d.astype(np.float32)
    if d.ndim == 3:
        d = d.mean(2)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    n = np.clip(d / (255.0 if is8 else 20000.0), 0, 1)
    n[~(d > 0)] = 0
    if (d > 0).mean() < 0.001:
        n[:] = 0
    return np.rint(n * 255).astype(np.uint8)


@TRANSFORMS.register_module()
class LoadAICGateInputs(BaseTransform):
    """LoadImageFromFile (BGR, cv2 decode, as in the pure-RGB recipe) followed by thermal, thermal window and raw depth
    of the same stem from ir_dir / depth_dir: uint8 HxWx6. A missing IR or depth file raises."""

    def __init__(self, ir_dir: str, depth_dir: str) -> None:
        self.ir_dir, self.depth_dir = Path(ir_dir), Path(depth_dir)
        self.load = LoadImageFromFile()
        self._ir: dict[str, Path] | None = None
        self._depth: dict[str, Path] | None = None

    def transform(self, results: dict) -> dict:
        if self._ir is None:
            self._ir, self._depth = _index(self.ir_dir), _index(self.depth_dir)
        results = self.load(results)
        bgr = results['img']
        h, w = bgr.shape[:2]
        stem = Path(results['img_path']).stem
        if stem not in self._ir or stem not in self._depth:
            raise FileNotFoundError(f'no IR or depth for {stem} in {self.ir_dir} / {self.depth_dir}')
        ir = cv2.imread(str(self._ir[stem]), cv2.IMREAD_COLOR)
        if ir.shape[:2] != (h, w):
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
        t, win, a = decompose_thermal(bgr, ir)
        d = teammate_depth(self._depth[stem], (h, w))
        results['img'] = np.ascontiguousarray(np.concatenate([bgr, t[..., None], win[..., None], d[..., None]], -1))
        results['ir_blend'] = a
        return results


def _night_bgr(img: np.ndarray, level: float, rng: np.random.Generator, photon: float, read: float) -> np.ndarray:
    out = img.copy()
    out[..., 2::-1] = night_rgb(np.ascontiguousarray(img[..., 2::-1]), level, rng, photon, read)
    return out


@TRANSFORMS.register_module()
class AICRandomNightBGR(BaseTransform):
    """night.AICRandomNightRGB for an image whose channels 0-2 are B, G, R (LoadImageFromFile's order)."""

    def __init__(self, prob: float = 1.0, level_range: tuple[float, float] = (0.003, 0.08), photon: float = 70.0,
                 read: float = 0.012) -> None:
        self.prob, self.level_range, self.photon, self.read = prob, level_range, photon, read

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.prob:
            lo, hi = self.level_range
            level = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
            results['img'] = _night_bgr(results['img'], level, rng, self.photon, self.read)
        return results


@TRANSFORMS.register_module()
class AICNightProbeBGR(BaseTransform):
    """Test-time night probe on channels 0-2 (BGR), deterministic per image (night.probe_seed), midboost's evaluation
    noise by default."""

    def __init__(self, level: float, photon: float = 30.0, read: float = 0.02) -> None:
        self.level, self.photon, self.read = level, photon, read

    def transform(self, results: dict) -> dict:
        rng = np.random.default_rng(probe_seed(results.get('img_path', ''), self.level))
        results['img'] = _night_bgr(results['img'], self.level, rng, self.photon, self.read)
        return results


@TRANSFORMS.register_module()
class AICZeroAux(BaseTransform):
    """Probe: thermal, thermal window and depth channels set to zero (the 'IR / depth off' input)."""

    def transform(self, results: dict) -> dict:
        img = results['img'].copy()
        img[..., 3:] = 0
        results['img'] = img
        return results


def soft_window(w: torch.Tensor, erode_px: int, feather_px: int) -> torch.Tensor:
    """G0's soft window: the window (> 0.5) eroded by erode_px, then box-blurred over 2·feather_px + 1 pixels."""
    inner = (w > 0.5).to(w.dtype)
    if erode_px > 0:
        inner = -F.max_pool2d(-inner, 2 * erode_px + 1, stride=1, padding=erode_px)
    if feather_px > 0:
        inner = F.avg_pool2d(inner, 2 * feather_px + 1, stride=1, padding=feather_px, count_include_pad=False)
    return inner


def image_gate(y: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Scalar gate from the mean luma of one whole image: 1 at or below lo, 0 at or above hi, linear between."""
    return ((hi - y.mean()) / (hi - lo)).clamp(0, 1)


class GateFusionNet(nn.Module):
    """net(Y, T, W, D, V, G) → luma shift [1, H, W]; inputs centred as in G0 ((x − 0.5) / 0.25); last layer zero."""

    def __init__(self, hidden: int = 32, layers: int = 5) -> None:
        super().__init__()
        mods, c = [], 6
        for _ in range(layers - 1):
            mods += [nn.Conv2d(c, hidden, 3, padding=1), nn.ReLU(inplace=True)]
            c = hidden
        mods.append(nn.Conv2d(c, 1, 3, padding=1))
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x[None] - 0.5) / 0.25)[0]


@MODELS.register_module()
class AICNightGateDataPreprocessor(DetDataPreprocessor):
    """DetDataPreprocessor for the 6-channel input (B, G, R, T, W, D; uint8): the gated multimodal shift goes into the
    RGB, then only the three image channels continue through the parent (bgr_to_rgb, normalise, pad) exactly as in the
    pure-RGB recipe. Any other channel count raises."""

    def __init__(self, *args, gate: dict | None = None, hidden: int = 32, layers: int = 5, erode_px: int = 5,
                 feather_px: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        gate = {'lo': 0.02, 'hi': 0.05, **(gate or {})}
        self.lo, self.hi = float(gate['lo']), float(gate['hi'])
        self.erode_px, self.feather_px = erode_px, feather_px
        self.fuse = GateFusionNet(hidden, layers)
        self.last_gates: list[float] = []

    def fuse_one(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if x.dim() != 3 or x.shape[0] != 3 + N_AUX:
            raise ValueError(f'expected a {3 + N_AUX}-channel image (B, G, R, T, W, D), got {tuple(x.shape)}')
        bgr = x[:3]
        rgb = bgr.flip(0).float() / 255.0
        y = LUMA_RGB[0] * rgb[0] + LUMA_RGB[1] * rgb[1] + LUMA_RGB[2] * rgb[2]
        g = image_gate(y, self.lo, self.hi)
        self.last_gates.append(float(g))
        if not training and float(g) == 0.0:
            return bgr  # untouched: the parent sees the same uint8 tensor as in the pure-RGB model
        aux = x[3:].float() / 255.0
        t, w, d = aux[0:1], aux[1:2], aux[2:3]
        v = (d > 0).to(d.dtype)
        s = soft_window(w[None], self.erode_px, self.feather_px)[0]
        feats = torch.cat([y[None], t, w, d, v, g.expand_as(y)[None]], 0)
        dy = g * s * self.fuse(feats)
        return bgr.float() + 255.0 * dy  # same shift on B, G and R, in the 0-255 domain (exact when dy = 0)

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        self.last_gates = []
        data = dict(data, inputs=[self.fuse_one(x, training) for x in data['inputs']])
        return super().forward(data, training)


@MODELS.register_module()
class AICFrozenCoDETR(CoDETR):
    """CoDETR with every parameter frozen except those under `trainable`; the backbone always in eval mode."""

    def __init__(self, *args, trainable: tuple[str, ...] = ('data_preprocessor.fuse.',), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = tuple(trainable)
        if not any(n.startswith(self.trainable) for n, _ in self.named_parameters()):
            raise ValueError(f'no parameter matches trainable={self.trainable}')
        for n, p in self.named_parameters():
            p.requires_grad_(n.startswith(self.trainable))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        gates = getattr(self.data_preprocessor, 'last_gates', [])
        if gates:
            losses['gate_mean'] = batch_inputs.new_tensor(float(np.mean(gates)))
        return losses
