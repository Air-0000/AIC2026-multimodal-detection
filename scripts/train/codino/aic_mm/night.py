"""Night simulation of the midboost period (2026-09; reconstructed 2026-10-30, 原始 codino_fused 项目记录原 §五十八；本项目（AIC DEIMv2 仓 master 分支）见 docs/MODEL_RECORDS.md §五十) for the RGB channels
of the AIC multimodal input. Thermal and depth are left as they are (the thermal is decomposed from the original RGB in
LoadAICMultiModal, so it carries no day luminance — unlike the midboost 'heat' channel, which was IR − day grey).

Same model as scripts/prep/prep_nightxx_server.py:degrade_image: RGB × level × a low-frequency illumination field
(4x4 noise, vignette, 1–3 lamp blobs, normalised to mean 1) × a cool white-balance gain, then Poisson shot noise
(photon_max) and Gaussian read noise (read). Midboost trained with photon 70 / read 0.012 at levels drawn log-uniformly
from [0.005, 0.45]; its night01 … night0005 evaluation used photon 30 / read 0.020. Unlike the ×0.15 probe
(AICScaleRGB), at level 0.01 the RGB carries almost no information, which is where the midboost models leaned on the
thermal.

AICNightRGB is the deterministic test-time probe (seed from the image stem and the level), built to replace
AICScaleRGB at test pipeline index 1 through --cfg-options (it accepts and applies AICScaleRGB's `factor`).
AICRandomNightRGB is the training augmentation.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

WB_GAIN = np.array([0.92, 0.98, 1.12], np.float32)  # R, G, B (cool white balance)


def lowfreq_field(h: int, w: int, rng: np.random.Generator, work: int = 8) -> np.ndarray:
    """Illumination field HxW, mean 1, in [0.15, 3.5]/mean: drawn at 1/`work` resolution and resized (smooth anyway)."""
    sh, sw = max(4, h // work), max(4, w // work)
    coarse = rng.normal(0.0, 0.22, size=(4, 4)).astype(np.float32)
    f = cv2.resize(coarse, (sw, sh), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    cy, cx = sh / 2.0, sw / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    f = 1.0 + f - 0.45 * np.clip(r - 0.4, 0, None)
    for _ in range(int(rng.integers(1, 4))):
        by, bx = rng.uniform(0.0, sh), rng.uniform(0.0, sw)
        rad = rng.uniform(0.10, 0.28) * max(sh, sw)
        f = f + rng.uniform(0.6, 2.2) * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * rad ** 2)))
    f = np.clip(f, 0.15, 3.5)
    f = cv2.resize(f / max(float(f.mean()), 1e-6), (w, h), interpolation=cv2.INTER_LINEAR)
    return f.astype(np.float32)


def night_rgb(rgb: np.ndarray, level: float, rng: np.random.Generator, photon: float, read: float) -> np.ndarray:
    """rgb uint8 HxWx3 (R, G, B) → the same scene at `level` with sensor noise, uint8."""
    if level <= 0.0:
        return np.zeros_like(rgb)
    h, w = rgb.shape[:2]
    x = rgb.astype(np.float32) / 255.0 * level * lowfreq_field(h, w, rng)[..., None]
    x = np.clip(x * (rng.uniform(0.85, 1.15, size=3).astype(np.float32) * WB_GAIN), 0.0, None)
    if photon > 0:
        x = rng.poisson(x * photon).astype(np.float32) / photon
    if read > 0:
        x = x + rng.normal(0.0, read, size=x.shape).astype(np.float32)
    return np.clip(np.rint(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def probe_seed(img_path: str, level: float) -> int:
    return zlib.crc32(f'{Path(img_path).stem}|{level:.4f}'.encode()) & 0xFFFFFFFF


@TRANSFORMS.register_module()
class AICNightRGB(BaseTransform):
    """Test-time night probe on the RGB channels (0-2), deterministic per image. level 1 with no noise is the identity.
    `factor` (AICScaleRGB's argument) multiplies the RGB first, so the probe can replace AICScaleRGB in place."""

    def __init__(self, level: float = 1.0, photon: float = 0.0, read: float = 0.0, factor: float = 1.0) -> None:
        self.level, self.photon, self.read, self.factor = level, photon, read, factor

    def transform(self, results: dict) -> dict:
        if self.level == 1.0 and self.photon == 0 and self.read == 0 and self.factor == 1.0:
            return results
        img = results['img'].copy()
        rgb = img[..., :3]
        if self.factor != 1.0:
            rgb = np.clip(np.rint(rgb.astype(np.float32) * self.factor), 0, 255).astype(np.uint8)
        if self.level != 1.0 or self.photon or self.read:
            rng = np.random.default_rng(probe_seed(results.get('img_path', ''), self.level))
            rgb = night_rgb(rgb, self.level, rng, self.photon, self.read)
        img[..., :3] = rgb
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class AICRandomNightRGB(BaseTransform):
    """Training: with probability `prob` the RGB becomes a night scene at a level drawn log-uniformly from
    `level_range`, with shot noise `photon` and read noise `read` (midboost's training values by default)."""

    def __init__(self, prob: float = 0.5, level_range: tuple[float, float] = (0.005, 0.45), photon: float = 70.0,
                 read: float = 0.012) -> None:
        self.prob, self.level_range, self.photon, self.read = prob, level_range, photon, read

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.prob:
            lo, hi = self.level_range
            level = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
            img = results['img'].copy()
            img[..., :3] = night_rgb(img[..., :3], level, rng, self.photon, self.read)
            results['img'] = img
        return results
