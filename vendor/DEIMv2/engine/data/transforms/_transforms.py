"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()

import os
import sys
import numpy as np
for _p in ["/root/autodl-tmp/AIC/scripts/lib", "/root/autodl-tmp/AIC", "/home/constant/CodeBuddy/Dev/AIC", "/home/constant/CodeBuddy/Dev/AIC/scripts/lib"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from nightify_aic import nightify
except ImportError:
    nightify = None


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)


@register()
class RandomPhotometricDistort4ch(T.RandomPhotometricDistort):
    """Photometric (color) augmentation for 4-channel (RGB+heat) tensors.

    torchvision's color ops only accept 1- or 3-channel tensors, and its
    ``RandomPhotometricDistort`` also appends a ``channel_permutation`` of
    length 3 — which breaks on our 4-channel (RGB+heat) input. The heat
    channel is an infrared residual and must NOT be color-distorted anyway.

    We override ``transform`` to scope the brightness/contrast/saturation/hue
    ops to the first 3 (RGB) channels only and pass the heat channel through
    untouched. The optional channel permutation (RGB↔RGB swap) is skipped —
    it has negligible effect on detection and is unsafe on 4 channels.
    """

    def transform(self, inpt, params):
        if isinstance(inpt, torch.Tensor) and inpt.dim() == 3 and inpt.shape[0] == 4:
            from torchvision.transforms.v2 import functional as F  # local import
            rgb = inpt[:3]
            heat = inpt[3:]
            if params["brightness_factor"] is not None:
                rgb = F.adjust_brightness(rgb, params["brightness_factor"])
            if params["contrast_factor"] is not None and params["contrast_before"]:
                rgb = F.adjust_contrast(rgb, params["contrast_factor"])
            if params["saturation_factor"] is not None:
                rgb = F.adjust_saturation(rgb, params["saturation_factor"])
            if params["hue_factor"] is not None:
                rgb = F.adjust_hue(rgb, params["hue_factor"])
            if params["contrast_factor"] is not None and not params["contrast_before"]:
                rgb = F.adjust_contrast(rgb, params["contrast_factor"])
            return torch.cat([rgb, heat], dim=0)
        return super().transform(inpt, params)


@register()
class RandomPhotometricDistort5ch(T.RandomPhotometricDistort):
    """Photometric (color) augmentation for 5-channel (RGB+IR+Depth) tensors.

    Same idea as RandomPhotometricDistort4ch but for the tri-modal input
    [RGB(3), IR(1), Depth(1)]: brightness/contrast/saturation/hue are scoped to
    the first 3 (RGB) channels only; the IR and Depth channels (illumination-
    invariant geometric / thermal cues) are passed through untouched.
    """

    def _transform(self, inpt, params):
        return self.transform(inpt, params)

    def transform(self, inpt, params):
        if isinstance(inpt, torch.Tensor) and inpt.dim() == 3 and inpt.shape[0] >= 4:
            from torchvision.transforms.v2 import functional as F  # local import
            rgb = inpt[:3]
            aux = inpt[3:]  # IR + Depth
            if params["brightness_factor"] is not None:
                rgb = F.adjust_brightness(rgb, params["brightness_factor"])
            if params["contrast_factor"] is not None and params["contrast_before"]:
                rgb = F.adjust_contrast(rgb, params["contrast_factor"])
            if params["saturation_factor"] is not None:
                rgb = F.adjust_saturation(rgb, params["saturation_factor"])
            if params["hue_factor"] is not None:
                rgb = F.adjust_hue(rgb, params["hue_factor"])
            if params["contrast_factor"] is not None and not params["contrast_before"]:
                rgb = F.adjust_contrast(rgb, params["contrast_factor"])
            return torch.cat([rgb, aux], dim=0)
        if hasattr(super(), "_transform"):
            return super()._transform(inpt, params)
        return super().transform(inpt, params)


@register()
class RandomRGBDegrade4ch(T.Transform):
    """Simulate a night / low-light degraded visible image while keeping the IR
    channel intact.

    At train time with probability ``p`` the first 3 (RGB) channels are degraded
    (brightness down, contrast down, additive gaussian noise) but the 4th (IR)
    channel is passed through untouched. This forces the network to learn to rely
    on the illumination-invariant IR signal when the visible image is unreliable
    -- exactly the night test condition, with no real night data available.
    """
    def __init__(self, p=0.5, brightness=(0.15, 0.5), contrast=(0.3, 0.8), noise_std=(0.03, 0.12)):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def forward(self, *inputs):
        if torch.rand(1).item() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        img = inputs[0]
        if isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[0] == 4:
            rgb = img[:3]
            heat = img[3:]
            b = float(torch.empty(1).uniform_(*self.brightness).item())
            c = float(torch.empty(1).uniform_(*self.contrast).item())
            n = float(torch.empty(1).uniform_(*self.noise_std).item())
            rgb = rgb * b
            rgb = (rgb - 0.5) * c + 0.5
            rgb = rgb + torch.randn_like(rgb) * n
            rgb = rgb.clamp(0.0, 1.0)
            img = torch.cat([rgb, heat], dim=0)
            inputs = (img,) + inputs[1:]
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class RandomRGBDegrade5ch(T.Transform):
    """Simulate a night / low-light degraded visible image for the tri-modal
    5-channel input ``[RGB(3), IR(1), Depth(1)]``, leaving IR and Depth intact.

    AIC is a **daytime-only** training set, yet the deployment/test condition is
    **night**. With probability ``p`` the first 3 (RGB) channels are attenuated
    (brightness down, contrast down, additive gaussian noise) while the
    illumination-invariant thermal / geometric channels are passed through
    untouched. This forces the detector to fall back on IR/Depth when the
    visible camera fails, so night robustness is *trained* from day data --
    matching the night-robustness evaluation levels (0.50 dusk .. 0.05 deep
    night) used by ``prep_night_robust.py``.

    ``brightness`` samples the visible light level, so the default
    ``(0.08, 0.60)`` spans deep-night .. dusk. With ``log_brightness=True`` the
    light level is drawn log-uniformly between the two bounds, which gives the
    octaves of the evaluation curve equal mass -- the four levels
    dark05/12/25/50 are roughly log-spaced (x2 steps), so log-uniform squarely
    targets them instead of over-sampling the bright end.

    Implemented through the v2 ``transform``/``_get_params`` hooks (NOT
    ``forward``): the pipeline's ``Compose`` calls ``transform(sample)`` with the
    whole ``(image, target, dataset)`` tuple, so only the v2 hook receives the
    flattened 5-channel image tensor.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, p=0.5, brightness=(0.08, 0.60), contrast=(0.30, 0.90),
                 noise_std=(0.02, 0.12), log_brightness=False):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.log_brightness = log_brightness

    def _sample_brightness(self):
        lo, hi = float(self.brightness[0]), float(self.brightness[1])
        u = float(torch.empty(1).uniform_(0.0, 1.0).item())
        if self.log_brightness and lo > 0:
            return lo * (hi / lo) ** u          # log-uniform on [lo, hi]
        return lo + (hi - lo) * u

    def _get_params(self, flat_inputs):
        return {
            "apply": bool(torch.rand(1).item() < self.p),
            "b": self._sample_brightness(),
            "c": float(torch.empty(1).uniform_(*self.contrast).item()),
            "n": float(torch.empty(1).uniform_(*self.noise_std).item()),
        }

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        # NOTE: guard widened from `in (3, 4, 5)` to `>= 3`. The 6-channel deepir
        # layout (RGB3 + heat1 + raw_ir1 + depth1) silently skipped this op under
        # the old guard, so the night degradation never fired. The split below is
        # channel-count agnostic (rgb = first 3, aux = the rest).
        if isinstance(inpt, torch.Tensor) and inpt.dim() == 3 and inpt.shape[0] >= 3:
            rgb = inpt[:3]
            aux = inpt[3:]  # IR (+ Depth): illumination-invariant, keep untouched
            # Contrast/colour jitter FIRST (about mid-grey), THEN attenuate.
            # The other order ((rgb*b) then (x-0.5)*c+0.5) pulls a darkened image
            # back toward 0.5 -- at b=0.05, c=0.3 a mid-grey 0.5 ends up at 0.35
            # instead of 0.025, i.e. the night simulation is silently cancelled.
            rgb = (rgb - 0.5) * params["c"] + 0.5
            rgb = rgb * params["b"]
            rgb = rgb + torch.randn_like(rgb) * params["n"]
            rgb = rgb.clamp(0.0, 1.0)
            return torch.cat([rgb, aux], dim=0)
        return inpt


@register()
class RandomNight5ch(T.Transform):
    """Realistic night degradation for [RGB(3), IR(1), Depth(1)] at TRAIN time.

    ``RandomRGBDegrade5ch`` only applies a global attenuation (plus contrast and
    additive noise). The nightXX evaluation protocol (prep_night_real.py) showed
    that is not enough to be night-like, because real night also has

      * spatially NON-UNIFORM illumination -- street lamps and vignetting, so
        brightness varies across the frame instead of scaling as a constant;
      * Poisson-Gaussian SENSOR noise -- photon shot noise plus read noise, so
        deep night is noise-limited rather than a clean dim image;
      * a COLOUR SHIFT from night white balance.

    This transform reproduces that degradation, randomised per sample with a
    log-uniform exposure level, so training sees the whole nightXX range.
    IR/Depth are illumination-invariant and pass through untouched.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, p=0.5, level=(0.05, 0.80), photon_max=70.0,
                 read_sigma=0.012, log_level=True, lamps=(1, 4),
                 color_shift=True, bands=None):
        super().__init__()
        self.p = p
        self.level = level
        self.photon_max = photon_max
        self.read_sigma = read_sigma
        self.log_level = log_level
        self.lamps = lamps
        self.color_shift = color_shift
        # bands: list of [lo, hi, weight] sub-ranges; sampled by weight, then
        # log-uniform level within the chosen band. Enables level-weighted
        # oversampling (e.g. up-weight the dark/mid test band) + bright refill.
        self.bands = bands

    def _sample_level(self):
        if self.bands:
            weights = torch.tensor([float(b[2]) for b in self.bands])
            idx = int(torch.multinomial(weights, 1).item())
            lo, hi = float(self.bands[idx][0]), float(self.bands[idx][1])
        else:
            lo, hi = float(self.level[0]), float(self.level[1])
        lo, hi = float(lo), float(hi)
        u = float(torch.empty(1).uniform_(0.0, 1.0).item())
        if self.log_level and lo > 0:
            return lo * (hi / lo) ** u
        return lo + (hi - lo) * u

    def _illumination_field(self, h, w, n_lamps):
        """Low-frequency lamp/vignette field with mean 1, shape [1,1,h,w]."""
        hh, ww = max(h // 8, 4), max(w // 8, 4)
        yy = torch.linspace(0.0, 1.0, hh).view(hh, 1)
        xx = torch.linspace(0.0, 1.0, ww).view(1, ww)
        coarse = torch.randn(1, 1, 4, 4)
        f = torch.nn.functional.interpolate(coarse, size=(hh, ww),
                                            mode="bilinear", align_corners=False)[0, 0]
        f = 1.0 + 0.22 * f
        r = ((yy - 0.5) ** 2 + (xx - 0.5) ** 2).sqrt() * 1.4142
        f = f - 0.45 * (r - 0.4).clamp_min(0.0)            # vignette
        for _ in range(int(n_lamps)):
            by = float(torch.rand(1).item())
            bx = float(torch.rand(1).item())
            rad = float(torch.empty(1).uniform_(0.08, 0.25).item())
            amp = float(torch.empty(1).uniform_(0.6, 2.2).item())
            f = f + amp * torch.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) /
                                      (2.0 * rad * rad)))
        f = f.clamp(0.15, 3.5)
        f = f / f.mean()
        f = torch.nn.functional.interpolate(f.view(1, 1, hh, ww), size=(h, w),
                                            mode="bilinear", align_corners=False)
        return f[0]          # [1,h,w] so it broadcasts against the [3,h,w] RGB

    def _get_params(self, flat_inputs):
        return {
            "apply": bool(torch.rand(1).item() < self.p),
            "level": self._sample_level(),
            "lamps": int(torch.randint(self.lamps[0], self.lamps[1] + 1, (1,)).item()),
            "gain": (torch.empty(3).uniform_(0.85, 1.15) *
                     torch.tensor([0.92, 0.98, 1.12])).view(3, 1, 1),
        }

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        if not (isinstance(inpt, torch.Tensor) and inpt.dim() == 3
                and inpt.shape[0] >= 4):
            return inpt
        rgb = inpt[:3]
        aux = inpt[3:]
        h, w = rgb.shape[-2:]
        x = rgb * params["level"]
        x = x * self._illumination_field(h, w, params["lamps"])
        if self.color_shift:
            x = x * params["gain"]
        x = x.clamp_min(0.0)
        shot = (x / self.photon_max).sqrt()          # Poisson variance / mean
        x = x + torch.randn_like(x) * (shot + self.read_sigma)
        x = x.clamp(0.0, 1.0)
        if aux.shape[0] == 0:
            return x
        return torch.cat([x, aux], dim=0)


@register()
class RandomDNDTNightify(T.Transform):
    """Online DNDT (Day-to-Night Domain Transform) augmentation for the visible RGB.

    Lets a DAYTIME-only training set teach the detector to *invert* the physical
    night degradation itself -- gamma darkening, ambient attenuation, colour-temp
    shift, headlight bloom, vignette, desaturation, sensor noise -- instead of us
    hand-coding an inverse at test time. Wraps ``nightify_aic.nightify`` (our own
    physically-grounded forward model) with a FRESH random generator every sample,
    so the full DNDT factor manifold (each stage randomised independently) is seen.

    The overall exposure is sampled from ``bright_range`` (log-uniform) to cover the
    whole dusk..deep-night bracket set of the evaluation protocol. With
    ``curriculum=True`` the sampled exposure anneals from bright (easy) to dark
    (hard) across ``total_epochs`` -- the current epoch is read from the dataset's
    ``epoch`` attribute (set by the solver's ``set_epoch``), so NO training-loop
    change is needed.

    IR/Depth (channels >= 3) are illumination-invariant and pass through untouched,
    so this is safe for 3/4/5-channel inputs. Geometry is unchanged -> boxes valid.

    ``seeded=True`` draws the nightify noise generator's seed from torch's RNG, so a
    seeded / resumed run reproduces the same darkening; the default keeps the
    historical fresh-entropy behaviour.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, p=0.5, bright_range=(0.005, 0.5), log_bright=True,
                 curriculum=True, total_epochs=30, mid_bright=0.06, seeded=False, strict=False):
        super().__init__()
        # strict: refuse to build when nightify_aic is not importable, instead of silently passing images through.
        if strict and nightify is None:
            raise ImportError('RandomDNDTNightify(strict=True): nightify_aic not importable; put scripts/lib on PYTHONPATH')
        self.p = p
        self.bright_range = bright_range
        self.log_bright = log_bright
        self.curriculum = curriculum
        self.total_epochs = total_epochs
        self.mid_bright = mid_bright
        self.seeded = seeded

    @staticmethod
    def _log_uniform(lo, hi, u):
        lo = max(float(lo), 1e-4)
        return lo * (float(hi) / lo) ** u

    def _sample_bright(self, epoch):
        lo, hi = float(self.bright_range[0]), float(self.bright_range[1])
        if self.curriculum and epoch is not None and epoch >= 0 and self.total_epochs:
            t = min(max(epoch / float(self.total_epochs), 0.0), 1.0)
            # anneal exposure bracket from [mid, hi] (easy, dusk) to [lo, mid]
            # (hard, deep night)
            rlo = self._log_uniform(self.mid_bright, lo, t)
            rhi = self._log_uniform(hi, self.mid_bright, t)
        else:
            rlo, rhi = lo, hi
        u = float(torch.empty(1).uniform_(0.0, 1.0).item())
        return self._log_uniform(rlo, rhi, u)

    def _get_params(self, flat_inputs):
        epoch = None
        if len(flat_inputs) > 2:
            epoch = getattr(flat_inputs[2], "epoch", None)
        params = {
            "apply": bool(torch.rand(1).item() < self.p),
            "target_bright": self._sample_bright(epoch),
        }
        if self.seeded:
            params["seed"] = int(torch.randint(0, 2**31 - 1, (1,)).item())
        return params

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        if not (isinstance(inpt, torch.Tensor) and inpt.dim() == 3
                and inpt.shape[0] >= 3):
            return inpt
        rgb = inpt[:3].clamp(0.0, 1.0)
        aux = inpt[3:]
        arr = (rgb.mul(255).round().byte().cpu().numpy()
               .transpose(1, 2, 0))
        out = nightify(arr, np.random.default_rng(params.get("seed")), params["target_bright"])
        ten = torch.from_numpy(out).float().div(255.0).permute(2, 0, 1)
        if aux.shape[0] == 0:
            return ten
        return torch.cat([ten, aux], dim=0)


@register()
class IlluminationNorm(T.Transform):
    """Per-image illumination normalisation for low-light visible imagery.

    The detector is trained on **daytime-only** AIC (RGB luma mean ~105/255)
    with a fixed ImageNet ``Normalize``; a night frame is therefore a heavily
    shifted input distribution. This rescales the visible (first 3) channels so
    their luma statistics match a daytime target, optionally after a gamma
    correction -- a *zero-retrain* probe that tells us how much of the night
    drop is pure photometric drift rather than lost information.

        img = clamp(img, 0, 1) ** gamma              # <1 lifts shadows
        img = img * (target_mean / mean(luma))       # match day brightness
        img = (img - mean) / std   (if standardize)  # match unit variance

    ``IR/Depth`` pass through untouched. Applied AFTER the tensor is scaled to
    [0,1] and BEFORE ``Normalize``.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, target_mean=0.45, gamma=0.7, standardize=False, p=1.0):
        super().__init__()
        self.target_mean = target_mean
        self.gamma = gamma
        self.standardize = standardize
        self.p = p

    def _get_params(self, flat_inputs):
        return {"apply": bool(torch.rand(1).item() < self.p)}

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        # NOTE: guard widened from `in (3, 4, 5)` to `>= 3` -- see RandomRGBDegrade5ch.
        # The 6-channel deepir layout silently skipped this op under the old guard.
        if not (isinstance(inpt, torch.Tensor) and inpt.dim() == 3
                and inpt.shape[0] >= 3):
            return inpt
        rgb = inpt[:3].clamp(0.0, 1.0)
        aux = inpt[3:]
        if self.gamma != 1.0:
            rgb = rgb ** self.gamma
        m = rgb.mean()
        if float(m) > 1e-3:
            rgb = (rgb * (self.target_mean / m)).clamp(0.0, 1.0)
        if self.standardize:
            s = rgb.std()
            if float(s) > 1e-3:
                rgb = ((rgb - rgb.mean()) / s).clamp(-4.0, 4.0) * 0.25 + self.target_mean
        if aux.shape[0] == 0:
            return rgb
        return torch.cat([rgb, aux], dim=0)


@register()
class RandomTriStateIllumination(T.Transform):
    """Tri-State Illumination Consistency Learning for Multi-Modal Detection.

    Tackles the AIC2026 Day-to-Night domain shift by training on three distinct
    illumination states within each batch:
      1. Day state (pristine daylight RGB): preserves high-resolution daylight features;
      2. Night state (DNDT physics-grounded night degradation: ambient light attenuation,
         gamma darkening, photon shot noise, headlight glare): forces model to abandon
         unreliable visible features and rely on thermal (IR) and geometric (Depth) cues;
      3. Brightened state (night degradation + adaptive gamma lift & target luma matching):
         teaches model to discern true object contours amidst lifted sensor noise and glare.

    Operates on RGB (first 3 channels) only; IR & Depth (channels >= 3) are completely untouched.
    """
    _transformed_types = (torch.Tensor,)

    def __init__(self, p=0.6, bright_range=(0.01, 0.35), boost_ratio=0.5,
                 gamma=0.7, target_mean=0.45):
        super().__init__()
        self.p = p                     # Probability of non-day degradation
        self.bright_range = bright_range
        self.boost_ratio = boost_ratio # Fraction of degraded samples that get brightened (State 3)
        self.gamma = gamma
        self.target_mean = target_mean

    def _sample_bright(self):
        lo, hi = float(self.bright_range[0]), float(self.bright_range[1])
        u = float(torch.empty(1).uniform_(0.0, 1.0).item())
        lo = max(lo, 1e-4)
        return lo * (hi / lo) ** u

    def _get_params(self, flat_inputs):
        apply = bool(torch.rand(1).item() < self.p)
        is_boosted = bool(torch.rand(1).item() < self.boost_ratio)
        return {
            "apply": apply,
            "target_bright": self._sample_bright(),
            "is_boosted": is_boosted,
        }

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        if not (isinstance(inpt, torch.Tensor) and inpt.dim() == 3 and inpt.shape[0] >= 3):
            return inpt
        if nightify is None:
            return inpt

        rgb = inpt[:3].clamp(0.0, 1.0)
        aux = inpt[3:]
        arr = (rgb.mul(255).round().byte().cpu().numpy().transpose(1, 2, 0))

        # State 2: Physical Nightification (DNDT)
        out = nightify(arr, np.random.default_rng(), params["target_bright"])
        ten = torch.from_numpy(out).float().div(255.0).permute(2, 0, 1)

        # State 3: Adaptive Illumination Lifting
        if params["is_boosted"]:
            if self.gamma != 1.0:
                ten = ten ** self.gamma
            m = ten.mean()
            if float(m) > 1e-3:
                ten = (ten * (self.target_mean / m)).clamp(0.0, 1.0)

        if aux.shape[0] == 0:
            return ten
        return torch.cat([ten, aux], dim=0)


RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
# Geometric jitter for the multi-channel (RGB+IR+Depth) tensor. torchvision v2
# RandomAffine is purely spatial, so it treats all C channels identically --
# safe for 6ch input, unlike the colour ops which must be RGB-only.
RandomAffine = register()(T.RandomAffine)


@register()
class RandomMicroShift(T.Transform):
    """Sub-grid / micro-pixel spatial translation augmentation for multi-channel inputs.
    Applies small pixel-level translations (e.g. max_shift_px=16) to simulate camera
    jitter and break phase-dependent grid quantization on discrete feature maps.
    """
    def __init__(self, max_shift_px=16, p=0.5):
        super().__init__()
        self.max_shift_px = max_shift_px
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        is_single_seq = len(inputs) == 1 and isinstance(inputs[0], (tuple, list))
        unpacked = inputs[0] if is_single_seq else inputs

        if torch.rand(1).item() > self.p:
            return inputs[0] if is_single_seq else (unpacked if len(unpacked) > 1 else unpacked[0])

        img = unpacked[0]
        if hasattr(img, 'shape'):
            h, w = img.shape[-2:]
        else:
            h, w = F.get_size(img)

        trans = T.RandomAffine(degrees=0, translate=(float(self.max_shift_px)/w, float(self.max_shift_px)/h))
        res = trans(*unpacked)
        if is_single_seq:
            return tuple(res) if isinstance(inputs[0], tuple) else list(res)
        return res


Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
# ==================== Copy-Paste (SAM mask library) ====================
# Appended 2026-09-28: sqrt-tempered online instance pasting for tail classes.
# Mask library is built offline by scripts/lib/build_sam_masklib.py (SAM ViT-B
# box-prompt); this op pastes library instances into training images with
# random scale/flip/photometric jitter and appends their GT boxes/labels.
# MUST be placed FIRST in the train transforms pipeline (original pixel
# coords, before ZoomOut/Crop/Resize). PIL image in / PIL image out.

import json as _json
import os as _os
from collections import Counter as _Counter

import numpy as _np
import PIL.Image as _PILImage


@register()
class RandomCopyPaste(T.Transform):
    """Online Copy-Paste from a SAM mask library (tail-class oversampling).

    Instance sampling weight w_c = cnt_c ** -sqrt_pow (sqrt_pow=0.5 -> sqrt
    tempering, LVIS repeat-factor style). Labels are raw category_id
    (this dataset uses remap_mscoco_category=False).
    """

    def __init__(self, mask_lib: str = '', prob: float = 0.5,
                 max_paste: int = 3, scale_range=(0.5, 1.3),
                 iou_thresh: float = 0.3, sqrt_pow: float = 0.5,
                 min_inst_px: int = 24, max_frac: float = 0.4,
                 brightness: float = 0.15, contrast: float = 0.15,
                 trials: int = 10, min_score: float = 0.7,
                 target_range=(64.0, 160.0), max_upscale: float = 6.0,
                 ann_file: str = '', size_mode: str = 'fixed',
                 size_jitter=(0.8, 1.25), no_overlap: bool = True,
                 min_scale: float = 0.25) -> None:
        super().__init__()
        with open(mask_lib) as f:
            index = _json.load(f)
        index = [r for r in index if r.get('score', 1.0) >= min_score]
        self.inst = index
        self._lib_dir = _os.path.dirname(_os.path.abspath(mask_lib))
        cnt = _Counter(r['cat_id'] for r in index)
        self.w = _np.array([cnt[r['cat_id']] ** -sqrt_pow for r in index],
                           dtype=_np.float64)
        self.w /= self.w.sum()
        self.prob = prob
        self.max_paste = max_paste
        self.scale_range = tuple(scale_range)
        self.target_range = tuple(target_range)
        self.max_upscale = max_upscale
        self.size_mode = size_mode
        self.size_jitter = tuple(size_jitter)
        self.no_overlap = no_overlap
        self.min_scale = min_scale
        self._size_stats = {}
        if size_mode == 'class' and ann_file:
            # 每类 GT 框最长边的经验分布: 贴入尺寸跟随该类真实大小,
            # 而不是固定 64-160px(uav 被拉到 160px 会糊成马赛克)
            with open(ann_file) as f:
                A = _json.load(f)
            by_cat = {}
            for a in A['annotations']:
                x, y, w, h = a['bbox'][:4]
                by_cat.setdefault(a['category_id'], []).append(max(w, h))
            self._size_stats = {k: _np.asarray(v, dtype=_np.float64)
                                for k, v in by_cat.items() if len(v)}
        self.iou_thresh = iou_thresh
        self.min_inst_px = min_inst_px
        self.max_frac = max_frac
        self.brightness = brightness
        self.contrast = contrast
        self.trials = trials
        self._cache = {}
        print(f'[RandomCopyPaste] lib={self._lib_dir} n={len(index)} '
              f'classes={len(cnt)} prob={prob} sqrt_pow={sqrt_pow} '
              f'target_range={self.target_range} max_upscale={self.max_upscale} '
              f'size_mode={self.size_mode} stat_classes={len(self._size_stats)} '
              f'no_overlap={self.no_overlap}')

    # -- instance loading with a small LRU-ish cache ----------------------
    def _load_inst(self, rec):
        import cv2 as _cv2
        key = rec['id']
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            rgb = _cv2.imread(_os.path.join(self._lib_dir, f"{key}_rgb.png"))
            m = _cv2.imread(_os.path.join(self._lib_dir, f"{key}_m.png"), 0)
            if rgb is None or m is None:
                self._cache[key] = (None, None)
            else:
                self._cache[key] = (_cv2.cvtColor(rgb, _cv2.COLOR_BGR2RGB), m)
        return self._cache[key]

    def _sample_target(self, cat_id):
        """目标最长边(px): class 模式下从该类 GT 尺寸经验分布抽样."""
        if self.size_mode == 'class':
            pool = self._size_stats.get(cat_id)
            if pool is not None and len(pool):
                j = _np.random.uniform(*self.size_jitter)
                return float(_np.random.choice(pool) * j)
        return float(_np.random.uniform(*self.target_range))

    def forward(self, *inputs):
        # 容器以 transform(sample) 调用, sample=(img, target[, dataset]) 元组,
        # 此时 *inputs 是单元素嵌套元组, 须先解包(与其他算子的约定一致);
        # ep0-1 策略期 CP 被跳过, 该 bug 直到 ep2 才暴露
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])
        if len(inputs) < 2:
            return inputs
        img, target = inputs[0], inputs[1]
        if 'boxes' not in target or len(self.inst) == 0:
            return inputs
        if torch.rand(1).item() >= self.prob:
            return inputs

        W, H = img.size  # PIL
        boxes = target['boxes'].detach().cpu().numpy().astype(_np.float64)
        labels = target['labels'].tolist()
        pasted = 0
        self.last_pasted = []  # demo/调试用: 记录本次实际贴入的实例记录
        k = int(torch.randint(1, self.max_paste + 1, (1,)).item())
        for _ in range(k):
            for _try in range(self.trials):
                rec = self.inst[int(_np.random.choice(len(self.inst), p=self.w))]
                rgb, mask = self._load_inst(rec)
                if rgb is None:
                    continue
                ih, iw = mask.shape
                # 目标尺寸: class 模式=从该类 GT 尺寸经验分布采样(×jitter);
                # 红线: 只缩不放 —— 源实例小于目标尺寸就直接换一个更大的源,
                # 上采样会把 SAM mask 贴成马赛克(用户明确禁止)
                side = max(ih, iw)
                tgt = self._sample_target(rec['cat_id'])
                need = tgt / max(side, 1)
                if self.size_mode == 'class':
                    # 只缩不放(保原画质), 也不能缩过头(<min_scale 缩成色块)
                    if need > 1.0 + 1e-6 or need < self.min_scale:
                        continue
                    s = need
                else:
                    if need > self.max_upscale:
                        continue
                    s = min(max(need, 1.0 / self.max_upscale), self.max_upscale)
                nh, nw = int(ih * s), int(iw * s)
                if nh < self.min_inst_px or nw < self.min_inst_px:
                    continue
                if nh > self.max_frac * H or nw > self.max_frac * W:
                    continue
                if _np.random.rand() < 0.5:  # hflip
                    rgb, mask = rgb[:, ::-1], mask[:, ::-1]
                # resize instance to the scaled size (keep rgb/mask consistent)
                if (nh, nw) != mask.shape:
                    import cv2 as _cv2
                    rgb = _cv2.resize(rgb, (nw, nh), interpolation=_cv2.INTER_LINEAR)
                    mask = _cv2.resize(mask, (nw, nh), interpolation=_cv2.INTER_NEAREST)
                if self.brightness > 0:  # photometric jitter on the crop
                    b = 1.0 + _np.random.uniform(-self.brightness, self.brightness)
                    c = 1.0 + _np.random.uniform(-self.contrast, self.contrast)
                    f = rgb.astype(_np.float32)
                    f = (f - f.mean()) * c + f.mean()
                    rgb = _np.clip(f * b, 0, 255).astype(_np.uint8)
                x0 = int(_np.random.randint(0, max(1, W - nw)))
                y0 = int(_np.random.randint(0, max(1, H - nh)))
                nb = [float(x0), float(y0), float(x0 + nw), float(y0 + nh)]
                if len(boxes):
                    if self.no_overlap:
                        # 框之间零相交(GT + 已贴入都算): 叠在一起没法看
                        if (_inter_area(boxes, nb) > 1.0).any():
                            continue
                    elif (_iou_xyxy(boxes, nb) > self.iou_thresh).any():
                        continue
                arr = _np.array(img)  # HWC uint8 RGB
                if arr.ndim == 2:  # grayscale source image
                    arr = _np.stack([arr] * 3, axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                if nh != mask.shape[0] or nw != mask.shape[1] or rgb.shape[:2] != (nh, nw):
                    continue  # consistency guard
                m3 = mask[:, :, None].astype(_np.float32) / 255.0
                region = arr[y0:y0 + nh, x0:x0 + nw].astype(_np.float32)
                arr[y0:y0 + nh, x0:x0 + nw] = (
                    rgb * m3 + region * (1 - m3)).astype(_np.uint8)
                img = _PILImage.fromarray(arr)
                boxes = _np.vstack([boxes, _np.array(nb, dtype=_np.float64)])
                labels.append(rec['cat_id'])
                pasted += 1
                self.last_pasted.append(rec)
                break
        if pasted:
            from torchvision.tv_tensors import BoundingBoxFormat as _BF
            from torchvision.tv_tensors import BoundingBoxes as _BB
            target['boxes'] = _BB(
                torch.as_tensor(boxes, dtype=torch.float32),
                format=_BF.XYXY, canvas_size=(H, W))
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
        # ⚠ 必须返回替换后的 img: inputs 元组仍持有旧 PIL 对象,
        #   return inputs 会把像素改动整体丢弃(只留 target 的框), 贴图等于没贴
        out = list(inputs)
        out[0] = img
        return tuple(out)


def _inter_area(a, b):
    """a: (N,4) xyxy, b: (4,) xyxy -> (N,) 相交面积(px^2)."""
    ix1 = _np.maximum(a[:, 0], b[0])
    iy1 = _np.maximum(a[:, 1], b[1])
    ix2 = _np.minimum(a[:, 2], b[2])
    iy1b = _np.minimum(a[:, 3], b[3])
    return _np.clip(ix2 - ix1, 0, None) * _np.clip(iy1b - iy1, 0, None)


def _iou_xyxy(a, b):
    """a: (N,4) xyxy, b: (4,) xyxy -> (N,) iou."""
    ix1 = _np.maximum(a[:, 0], b[0])
    iy1 = _np.maximum(a[:, 1], b[1])
    ix2 = _np.minimum(a[:, 2], b[2])
    iy2 = _np.minimum(a[:, 3], b[3])
    inter = _np.clip(ix2 - ix1, 0, None) * _np.clip(iy2 - iy1, 0, None)
    area_a = _np.clip(a[:, 2] - a[:, 0], 0, None) * _np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / _np.clip(area_a + area_b - inter, 1e-9, None)


@register()
class RandomModalityFailure(T.Transform):
    """
    Stochastically simulates individual sensor outages and modality dropouts:
    For N-channel multi-modal tensors (N >= 4, e.g. 5ch or 6ch):
      Channels:
        0..2: RGB (Visible)
        3..4: IR (Aligned + LBG/Raw)
        5:    Depth
    - RGB failure (pitch black / camera failure):
      with probability p_rgb (default 0.15), inpt[:3] is set to 0.0
    - IR failure (thermal sensor saturation / disconnection):
      with probability p_ir (default 0.10), inpt[3:5] is set to 0.5 (neutral mean before Normalize)
    - Depth failure (blind / out-of-range / NaN):
      with probability p_depth (default 0.15), inpt[5:] is set to 0.0
    Guarantees at least one primary sensing modality (RGB or IR) remains alive.
    """
    _transformed_types = (torch.Tensor,)

    def __init__(self, p_rgb=0.15, p_ir=0.10, p_depth=0.15):
        super().__init__()
        self.p_rgb = p_rgb
        self.p_ir = p_ir
        self.p_depth = p_depth

    def _get_params(self, flat_inputs):
        drop_rgb = bool(torch.rand(1).item() < self.p_rgb)
        drop_ir = bool(torch.rand(1).item() < self.p_ir)
        drop_depth = bool(torch.rand(1).item() < self.p_depth)

        # Safety guard: never drop both RGB and IR simultaneously
        if drop_rgb and drop_ir:
            if torch.rand(1).item() < 0.5:
                drop_rgb = False
            else:
                drop_ir = False

        return {
            "drop_rgb": drop_rgb,
            "drop_ir": drop_ir,
            "drop_depth": drop_depth,
        }

    def transform(self, inpt, params):
        if not (isinstance(inpt, torch.Tensor) and inpt.dim() == 3 and inpt.shape[0] >= 4):
            return inpt
        
        out = inpt.clone()
        if params["drop_rgb"]:
            out[:3] = 0.0
        if params["drop_ir"]:
            if out.shape[0] >= 5:
                out[3:5] = 0.5
            else:
                out[3:4] = 0.5
        if params["drop_depth"] and out.shape[0] >= 6:
            out[5:] = 0.0

        return out
