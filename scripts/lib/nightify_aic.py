"""DNDT-style (Day-to-Night Domain Transforming) day->night simulation for AIC.

Inspired by Cui et al., "Debiased Teacher for Day-to-Night Domain Adaptive Object
Detection", ICCV 2025 -- we model the KEY day-night physical differences instead of
training a GAN:

    Stage 1  Illumination attenuation : night ambient irradiance is a small fraction
                                       of daytime; gamma models the camera's nonlinear
                                       response under low light.
    Stage 2  Color temperature shift  : cool blue ambient (moon/skyglow) + warm tint
                                       reserved for artificial lights.
    Stage 3  Headlight highlights      : bloom ONLY where the daytime image is ALREADY
                                       bright -> real lamps/headlights glow, never
                                       invent new lights (zero hallucination).
    Stage 4  Sensor noise              : physical photon shot noise + read noise
                                       (var = read^2 + k_shot * signal).

No training, no hallucinated lights -> content/labels stay valid.
Per-image randomized params give dataset variety. Writes <base>.png next to the
daytime frame, matching build_aic_nightcut_dataset.py.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFilter

AIC = "/home/constant/CodeBuddy/Dev/AIC"


def _rng_for(name):
    h = int.from_bytes(name.encode(), "little") & 0xFFFFFFFF
    return np.random.default_rng(h)


def nightify(day, rng, target_bright=None):
    """day: uint8 HxWx3 -> uint8 HxWx3 night (DNDT physical priors).

    target_bright: if set (0<tb<=1), the final signal is linearly rescaled so its
    MEAN equals target_bright, giving discrete ambient-level brackets (0.5/0.25/...)
    while keeping the same per-image physical priors (gamma/colour-temp/headlight/
    noise). If None, the original randomized ~0.11 ambient is kept.
    """
    img = day.astype("float32") / 255.0
    daymax = img.max(2)
    H, W = img.shape[:2]

    # --- Stage 1: Illumination attenuation (ambient light collapse) ---
    # Night ambient irradiance is a small fraction of daytime. gamma models the
    # camera's nonlinear response under low light; atten is that ambient fraction.
    gamma = rng.uniform(2.2, 3.4)
    atten = rng.uniform(0.05, 0.16)              # ambient fraction of daytime light
    night = np.power(img, gamma) * atten

    # --- Stage 2: Color temperature shift (cool blue ambient) ---
    cool = np.array([0.92, 0.98, 1.12])          # R-, B+  -> cool night ambient
    night = night * cool

    # --- Stage 3: Headlight / artificial-light highlights (zero hallucination) ---
    # Bloom only where the DAYTIME image is already bright, so existing lamps /
    # headlights / reflections glow. Warm-tinted like sodium/LED sources.
    lit = (daymax > 0.70)[..., None] * img
    glow = np.asarray(
        Image.fromarray((lit * 255).clip(0, 255).astype("uint8"))
        .filter(ImageFilter.GaussianBlur(9))
    ).astype("float32") / 255.0
    glow = glow.mean(2, keepdims=True)
    warm = np.array([1.10, 1.00, 0.86])          # warm headlight tint
    night = night + glow * rng.uniform(0.06, 0.16) * warm

    # radial vignette (optical falloff)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(np.hypot(cx, cy), 1e-6)
    vig = 1.0 - rng.uniform(0.10, 0.35) * r
    night = night * vig[..., None]

    # desaturate -> night mood, lit areas most
    lum = (night[:, :, 0] * 0.299 + night[:, :, 1] * 0.587
           + night[:, :, 2] * 0.114)[:, :, None]
    sat = rng.uniform(0.88, 0.97)
    sat_loc = (sat * (0.55 + 0.6 * lum.clip(0, 1))).clip(0, 0.95)
    night = night * (1 - sat_loc) + lum * sat_loc

    # --- ambient-level bracket: rescale mean to target_bright (keeps physical priors) ---
    if target_bright is not None and 0.0 < target_bright <= 1.0:
        m = night.mean()
        if m > 1e-6:
            night = np.clip(night * (target_bright / m), 0.0, 1.0)

    # --- Stage 4: Sensor noise (photon shot noise + read noise) ---
    # Physical model: var = read^2 + k_shot * signal. Bright lit pixels carry a
    # faint shot-grain; dark regions are governed by read noise (uniform fine
    # grain). Both kept mild to avoid visible "noise".
    sig = night.clip(0, 1)
    read = rng.uniform(0.008, 0.015)             # read-noise std (normalized)
    k_shot = rng.uniform(0.001, 0.003)           # shot-noise coefficient
    noise_std = np.sqrt(read ** 2 + k_shot * sig)
    night = night + rng.normal(0.0, noise_std)

    return (night.clip(0, 1) * 255).round().astype("uint8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(AIC, "work_dir/deimv2_dataset/train/images"))
    ap.add_argument("--dst", default=os.path.join(AIC, "data_aux/aic_nightify/images"))
    ap.add_argument("--bright", type=float, default=0.0,
                    help="target mean brightness for a discrete ambient bracket "
                         "(e.g. 0.5/0.25/0.12/0.05/0.01); 0 = keep randomized ~0.11")
    ap.add_argument("-n", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.n:
        files = files[:args.n]
    done = 0
    for f in files:
        out = os.path.join(args.dst, os.path.splitext(f)[0] + ".png")
        if os.path.exists(out):
            continue
        day = np.asarray(Image.open(os.path.join(args.src, f)).convert("RGB"))
        arr = nightify(day, _rng_for(os.path.splitext(f)[0]))
        Image.fromarray(arr).save(out)
        done += 1
        if done % 200 == 0:
            print(f"{done}/{len(files)}", flush=True)
    print(f"nightified {done} -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
