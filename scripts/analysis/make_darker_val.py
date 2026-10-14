"""Generate darker-than-night005 RGB validation sets (night002/night001/night0005)
reusing the nightify physical model, but with stronger illumination attenuation +
higher gamma + weaker glow so the result is monotonically darker than night005
(mean brightness 0.0104). Filenames kept identical to orig to match annotations.
"""
import os
import shutil
import numpy as np
from PIL import Image, ImageFilter

VALID = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/valid/images"
OUT = "/home/constant/CodeBuddy/Dev/AIC/work_dir"


def rng_for(name):
    return np.random.default_rng(int.from_bytes(name.encode(), "little") & 0xFFFFFFFF)


def nightify(day, rng, ar, gr, glow_r):
    img = day.astype("float32") / 255.0
    daymax = img.max(2)
    H, W = img.shape[:2]
    gamma = rng.uniform(*gr)
    atten = rng.uniform(*ar)
    night = np.power(img, gamma) * atten
    cool = np.array([0.92, 0.98, 1.12])
    night = night * cool
    lit = (daymax > 0.70)[..., None] * img
    glow = np.asarray(
        Image.fromarray((lit * 255).clip(0, 255).astype("uint8"))
        .filter(ImageFilter.GaussianBlur(9))
    ).astype("float32") / 255.0
    glow = glow.mean(2, keepdims=True)
    warm = np.array([1.10, 1.00, 0.86])
    night = night + glow * rng.uniform(*glow_r) * warm
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(np.hypot(cx, cy), 1e-6)
    vig = 1.0 - rng.uniform(0.10, 0.35) * r
    night = night * vig[..., None]
    lum = night[:, :, 0] * 0.299 + night[:, :, 1] * 0.587 + night[:, :, 2] * 0.114
    sat = rng.uniform(0.88, 0.97)
    sat_loc = (sat * (0.55 + 0.6 * lum.clip(0, 1))).clip(0, 0.95)
    night = night * (1 - sat_loc)[..., None] + lum[..., None] * sat_loc[..., None]
    sig = night.clip(0, 1)
    read = rng.uniform(0.008, 0.015)
    k_shot = rng.uniform(0.001, 0.003)
    noise_std = np.sqrt(read ** 2 + k_shot * sig)
    night = night + rng.normal(0.0, noise_std)
    return (night.clip(0, 1) * 255).round().astype("uint8")


# target: mean brightness strictly below night005 (0.0104) and monotonically decreasing
LEVELS = {
    "night002": ([0.008, 0.025], [2.8, 3.5], (0.02, 0.05)),
    "night001": ([0.004, 0.014], [3.0, 3.7], (0.015, 0.04)),
    "night0005": ([0.002, 0.008], [3.2, 3.9], (0.01, 0.03)),
}

fs = sorted(f for f in os.listdir(VALID) if f.lower().endswith((".jpg", ".png")))
for lvl, (ar, gr, glow_r) in LEVELS.items():
    dst = os.path.join(OUT, "aic_valid_rgb_" + lvl)
    os.makedirs(dst, exist_ok=True)
    done = 0
    for f in fs:
        out = os.path.join(dst, f)
        if os.path.exists(out):
            done += 1
            continue
        day = np.asarray(Image.open(os.path.join(VALID, f)).convert("RGB"))
        arr = nightify(day, rng_for(os.path.splitext(f)[0]), ar, gr, glow_r)
        Image.fromarray(arr).save(out)
        done += 1
        if done % 50 == 0:
            print(f"{lvl} {done}/{len(fs)}", flush=True)
    print(f"made {lvl}: {done} images -> {dst}", flush=True)
