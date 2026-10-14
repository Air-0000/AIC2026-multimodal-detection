"""Realistic night-degradation protocol for the AIC val split.

WHY: the original dark05..dark50 protocol multiplies the visible channels by a
single global constant. That is *exactly* the transformation a per-image
instance normalisation is invariant to, so `inorm` (E5) scored dark05 0.490 vs
0.068 for the baseline and MEAN 0.5130 -- a +57% jump -- while being *worse* than
the baseline on real night (BDD100K night AP50 0.274 vs 0.284). The curve was
measurable, not meaningful.

Real night is not a global scale factor. This protocol adds the three things that
break global invariance:

   1) spatially NON-UNIFORM illumination -- a low-frequency field with dark
      vignetting plus a few bright "street-lamp" blobs (mean-normalised to 1),
      so brightness varies across the frame instead of being a constant;
   2) POISSON-GAUSSIAN sensor noise -- photon (shot) noise from the attenuated
      signal plus read noise, so low light is dominated by noise rather than
      being a clean scaled image;
   3) COLOUR SHIFT -- per-channel gains with a cool bias, as night white
      balance drifts;
   plus the original global attenuation as the level knob.

Deterministic: the RNG is seeded from (level, filename), so every model sees
byte-identical inputs and runs are reproducible.

Outputs mirror prep_night_robust.py so the existing eval flow works unchanged:
    data_aux/aic_valid_rgb_night{50,25,12,05}/          degraded images
    configs/dataset/aic_detection_night{50,25,12,05}.yml (RGB-only models)
    configs/deimv2/<stem>_val_night{50,25,12,05}.yml     (per model)
"""
import os
import zlib
import glob
import numpy as np
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
CFG = f"{ROOT}/vendor/DEIMv2/configs/deimv2"
DS = f"{ROOT}/vendor/DEIMv2/configs/dataset"
LEVELS = [0.50, 0.25, 0.12, 0.05]

# photon scale at full exposure and read-noise sigma: tuned so that dark05 is
# clearly noise-limited while dark50 still shows structure.
PHOTON_MAX = 70.0
READ_SIGMA = 0.012

VAL_IMG_LINE = f"    img_folder: {SRC}"

# (config, out_dir, tag, kind) -- mirrors prep_night_robust.py
MODELS = [
    ("deimv2_dinov3_l_aic.yml",               "./outputs/deimv2_dinov3_l_aic",               "rgb3ch",  "rgb"),
    ("deimv2_dinov3_l_aic_nightsim.yml",      "./outputs/deimv2_dinov3_l_aic_nightsim",      "nightsim", "rgb"),
    ("deimv2_dinov3_l_aic_nightsim_inorm.yml","./outputs/deimv2_dinov3_l_aic_nightsim_inorm","inorm",   "rgb"),
    ("deimv2_dinov3_l_aic_nightcut.yml",      "./outputs/deimv2_dinov3_l_aic_nightcut",      "nightcut", "rgb"),
    ("deimv2_dinov3_l_aic_rdttrack_heat.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat", "prompt",  "mm"),
    ("deimv2_dinov3_l_aic_rdtdiffuse.yml",    "./outputs/deimv2_dinov3_l_aic_rdtdiffuse",    "diffuse", "mm"),
    ("deimv2_dinov3_l_aic_rdtdiffuse_unfreeze.yml", "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze", "unfrz", "mm"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night", "nightaug", "mm"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log", "nightlog", "mm"),
    ("deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night.yml", "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night", "difnight", "mm"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf6.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf6", "uf6", "mm"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf12.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf12", "uf12", "mm"),
    # E9 = nightlog + per-image RGB illumination normalisation (merge the two winners)
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm", "nli", "mm"),
    # E10 = nightlog + REALISTIC night augmentation (lamps + Poisson-Gaussian noise + colour shift)
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight", "rnight", "mm"),
]


def _lowfreq_field(h, w, rng):
    """Smooth multiplicative illumination field, mean 1.

    Structured so the frame is NOT uniformly scaled:
      * low-frequency random variation (coarse grid, bicubic upsample);
      * a radial vignette (darker at the edges);
      * 1-3 bright "street lamp" blobs, which is the light that actually
        remains at deep night.
    Bounded to [0.15, 3.5] and renormalised to mean 1 so the level knob still
    controls overall exposure.
    """
    gy, gx = 4, 4
    coarse = rng.normal(0.0, 0.22, size=(gy, gx)).astype(np.float32)
    f = np.asarray(Image.fromarray(coarse, mode="F").resize((w, h), Image.BICUBIC),
                   dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    f = 1.0 + f - 0.45 * np.clip(r - 0.4, 0, None)     # smooth + vignette

    for _ in range(int(rng.integers(1, 4))):           # lamp blobs
        by = rng.uniform(0.0, h)
        bx = rng.uniform(0.0, w)
        rad = rng.uniform(0.10, 0.28) * max(h, w)
        blob = np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * rad ** 2)))
        f = f + rng.uniform(0.6, 2.2) * blob

    f = np.clip(f, 0.15, 3.5)
    return (f / max(f.mean(), 1e-6)).astype(np.float32)


def degrade(arr, level, seed):
    """arr: float32 HxWx3 in [0,1] (day). Returns the degraded image."""
    rng = np.random.default_rng(seed)
    h, w = arr.shape[:2]

    # 1) global attenuation (the original level knob)
    x = arr * level
    # 2) non-uniform illumination
    x = x * _lowfreq_field(h, w, rng)[..., None]
    # 3) colour shift (cool bias + random gains)
    gains = rng.uniform(0.85, 1.15, size=3).astype(np.float32) * np.array(
        [0.92, 0.98, 1.12], np.float32)
    x = x * gains[None, None, :]
    # 4) Poisson-Gaussian sensor noise
    x = np.clip(x, 0.0, None)
    x = rng.poisson(x * PHOTON_MAX).astype(np.float32) / PHOTON_MAX
    x = x + rng.normal(0.0, READ_SIGMA, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def _seed(level, name):
    return (zlib.crc32((name + "|%.3f" % level).encode()) & 0xFFFFFFFF)


def main():
    names = sorted(os.path.basename(p) for p in glob.glob(SRC + "/*"))
    print(f"val images: {len(names)}")

    for lv in LEVELS:
        tag = f"night{int(round(lv * 100)):02d}"
        out = f"{ROOT}/data_aux/aic_valid_rgb_{tag}"
        os.makedirs(out, exist_ok=True)
        for n in names:
            dst = os.path.join(out, n)
            if os.path.exists(dst):
                continue
            arr = np.asarray(Image.open(os.path.join(SRC, n)).convert("RGB"),
                             np.float32) / 255.0
            deg = degrade(arr, lv, _seed(lv, n))
            Image.fromarray((deg * 255.0 + 0.5).astype(np.uint8)).save(dst, quality=95)
        print(f"{tag}: {len(os.listdir(out))} images -> {out}")

        ds_src = open(os.path.join(DS, "aic_detection.yml")).read()
        assert VAL_IMG_LINE in ds_src, "aic_detection.yml val img_folder not found"
        ds_dst = f"aic_detection_{tag}.yml"
        open(os.path.join(DS, ds_dst), "w").write(
            ds_src.replace(VAL_IMG_LINE, f"    img_folder: {out}"))

        for cfg, out_dir, mtag, kind in MODELS:
            path = os.path.join(CFG, cfg)
            if not os.path.exists(path):
                continue
            text = open(path).read()
            head, val = text.split("val_dataloader:", 1)
            if kind == "mm":
                assert VAL_IMG_LINE in val, f"{cfg}: val img_folder not found"
                val = val.replace(VAL_IMG_LINE, f"    img_folder: {out}")
            else:
                assert "'../dataset/aic_detection.yml'" in head, f"{cfg}: include not found"
                head = head.replace("'../dataset/aic_detection.yml'",
                                    f"'../dataset/{ds_dst}'")
            head = head.replace(f"output_dir: {out_dir}", f"output_dir: {out_dir}__{tag}")
            dst_cfg = cfg.replace(".yml", f"_val_{tag}.yml")
            open(os.path.join(CFG, dst_cfg), "w").write(head + "val_dataloader:" + val)
        print(f"  -> configs for {len(MODELS)} models at level {lv}")


if __name__ == "__main__":
    main()
