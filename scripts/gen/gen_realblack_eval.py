"""Near-black realistic-night eval levels to match the COMPETITION's real
night frames (the user's sample is ~near-black: RGB carries almost no object
structure, only local light + heavy sensor noise).

The existing night05 (day x0.05) is still recoverable RGB, so blacking aux
(bothblk) only costs +0.043 -- the robust RGB backbone alone scores 0.383. To
see whether the fusion actually READS IR/Depth at true black, push the visible
channel to ambient ~0 (x0.01 / x0.005) with stronger Poisson-Gaussian noise,
RGB only, IR/Depth untouched.

Reuses prep_night_real.degrade (overriding PHOTON_MAX/READ_SIGMA for heavier
noise) and regenerates the per-model val configs + ufall aux-ablation configs
for the new levels.
"""
import os, glob, numpy as np
from PIL import Image
import prep_night_real as P

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
CFG = f"{ROOT}/vendor/DEIMv2/configs/deimv2"
DS = f"{ROOT}/vendor/DEIMv2/configs/dataset"

# level -> tag ; ambient near zero, only local light (lamps) survives
# extended to the 3 darkest levels so the WHOLE sweep uses ONE consistent
# prep_night_real protocol (the same family as night50..night005), instead of
# the DNDT make_darker_val.py method that was previously used for 002/001/0005.
LEVELS = {0.01: "night01", 0.005: "night005", 0.002: "night002",
          0.001: "night001", 0.0005: "night0005"}
P.PHOTON_MAX = 30.0     # weaker signal -> heavier shot noise at low exposure
P.READ_SIGMA = 0.020    # stronger read noise, closer to a real night sensor

MODELS = [
    ("deimv2_dinov3_l_aic", "rgb3ch", False),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall", "ufall", True),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight", "rnight", True),
]


def gen_images(level, tag):
    out = f"{ROOT}/data_aux/aic_valid_rgb_{tag}"
    os.makedirs(out, exist_ok=True)
    n = 0
    for name in sorted(os.path.basename(p) for p in glob.glob(SRC + "/*")):
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            continue
        arr = np.asarray(Image.open(os.path.join(SRC, name)).convert("RGB"),
                         np.float32) / 255.0
        deg = P.degrade(arr, level, P._seed(level, name))
        Image.fromarray((deg * 255.0 + 0.5).astype(np.uint8)).save(dst, quality=95)
        n += 1
    print(f"{tag}: {len(os.listdir(out))} images ({n} new) -> {out}")


def mk_configs(tag):
    # 3ch dataset config (rgb3ch includes it)
    ds_src = open(f"{DS}/aic_detection_night05.yml").read()
    open(f"{DS}/aic_detection_{tag}.yml", "w").write(
        ds_src.replace("aic_valid_rgb_night05", "aic_valid_rgb_" + tag))

    for stem, _t, mm in MODELS:
        src = open(f"{CFG}/{stem}_val_night05.yml").read()
        if mm:
            txt = src.replace("aic_valid_rgb_night05", "aic_valid_rgb_" + tag)
        else:
            txt = src.replace("aic_detection_night05.yml",
                              f"aic_detection_{tag}.yml")
        txt = txt.replace("__night05", "__" + tag)
        open(f"{CFG}/{stem}_val_{tag}.yml", "w").write(txt)

    # ufall aux ablation (black IR / Depth / both) at the new level
    for v in ["irblk", "depblk", "bothblk"]:
        src = open(f"{CFG}/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_night05_ab{v}.yml").read()
        txt = src.replace("aic_valid_rgb_night05", "aic_valid_rgb_" + tag)
        txt = txt.replace("__night05_ab" + v, "__" + tag + "_ab" + v)
        open(f"{CFG}/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_val_{tag}_ab{v}.yml", "w").write(txt)
    print(f"configs for {tag} written")


if __name__ == "__main__":
    for level, tag in LEVELS.items():
        gen_images(level, tag)
        mk_configs(tag)
    print("OK")
