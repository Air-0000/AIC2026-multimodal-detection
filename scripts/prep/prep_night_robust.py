"""Night-scenario robustness protocol (training stays AIC-only).

AIC is all daytime (RGB luma min 105, 0% dark frames) and AIC must remain the
only training set, so night capability cannot be trained -- it has to be
*measured*. This builds a controlled low-light degradation of the AIC val split:
only the RGB channels are attenuated, IR and depth are untouched. That isolates
exactly the question "when the visible camera stops working, does the model
still detect because of the other modalities?".

    level 1.00 = original day
    level 0.50 = dusk
    level 0.25 = twilight
    level 0.12 = night
    level 0.05 = deep night

Two config shapes have to be handled:
  * multimodal models define the val dataset inline (CocoDetectionMultiModal)
    -> patch `img_folder` in the val block.
  * the RGB-only model inherits its val dataset from ../dataset/aic_detection.yml
    -> emit a darkened copy of that dataset file and re-point __include__.
"""
import glob
import os
import numpy as np
from PIL import Image

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
SRC = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
CFG = f"{ROOT}/vendor/DEIMv2/configs/deimv2"
DS = f"{ROOT}/vendor/DEIMv2/configs/dataset"

LEVELS = [0.50, 0.25, 0.12, 0.05]
VAL_IMG_LINE = f"    img_folder: {SRC}"

# (config, output_dir, tag, kind)
MODELS = [
    ("deimv2_dinov3_l_aic.yml",               "./outputs/deimv2_dinov3_l_aic",               "rgb3ch",  "rgb"),
    ("deimv2_dinov3_l_aic_rdttrack_heat.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat", "prompt",  "mm"),
    ("deimv2_dinov3_l_aic_rdtdiffuse.yml",    "./outputs/deimv2_dinov3_l_aic_rdtdiffuse",    "diffuse", "mm"),
    # stage-2 backbone unfreeze (in-domain 0.432 -> 0.481): does unfreezing the
    # RGB backbone help or hurt night robustness?
    ("deimv2_dinov3_l_aic_rdtdiffuse_unfreeze.yml", "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze", "unfrz", "mm"),
    # day->night trained (RandomRGBDegrade5ch): the actual night-robust model
    ("deimv2_dinov3_l_aic_rdttrack_heat_night.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night", "nightaug", "mm"),
    # E2: prompt + unfreeze + degrade with LOG-uniform light sampling
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log", "nightlog", "mm"),
    # E3: diffuse + unfreeze + degrade (log) -- also re-tests the fixed
    # OrthogonalProjectionFusion (true Gram-Schmidt)
    ("deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night.yml", "./outputs/deimv2_dinov3_l_aic_rdtdiffuse_unfreeze_night", "difnight", "mm"),
    # RGB-only trained on day + CUT-synthesised night (translation, not photometric sim)
    ("deimv2_dinov3_l_aic_nightcut.yml", "./outputs/deimv2_dinov3_l_aic_nightcut", "nightcut", "rgb"),
    # RGB-only day->night SIMULATED (photometric degrade, E4) + input_norm variant (E5)
    ("deimv2_dinov3_l_aic_nightsim.yml", "./outputs/deimv2_dinov3_l_aic_nightsim", "nightsim", "rgb"),
    ("deimv2_dinov3_l_aic_nightsim_inorm.yml", "./outputs/deimv2_dinov3_l_aic_nightsim_inorm", "inorm", "rgb"),
    # deeper backbone unfreeze on the nightlog recipe (12 blocks is the whole ViT-S)
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf6.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf6", "uf6", "mm"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf12.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_uf12", "uf12", "mm"),
    # E9 = nightlog + per-image RGB illumination normalisation (merge the two winners)
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_inorm", "nli", "mm"),
    # E10 = nightlog + REALISTIC night augmentation (lamps + Poisson-Gaussian noise + colour shift)
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight", "rnight", "mm"),
    ("deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall.yml", "./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall", "rnight_ufall", "mm"),
]

names = sorted(os.path.basename(p) for p in glob.glob(SRC + "/*"))
print(f"val images: {len(names)}")

for lv in LEVELS:
    tag = f"dark{int(round(lv * 100)):02d}"
    out = f"{ROOT}/data_aux/aic_valid_rgb_{tag}"
    os.makedirs(out, exist_ok=True)
    for n in names:
        dst = os.path.join(out, n)
        if os.path.exists(dst):
            continue
        a = np.array(Image.open(os.path.join(SRC, n)).convert("RGB"), np.float32) * lv
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(dst, quality=95)
    print(f"{tag}: {len(os.listdir(out))} images -> {out}")

    # darkened dataset file for the RGB-only model
    ds_src = open(os.path.join(DS, "aic_detection.yml")).read()
    assert VAL_IMG_LINE in ds_src, "aic_detection.yml val img_folder not found"
    ds_dst = f"aic_detection_{tag}.yml"
    open(os.path.join(DS, ds_dst), "w").write(ds_src.replace(VAL_IMG_LINE, f"    img_folder: {out}"))

    for cfg, out_dir, mtag, kind in MODELS:
        text = open(os.path.join(CFG, cfg)).read()
        head, val = text.split("val_dataloader:", 1)
        if kind == "mm":
            assert VAL_IMG_LINE in val, f"{cfg}: val img_folder not found"
            val = val.replace(VAL_IMG_LINE, f"    img_folder: {out}")
        else:
            assert "'../dataset/aic_detection.yml'" in head, f"{cfg}: include not found"
            head = head.replace("'../dataset/aic_detection.yml'",
                                f"'../dataset/{ds_dst}'")
        head = head.replace(f"output_dir: {out_dir}", f"output_dir: {out_dir}__{tag}")
        dst = cfg.replace(".yml", f"_val_{tag}.yml")
        open(os.path.join(CFG, dst), "w").write(head + "val_dataloader:" + val)
    print(f"  -> configs for {len(MODELS)} models at level {lv}")
