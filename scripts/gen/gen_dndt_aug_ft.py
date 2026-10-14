"""DNDT-online-aug fine-tune: let the detector LEARN the inverse of the DNDT night
degradation from the training data itself (no hand-coded test-time inverse).

We take the strongest single-modality model (nightsim_inorm, per-image RGB
illumination normalisation) and fine-tune it on BDD100K *daytime* frames with a
new online augmentation ``RandomDNDTNightify`` (vendor/.../transforms/_transforms.py)
that wraps our own physically-grounded DNDT forward model (nightify_aic.nightify).

Every training sample is, with probability p, pushed through a FRESH random draw of
the full DNDT factor manifold (gamma darkening, ambient attenuation, colour-temp
shift, headlight bloom, vignette, desaturation, sensor noise) at a随机 exposure
level spanning dusk..deep-night -- and with curriculum=True the exposure anneals
from bright (easy) to dark (hard) across the epochs. The detector must therefore
learn to *see through* the degradation, i.e. to invert it internally, using only
daytime labels.

Validation is on REAL BDD100K night (the target metric), so the training signal is
synthetic DNDT but the measured gain is on真实夜间.

Init : deimv2_dinov3_l_aic_nightsim_inorm/best_stg1  (inorm base)
Train: instances_bdd_daytime.json  + RandomDNDTNightify (inorm on)
Val  : instances_bdd_night.json    (real night)
"""
import copy
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic.yml"
BDD = "/home/constant/CodeBuddy/Dev/AIC/data_aux/BDD100K"
RGB = os.path.join(BDD, "rgb")

base = yaml.safe_load(open(os.path.join(CFG, BASE)))


def make_ops(p=0.5, bright_range=(0.005, 0.5), curriculum=True, total_epochs=30):
    o = [
        {"type": "Mosaic", "output_size": 320, "rotation_range": 10,
         "translation_range": [0.1, 0.1], "scaling_range": [0.5, 1.5],
         "probability": 1.0, "fill_value": 0, "use_cache": True,
         "max_cached_images": 50, "random_pop": True},
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [640, 640]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        # --- learn the DNDT inverse online (random factors + curriculum) ---
        {"type": "RandomDNDTNightify", "p": p, "bright_range": list(bright_range),
         "curriculum": curriculum, "total_epochs": total_epochs},
        {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
         "std": [0.229, 0.224, 0.225]},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]
    return o


def train_loader(ann, p, bright_range, curriculum, total_epochs):
    return {
        "type": "DataLoader",
        "dataset": {
            "type": "CocoDetection",
            "img_folder": RGB,
            "ann_file": ann,
            "return_masks": False,
            "transforms": {
                "type": "Compose",
                "ops": make_ops(p, bright_range, curriculum, total_epochs),
                "policy": {"epoch": [4, 15, 28]},
            },
        },
        "shuffle": True,
        "num_workers": 4,
        "drop_last": True,
        "collate_fn": {
            "type": "BatchImageCollateFunction",
            "mixup_epochs": [4, 15],
            "stop_epoch": 28,
            "copyblend_epochs": [4, 28],
            "base_size_repeat": 3,
        },
        "total_batch_size": 8,
    }


VAL = {
    "type": "DataLoader",
    "dataset": {
        "type": "CocoDetection",
        "img_folder": RGB,
        "ann_file": os.path.join(BDD, "instances_bdd_night.json"),
        "return_masks": False,
        "transforms": {
            "type": "Compose",
            "ops": [
                {"type": "Resize", "size": [640, 640]},
                {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]},
                {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
            ],
        },
    },
    "shuffle": False,
    "num_workers": 4,
    "drop_last": False,
    "collate_fn": {"type": "BatchImageCollateFunction"},
    "total_batch_size": 16,
}


def configure(out_dir, ann, p, bright_range, curriculum, total_epochs):
    c = copy.deepcopy(base)
    c["output_dir"] = out_dir
    c["epoches"] = total_epochs
    c["flat_epoch"] = max(1, total_epochs // 2)
    c["no_aug_epoch"] = 8
    c["optimizer"]["lr"] = 0.0001
    c["train_dataloader"] = train_loader(ann, p, bright_range, curriculum, total_epochs)
    c["val_dataloader"] = VAL
    c["DINOv3STAs"]["input_norm"] = "instance"   # inorm already learns brightness inv
    return c


RUNS = [
    ("deimv2_dinov3_l_aic_nightsim_inorm_dndtaug.yml",
     "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_dndtaug",
     os.path.join(BDD, "instances_bdd_daytime.json"), 0.5, (0.005, 0.5), True, 30),
]

HDR = ("__include__:\n"
       "  - '../dataset/aic_detection.yml'\n"
       "  - '../runtime.yml'\n"
       "  - '../base/dataloader.yml'\n"
       "  - '../base/optimizer.yml'\n"
       "  - '../base/deimv2.yml'\n\n")

for nm, out, ann, p, br, cur, te in RUNS:
    c = configure(out, ann, p, br, cur, te)
    dst = os.path.join(CFG, nm)
    open(dst, "w").write(HDR + yaml.safe_dump(c, sort_keys=False,
                                              default_flow_style=False, width=1000))
    print("wrote", dst)

# --- sanity check ------------------------------------------------------------
for nm, out, ann, p, br, cur, te in RUNS:
    c = yaml.safe_load(open(os.path.join(CFG, nm)))
    tr, va = c["train_dataloader"]["dataset"], c["val_dataloader"]["dataset"]
    has_dndt = any(o.get("type") == "RandomDNDTNightify"
                   for o in tr["transforms"]["ops"])
    print(f"\n{nm}")
    print("  train ann:", os.path.basename(tr["ann_file"]),
          "| RandomDNDTNightify:", has_dndt,
          "| inorm:", c["DINOv3STAs"].get("input_norm", "none"))
    print("  val   ann:", os.path.basename(va["ann_file"]),
          "| epoches:", c["epoches"], "| lr:", c["optimizer"]["lr"],
          "| out:", c["output_dir"])
print("\nOK")
