"""BDD100K fine-tune configs -- train the converged AIC/Baseline ckpts ON real
BDD100K frames so the model adapts to the true night (and daytime) distribution.

Key facts discovered while wiring this up:
  * The AIC detector head is 12-class (num_classes: 12 in aic_detection.yml).
  * BDD100K's 5 AIC-overlapping classes use ids {0,4,5,6,8} in
    instances_bdd_{night,daytime}.json -- these ids ALREADY align with the AIC
    head (that is why the zero-shot eval produced sensible 0.140 / 0.147 numbers).
  * Therefore we fine-tune on instances_bdd_*.json directly (NOT the c5 0-4
    remap, which would misalign the 12-way head). num_classes stays 12; only the
    5 relevant logits receive gradient.

Two runs (both validate on BDD night so we can watch the target metric):
  night FT: resume deimv2_dinov3_l_aic_nightsim_inorm/best_stg1 (BDD-night 0.140)
            train on instances_bdd_night.json
  day  FT: resume deimv2_dinov3_l_aic/best_stg2          (BDD-night 0.147, rgb3ch)
            train on instances_bdd_daytime.json   (day->night transfer probe)

Short fine-tune schedule (resume from a converged ckpt): 30 ep, flat 15, low LR.
"""
import copy
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BASE = "deimv2_dinov3_l_aic.yml"
BDD = "/home/constant/CodeBuddy/Dev/AIC/data_aux/BDD100K"
RGB = os.path.join(BDD, "rgb")

base = yaml.safe_load(open(os.path.join(CFG, BASE)))


def make_ops(degrade):
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
    ]
    if degrade:
        o.append({"type": "RandomRGBDegrade5ch", "p": 0.5,
                  "brightness": [0.05, 0.80], "contrast": [0.30, 0.90],
                  "noise_std": [0.02, 0.12], "log_brightness": True})
    o += [
        {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
         "std": [0.229, 0.224, 0.225]},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]
    return o


def train_loader(ann, degrade):
    return {
        "type": "DataLoader",
        "dataset": {
            "type": "CocoDetection",
            "img_folder": RGB,
            "ann_file": ann,
            "return_masks": False,
            "transforms": {
                "type": "Compose",
                "ops": make_ops(degrade),
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


def configure(out_dir, ann, degrade, inorm):
    c = copy.deepcopy(base)
    c["output_dir"] = out_dir
    c["epoches"] = 30
    c["flat_epoch"] = 15
    c["no_aug_epoch"] = 8
    c["optimizer"]["lr"] = 0.0001
    c["train_dataloader"] = train_loader(ann, degrade)
    c["val_dataloader"] = VAL
    if inorm:
        c["DINOv3STAs"]["input_norm"] = "instance"
    return c


RUNS = [
    ("deimv2_dinov3_l_aic_nightsim_inorm_bdd_night.yml",
     "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_bdd_night",
     os.path.join(BDD, "instances_bdd_night.json"), True, True),
    ("deimv2_dinov3_l_aic_rgb3ch_bdd_daytime.yml",
     "./outputs/deimv2_dinov3_l_aic_rgb3ch_bdd_daytime",
     os.path.join(BDD, "instances_bdd_daytime.json"), False, False),
    # DNDT (nightcut_inorm) base -- strongest single-modality model on the synthetic
    # / realistic night protocol (darkXX MEAN 0.5210, beats inorm 0.513). FT on real
    # BDD daytime to close its synthetic->real BDD night domain gap.
    ("deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime.yml",
     "./outputs/deimv2_dinov3_l_aic_nightcut_inorm_bdd_daytime",
     os.path.join(BDD, "instances_bdd_daytime.json"), False, True),
]

HDR = ("__include__:\n"
       "  - '../dataset/aic_detection.yml'\n"
       "  - '../runtime.yml'\n"
       "  - '../base/dataloader.yml'\n"
       "  - '../base/optimizer.yml'\n"
       "  - '../base/deimv2.yml'\n\n")

for nm, out, ann, degrade, inorm in RUNS:
    c = configure(out, ann, degrade, inorm)
    dst = os.path.join(CFG, nm)
    open(dst, "w").write(HDR + yaml.safe_dump(c, sort_keys=False,
                                              default_flow_style=False, width=1000))
    print("wrote", dst)

# --- sanity check ------------------------------------------------------------
for nm, out, ann, degrade, inorm in RUNS:
    c = yaml.safe_load(open(os.path.join(CFG, nm)))
    tr, va = c["train_dataloader"]["dataset"], c["val_dataloader"]["dataset"]
    print(f"\n{nm}")
    print("  train ann:", os.path.basename(tr["ann_file"]),
          "| degrade:", any(o.get("type") == "RandomRGBDegrade5ch"
                            for o in tr["transforms"]["ops"]),
          "| inorm:", c["DINOv3STAs"].get("input_norm", "none"))
    print("  val   ann:", os.path.basename(va["ann_file"]),
          "| epoches:", c["epoches"], "| lr:", c["optimizer"]["lr"],
          "| out:", c["output_dir"])
print("\nOK")
