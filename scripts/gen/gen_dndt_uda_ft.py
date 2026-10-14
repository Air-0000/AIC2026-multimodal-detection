"""DNDT (Debiased Teacher, Cui et al. ICCV 2025) UDA fine-tune for real-night detection.

Implements the paper's full self-training loop (solver = DNDTSolver / task: dndt):
  * DNDT  : SOURCE loader applies RandomDNDTNightify -> labelled daytime becomes night-like.
  * CDRC  : objects from those night-like frames are pasted into REAL-night scenes (ICFB
             class-frequency balancing) -> extra supervised night-object signal.
  * ConCal : the EMA teacher scores REAL-night frames; a pseudo box is kept only when the
             student agrees (teacher-student joint prediction) AND clears a class threshold.

Source (labelled, day->night-like)  : instances_bdd_daytime.json
Target (unlabelled, real night)     : instances_bdd_night.json   (boxes ignored in training)
Val   (real night, the metric)      : instances_bdd_night.json

Init : deimv2_dinov3_l_aic_nightsim_inorm/best_stg1  (inorm base)
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
    return [
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
        # DNDT: push daytime into the night-like manifold (random factors + curriculum)
        {"type": "RandomDNDTNightify", "p": p, "bright_range": list(bright_range),
         "curriculum": curriculum, "total_epochs": total_epochs},
        {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
         "std": [0.229, 0.224, 0.225]},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]


def train_loader(ann):
    return {
        "type": "DataLoader",
        "dataset": {
            "type": "CocoDetection",
            "img_folder": RGB,
            "ann_file": ann,
            "return_masks": False,
            "transforms": {"type": "Compose", "ops": make_ops(), "policy": {"epoch": [4, 15, 28]}},
        },
        "shuffle": True,
        "num_workers": 2,
        "drop_last": True,
        "collate_fn": {"type": "BatchImageCollateFunction",
                       "mixup_epochs": [4, 15], "stop_epoch": 28,
                       "copyblend_epochs": [4, 28], "base_size_repeat": 3},
        "total_batch_size": 8,
    }


def target_loader(ann):
    """Real-night UNLABELLED loader (boxes present but ignored by the solver)."""
    return {
        "type": "DataLoader",
        "dataset": {
            "type": "CocoDetection",
            "img_folder": RGB,
            "ann_file": ann,
            "return_masks": False,
            "transforms": {"type": "Compose", "ops": [
                {"type": "Resize", "size": [640, 640]},
                {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]},
                {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
            ]},
        },
        "shuffle": True,
        "num_workers": 0,
        "drop_last": True,
        "collate_fn": {"type": "BatchImageCollateFunction"},
        "total_batch_size": 8,
    }


VAL = {
    "type": "DataLoader",
    "dataset": {
        "type": "CocoDetection",
        "img_folder": RGB,
        "ann_file": os.path.join(BDD, "instances_bdd_night.json"),
        "return_masks": False,
        "transforms": {"type": "Compose", "ops": [
            {"type": "Resize", "size": [640, 640]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "Normalize", "mean": [0.485, 0.456, 0.406],
             "std": [0.229, 0.224, 0.225]},
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]},
    },
    "shuffle": False,
    "num_workers": 0,
    "drop_last": False,
    "collate_fn": {"type": "BatchImageCollateFunction"},
    "total_batch_size": 16,
}


def configure(out_dir, src_ann, tgt_ann, dndt):
    c = copy.deepcopy(base)
    c["task"] = "dndt"                       # select DNDTSolver
    c["output_dir"] = out_dir
    c["epoches"] = 30
    c["flat_epoch"] = 15
    c["no_aug_epoch"] = 8
    c["optimizer"]["lr"] = 0.0001
    c["train_dataloader"] = train_loader(src_ann)
    c["target_dataloader"] = target_loader(tgt_ann)
    c["val_dataloader"] = VAL
    c["dndt"] = dndt
    c["DINOv3STAs"]["input_norm"] = "instance"
    return c


DNDT = {"w_src": 1.0, "w_cdrc": 1.0, "w_tgt": 2.0,
        "cls_thr": 0.5, "iou_thr": 0.5, "max_obj": 4, "icfb": True, "p_mix": 1.0}

RUNS = [
    ("deimv2_dinov3_l_aic_nightsim_inorm_dndt_uda.yml",
     "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_dndt_uda",
     os.path.join(BDD, "instances_bdd_daytime.json"),
     os.path.join(BDD, "instances_bdd_night.json")),
]

HDR = ("__include__:\n"
       "  - '../dataset/aic_detection.yml'\n"
       "  - '../runtime.yml'\n"
       "  - '../base/dataloader.yml'\n"
       "  - '../base/optimizer.yml'\n"
       "  - '../base/deimv2.yml'\n\n")

for nm, out, src, tgt in RUNS:
    c = configure(out, src, tgt, DNDT)
    dst = os.path.join(CFG, nm)
    open(dst, "w").write(HDR + yaml.safe_dump(c, sort_keys=False,
                                              default_flow_style=False, width=1000))
    print("wrote", dst)

# --- sanity check ------------------------------------------------------------
for nm, out, src, tgt in RUNS:
    c = yaml.safe_load(open(os.path.join(CFG, nm)))
    print(f"\n{nm}")
    print("  task:", c.get("task"), "| inorm:", c["DINOv3STAs"].get("input_norm"),
          "| epoches:", c["epoches"], "| lr:", c["optimizer"]["lr"])
    print("  train ann:", os.path.basename(c["train_dataloader"]["dataset"]["ann_file"]),
          "| has RandomDNDTNightify:",
          any(o.get("type") == "RandomDNDTNightify" for o in c["train_dataloader"]["dataset"]["transforms"]["ops"]))
    print("  target ann:", os.path.basename(c["target_dataloader"]["dataset"]["ann_file"]),
          "| val ann:", os.path.basename(c["val_dataloader"]["dataset"]["ann_file"]))
    print("  dndt hyperparams:", c.get("dndt"))
print("\nOK")
