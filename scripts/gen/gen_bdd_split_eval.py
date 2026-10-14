"""Prepare fair evaluation for the BDD-night fine-tune:
  (1) night FT retrain config that uses an 80/20 held-out split
      (train on instances_bdd_night_train.json, val on instances_bdd_night_val.json)
      so the reported night mAP is on UNSEEN frames -- comparable to the zero-shot baseline.
  (2) a --test-only eval config that scores the zero-shot baseline
      (nightsim_inorm/best_stg1) on the SAME held-out 20% val, for a fair baseline number.
"""
import os
import yaml

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
BDD = "/home/constant/CodeBuddy/Dev/AIC/data_aux/BDD100K"
TRAIN = os.path.join(BDD, "instances_bdd_night_train.json")
VAL = os.path.join(BDD, "instances_bdd_night_val.json")

HDR = ("__include__:\n"
       "  - '../dataset/aic_detection.yml'\n"
       "  - '../runtime.yml'\n"
       "  - '../base/dataloader.yml'\n"
       "  - '../base/optimizer.yml'\n"
       "  - '../base/deimv2.yml'\n\n")


def val_block(ann):
    return {
        "type": "DataLoader",
        "dataset": {
            "type": "CocoDetection",
            "img_folder": os.path.join(BDD, "rgb"),
            "ann_file": ann,
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


# (1) night FT retrain on 80% split -----------------------------------------
base = yaml.safe_load(open(os.path.join(CFG, "deimv2_dinov3_l_aic_nightsim_inorm_bdd_night.yml")))
base["output_dir"] = "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_bdd_night_split"
base["train_dataloader"]["dataset"]["ann_file"] = TRAIN
base["val_dataloader"] = val_block(VAL)
# keep epoches=30 etc; bump batch a touch since fewer images (3143)
base["train_dataloader"]["total_batch_size"] = 8
dst = os.path.join(CFG, "deimv2_dinov3_l_aic_nightsim_inorm_bdd_night_split.yml")
open(dst, "w").write(HDR + yaml.safe_dump(base, sort_keys=False,
                                          default_flow_style=False, width=1000))
print("wrote", dst, "| train", os.path.basename(TRAIN), "val", os.path.basename(VAL))


# (2) baseline zero-shot eval on held-out 20% -------------------------------
evalcfg = yaml.safe_load(open(os.path.join(CFG, "deimv2_dinov3_l_aic_rgb3ch_bdd_night.yml")))
evalcfg["output_dir"] = "./outputs/deimv2_dinov3_l_aic_nightsim_inorm_bdd_night_val_baseline"
evalcfg["val_dataloader"] = val_block(VAL)
dst2 = os.path.join(CFG, "deimv2_dinov3_l_aic_nightsim_inorm_baseline_heldout.yml")
open(dst2, "w").write(HDR + yaml.safe_dump(evalcfg, sort_keys=False,
                                           default_flow_style=False, width=1000))
print("wrote", dst2, "| val", os.path.basename(VAL), "(zero-shot baseline eval)")
print("OK")
