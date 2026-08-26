# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Overview

This is an **AIC (AI City) competition** repository for **urban-scene visual multimodal object detection**. The task is to detect 12 object classes from multi-modal imagery (visible, infrared, depth) and emit predictions in the official submission format. The repo contains data-prep, training, inference, ensembling, and evaluation scripts plus vendored detection frameworks.

Multi-modal inputs live under `dataset/train/{visible,infrared,depth}` and `dataset/test/{visible,infrared,depth}`, with YOLO-format labels in `dataset/train/labels/*.txt` (`class_id cx cy w h`, normalized). Output predictions are per-image `.txt` files in the format `<class_id> <cx> <cy> <w> <h> <confidence>` (normalized, CRLF, ≤100 boxes/image, 12 classes). This format is verified by `check_submission.py`.

## Environment & Conda

Three conda environments are used and must be activated explicitly — scripts are NOT portable across them:
- `deimv2` (`/home/constant/miniconda3/envs/deimv2/bin/python`) — runs the DEIMv2/RT-DETR pipeline (`predict_deimv2.py`, all `deimv2*`/`*.log` training).
- `rfdetr` — runs RF-DETR training and RGB inference (`train_rgb*.py`, `predict_rgb.py`, `prepare_dataset.py`, `predict_val.py`).
- `torch25` — runs DINOv3-backbone RF-DETR training/inference (`train_rgb_dinov3.py`).

Scripts hardcode `ROOT = /home/constant/CodeBuddy/Dev/AIC` and several change cwd into `vendor/DEIMv2` (`predict_deimv2.py`, `train_rgb*.py`). Always run from the repo root or match these assumptions.

## Pipeline Architecture (big picture)

The repo is a **multi-strategy detection ensemble** built on two vendored detectors, glued by custom scripts:

1. **RF-DETR (RGB baseline)** — `vendor/rfdetr_models/` + `vendor/eva02/`. `prepare_dataset.py` builds a YOLO/Roboflow layout (`work_dir/rfdetr_dataset`) using **symlinks** to avoid copying ~2GB imagery. `train_rgb.py` trains a vanilla RF-DETR; `train_rgb_dinov3.py` swaps in a custom DINOv3 ViT-L/16 backbone (`vendor/dinov3_backbone.py`) that mirrors Meta's DINOv3 to load `vendor/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` key-for-key. `predict_rgb.py` produces RGB-only submissions.

2. **DEIMv2 (RT-DETR variant, primary model)** — vendored under `vendor/DEIMv2/` (its own `configs/`, `engine/`, `outputs/`). `predict_deimv2.py` loads `vendor/DEIMv2/outputs/deimv2_dinov3_l_aic/best_stg2.pth` and writes submission TXTs. Checkpoints use EMA (`ck["ema"]["module"]`) or `ck["model"]` keys — handle both (see `predict_deimv2.py` lines 42-49).

3. **Multimodal fusion via infrared "heat" channel** — `prep_ir_heat.py` converts paired RGB+infrared into a 4-channel (RGB + heat) PNG dataset at `work_dir/deimv2_heat`, and `deimv2_dataset` is the RGB version. `train_rgb_dinov3.py` is also driven to produce a "heat" variant model. This is the key multi-modal trick: infrared becomes an extra input channel rather than a separate branch.

4. **Test-Time Augmentation (TTA)** — `tta_predict.py`, `tta_multi_thr.py`, `predict_tta.py`, `val_tta_deimv2.py`, `tta_cache_raw.pkl` cache raw predictions so TTA flips/scales can be re-applied cheaply. TTA multiplies inference cost but improves mAP.

5. **Threshold sweep & ensemble** — `thresh_sweep.py` / `eval_threshold.py` scan confidence thresholds to maximize validation mAP (`work_dir/threshold_sweep.json`). `predict_multi_thr.py` emits per-threshold submission variants. `wbf_fuse.py` implements **Weighted Boxes Fusion** to merge overlapping predictions from multiple models (RGB, heat, TTA) into a single higher-quality submission.

## Evaluation & Validation

- `eval_submission.py` computes mAP against a held-out validation split built from `dataset/train` (split defined inside the script; uses `valid/` images). It reads prediction TXTs and compares to ground-truth labels.
- `check_submission.py` is the **format gate**: validates file coverage vs the test set, ≤100 boxes, 6 columns, valid class id (0-11), normalized coords/confidence in [0,1], and CRLF line endings. Run it before submitting.
- `predict_val.py` runs RF-DETR predictions on the validation split for `eval_submission.py`.

## Outputs & Artifacts

- `work_dir/submission*/` — generated prediction folders (one `.txt` per test image).
- `work_dir/submission_zips/` — zipped submissions for upload.
- `work_dir/*.log` — training/inference logs; `train_map_curve.png`, `deimv2_vs_baseline*.png` — diagnostic plots.
- `deimv2_dinov3_l_coco.pth` (repo root) and `vendor/*.pth` — backbone/pretrained weights.

## Common Commands

Build RF-DETR dataset (symlinks, splits 1800 train / 200 valid):
```
conda run -n rfdetr python prepare_dataset.py
```

Train vanilla RF-DETR (RGB baseline):
```
conda run -n rfdetr python train_rgb.py
```

Train DINOv3-backbone RF-DETR (RGB + heat variants):
```
conda run -n torch25 python train_rgb_dinov3.py
```

Run DEIMv2 inference to submission TXTs (CKPT OUT THRESH SIZE):
```
conda run -n deimv2 python predict_deimv2.py [CKPT] [OUT_DIR] [THRESH] [SIZE]
```

Build RGB-only submission:
```
conda run -n rfdetr python predict_rgb.py
```

Sweep confidence thresholds for best validation mAP:
```
conda run -n rfdetr python thresh_sweep.py
```

Fuse multiple model predictions via WBF into one submission:
```
conda run -n rfdetr python wbf_fuse.py
```

Validate a submission's format before upload:
```
python3 check_submission.py [SUB_DIR] [TEST_VISIBLE_DIR]
```

Evaluate mAP of a submission vs validation ground truth:
```
conda run -n rfdetr python eval_submission.py [PRED_DIR] [GT_DIR]
```

Run TTA predictions (cached raw, multi-threshold):
```
conda run -n deimv2 python tta_predict.py
conda run -n deimv2 python tta_multi_thr.py
```

## Key Constraints / Gotchas

- **12 classes** (order fixed in `prepare_dataset.py`): person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle.
- Labels are YOLO-normalized `[class cx cy w h]`; submissions add confidence and use CRLF.
- Symlink-based datasets: do not delete `dataset/` or break symlinks in `work_dir/*_dataset`.
- DEIMv2 scripts `os.chdir` into `vendor/DEIMv2`; run them from repo root.
- Checkpoint loading must handle EMA/`model` key variants; `strict=False` is used.
- The `dataset/test` has no labels — only `check_submission.py` (format) applies there, never mAP eval.
