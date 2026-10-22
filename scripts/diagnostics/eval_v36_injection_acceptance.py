#!/usr/bin/env python3
"""Acceptance evaluation of four V36 injection checkpoints on ailab40 GPU 1 (Attempt 2).

Strict load of checkpoint0007.pth / ema.module on common validation inputs.
Aligned execution environment with training (Volta SM_70: torch.backends.cudnn.enabled = False).
Evaluates both:
1. Standard COCO metrics via post-NMS predictions (NMS IoU 0.60, max 300 queries, min_score=0)
2. Diagnostic pre-NMS predictions (raw top 300 queries, min_score=0)
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import yaml


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, required=True,
                   help="Path to vendor_DEIMv2")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Path to dataset root (e.g. AIC_V29)")
    p.add_argument("--training-dir", type=Path, required=True,
                   help="Path to training output directory containing the 4 runs")
    p.add_argument("--config-dir", type=Path, required=True,
                   help="Path to configs/deimv2/v36_injection_warmstart_20261019")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for evaluations, predictions, and manifest")
    p.add_argument("--device", default="cuda:0",
                   help="Device string (mapped under CUDA_VISIBLE_DEVICES)")
    args = p.parse_args()

    framework = args.framework.resolve()
    data_root = args.data_root.resolve()
    training_dir = args.training_dir.resolve()
    config_dir = args.config_dir.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify GPU availability
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={physical}")
    if physical != "":
        gpu_query = subprocess.check_output(
            ["nvidia-smi", "-i", physical, "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True
        ).strip()
        if gpu_query:
            raise RuntimeError(f"GPU {physical} has active compute processes: {gpu_query}")
        print(f"Verified GPU {physical} is clean and idle.")

    # 2. Setup environment and imports
    sys.path[:0] = [str(framework), str(framework / "scripts_lib"), str(framework.parent / "scripts/lib")]
    os.chdir(framework)

    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.core.yaml_utils import load_config
    from engine.deim.postprocessor import PostProcessor
    from engine.data.dataset.coco_eval import CocoEvaluator
    from destripe_1d import stripe_non_uniformity_correction_1d

    stripe_non_uniformity_correction_1d(np.zeros((32, 32), dtype=np.uint8))
    print("Verified destripe_1d dependency.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Enforce exact match with Volta SM_70 train.py workaround:
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    print(f"Execution environment configured: cudnn.enabled={torch.backends.cudnn.enabled}, "
          f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
          f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}")

    # 3. Verify validation annotations and 800 input files
    ann_path = data_root / "work_dir/deimv2_dataset/valid/annotations/instances_val.json"
    ann_sha = sha256_file(ann_path)
    ann_data = json.loads(ann_path.read_text())
    assert len(ann_data["images"]) == 200, f"Expected 200 images, got {len(ann_data['images'])}"
    print(f"Validation annotations: {ann_path} (SHA: {ann_sha}, images: 200, annotations: {len(ann_data['annotations'])})")

    folders = {
        "img_folder": data_root / "dataset/train/visible",
        "ir_folder": data_root / "dataset/train/infrared_aligned",
        "raw_ir_folder": data_root / "work_dir/train_lbg_ir",
        "depth_folder": data_root / "dataset/train/depth"
    }

    val_inputs = []
    print("Verifying 800 input files across 4 modalities...")
    for img in ann_data["images"]:
        item = {"id": img["id"], "file_name": img["file_name"], "files": {}}
        stem = Path(img["file_name"]).stem
        for key, folder in folders.items():
            candidates = [folder / img["file_name"]] if key == "img_folder" else [folder / (stem + ".png"), folder / img["file_name"]]
            if key == "depth_folder":
                candidates = [folder / (stem + e) for e in (".png", ".jpg", ".jpeg")]
            f = next((c for c in candidates if c.is_file()), None)
            if f is None:
                raise FileNotFoundError(f"Missing {key} for image {img['file_name']} in {folder}")
            item["files"][key] = {"path": str(f), "sha256": sha256_file(f)}
        val_inputs.append(item)

    val_inputs_path = out_dir / "val_inputs.json"
    dump_json(val_inputs_path, val_inputs)
    val_inputs_sha = sha256_file(val_inputs_path)
    print(f"Verified 800 validation input files. Manifest written to {val_inputs_path} (SHA: {val_inputs_sha})")

    # Source code SHA256 snapshot
    engine_sources = {
        str(f.relative_to(framework)): sha256_file(f)
        for f in sorted((framework / "engine").rglob("*.py"))
    }

    runs = ["seed42_control", "seed42_p3p4", "seed43_control", "seed43_p3p4"]
    manifest = {
        "task": "A02 four final EMA checkpoint acceptance",
        "attempt": 2,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": args.device,
        "physical_gpu": physical,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_note": "Original train.py set TF32=True; Volta SM_70 hardware does not support hardware TF32 (Ampere+ only); evaluated with FP32 and matmul_tf32=False",
        "cudnn_note": "Original train.py on Volta SM_70 explicitly set cudnn.enabled=False to avoid Winograd FP16/alignment issues; matched here with cudnn.enabled=False",
        "eval_spatial_size": [1440, 2560],
        "eval_batch_size": 1,
        "annotation_path": str(ann_path),
        "annotation_sha256": ann_sha,
        "val_inputs_sha256": val_inputs_sha,
        "source_engine_sha256": engine_sources,
        "runs": {}
    }

    metrics_summary = {}

    device = torch.device(args.device)

    # 4. Iterate over the 4 runs
    for run_name in runs:
        print(f"\n==========================================")
        print(f"Evaluating Run: {run_name}")
        print(f"==========================================")
        start_time = time.time()

        # Load training config
        cfg_path = config_dir / f"{run_name}.yml"
        cfg_raw_sha = sha256_file(cfg_path)
        with open(cfg_path) as f:
            c = copy.deepcopy(yaml.safe_load(f))
        c.pop("__include__", None)

        # Confirm depth contract
        train_ds = c.get("train_dataloader", {}).get("dataset", {})
        if train_ds.get("depth_read", "raw16") != "raw16" or train_ds.get("depth_mode", "raw") != "raw":
            raise ValueError(f"Training depth contract mismatch for {run_name}")

        # Configure evaluation parameters
        c["use_amp"] = False
        c["remap_mscoco_category"] = False
        c["eval_spatial_size"] = [1440, 2560]
        c["DEIM"].update(
            use_dinov3_distill=False,
            use_co_aux=False,
            use_internal_tta=False,
            use_internal_sahi=False
        )
        bb = c["DEIM"]["backbone"]
        c[bb]["pretrained"] = False

        # PostProcessor: primary evaluation uses NMS 0.60
        c["PostProcessor"] = {
            "num_classes": 12,
            "num_top_queries": 300,
            "use_nms": True,
            "nms_iou_thr": 0.6
        }

        # Save aligned evaluation YAML
        aligned_cfg_path = out_dir / f"{run_name}_eval.yml"
        aligned_cfg_path.write_text(yaml.safe_dump(c, sort_keys=False))
        aligned_cfg_sha = sha256_file(aligned_cfg_path)

        # Build model and dataloader via YAMLConfig
        cfg = YAMLConfig(str(aligned_cfg_path))
        val_loader = cfg.val_dataloader

        model = cfg.model.float().eval()

        # Load Checkpoint 0007 EMA
        ckpt_path = training_dir / run_name / "checkpoint0007.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_sha = sha256_file(ckpt_path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        last_epoch = ck.get("last_epoch")
        ckpt_date = ck.get("date")

        if "ema" not in ck or "module" not in ck["ema"]:
            raise ValueError(f"No EMA module in checkpoint {ckpt_path}")
        state = ck["ema"]["module"]

        # Verify class count
        enc_weight = state.get("decoder.enc_score_head.weight")
        head_classes = enc_weight.shape[0] if enc_weight is not None else None
        if head_classes != 12:
            raise ValueError(f"Expected 12 classes in head, got {head_classes}")

        # Omit only training-specific keys
        omitted_keys = [k for k in state if k.startswith(("dinov3_distiller.", "co_aux_head."))]
        clean_state = {k.removeprefix("module."): v for k, v in state.items() if k not in omitted_keys}

        # Strict load
        load_result = model.load_state_dict(clean_state, strict=True)
        print(f"Model loaded with strict=True (missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})")
        print(f"Total EMA keys: {len(state)}, Omitted training keys: {len(omitted_keys)}, Loaded parameters: {len(clean_state)}")

        model.to(device)

        # Setup PostProcessors
        post_nms = PostProcessor(num_classes=12, num_top_queries=300, use_nms=True, nms_iou_thr=0.6).to(device).eval()
        pre_nms = PostProcessor(num_classes=12, num_top_queries=300, use_nms=False).to(device).eval()

        # Setup Evaluator
        evaluator = CocoEvaluator(val_loader.dataset.coco, ("bbox",))

        post_nms_predictions = []
        pre_nms_predictions = []
        seen_images = []

        print(f"Running evaluation on 200 validation images (cudnn.enabled={torch.backends.cudnn.enabled})...")
        with torch.inference_mode():
            for i, (samples, targets) in enumerate(val_loader):
                samples = samples.to(device)
                sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
                image_id = int(targets[0]["image_id"].item())
                seen_images.append(image_id)

                outputs = model(samples)
                post_res = post_nms(outputs, sizes)[0]
                pre_res = pre_nms(outputs, sizes)[0]

                evaluator.update({image_id: post_res})

                post_fmt = evaluator.prepare({image_id: post_res}, "bbox")
                post_nms_predictions.extend(post_fmt)

                pre_fmt = evaluator.prepare({image_id: pre_res}, "bbox")
                pre_nms_predictions.extend(pre_fmt)

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{run_name}] Processed {i + 1}/200 images ({elapsed:.1f}s)")

        assert len(seen_images) == 200 and len(set(seen_images)) == 200, "Validation set must contain exactly 200 unique images"

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        coco_stats = [float(x) for x in evaluator.coco_eval["bbox"].stats]
        metric_names = [
            "mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
            "AR_1", "AR_10", "AR_100", "AR_s", "AR_m", "AR_l"
        ]
        coco_metrics = dict(zip(metric_names, coco_stats))

        # Save predictions and evaluator object
        post_nms_file = out_dir / f"{run_name}_predictions_post_nms.json"
        dump_json(post_nms_file, post_nms_predictions)
        post_nms_sha = sha256_file(post_nms_file)

        pre_nms_file = out_dir / f"{run_name}_predictions_pre_nms.json"
        dump_json(pre_nms_file, pre_nms_predictions)
        pre_nms_sha = sha256_file(pre_nms_file)

        eval_pth = out_dir / f"{run_name}_coco_eval.pth"
        torch.save(evaluator.coco_eval["bbox"].eval, eval_pth)
        eval_pth_sha = sha256_file(eval_pth)

        # Extract training log epoch 7 metrics for comparison
        log_file = training_dir / run_name / "log.txt"
        train_log_ep7 = None
        if log_file.is_file():
            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("epoch") == 7:
                            train_log_ep7 = row.get("test_coco_eval_bbox")
                            break

        diff_vs_log = None
        if train_log_ep7:
            diff_vs_log = {
                "mAP": coco_metrics["mAP"] - train_log_ep7[0],
                "AP50": coco_metrics["AP50"] - train_log_ep7[1],
                "AP75": coco_metrics["AP75"] - train_log_ep7[2],
                "AP_s": coco_metrics["AP_s"] - train_log_ep7[3],
                "AP_m": coco_metrics["AP_m"] - train_log_ep7[4],
                "AP_l": coco_metrics["AP_l"] - train_log_ep7[5],
            }

        elapsed_total = time.time() - start_time

        run_record = {
            "name": run_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "last_epoch": last_epoch,
            "checkpoint_date": ckpt_date,
            "head_classes": head_classes,
            "total_ema_keys": len(state),
            "omitted_training_keys_count": len(omitted_keys),
            "omitted_training_keys": omitted_keys,
            "strict_load": True,
            "training_config_path": str(cfg_path),
            "training_config_sha256": cfg_raw_sha,
            "aligned_config_path": str(aligned_cfg_path),
            "aligned_config_sha256": aligned_cfg_sha,
            "predictions_post_nms_path": str(post_nms_file),
            "predictions_post_nms_sha256": post_nms_sha,
            "predictions_post_nms_count": len(post_nms_predictions),
            "predictions_pre_nms_path": str(pre_nms_file),
            "predictions_pre_nms_sha256": pre_nms_sha,
            "predictions_pre_nms_count": len(pre_nms_predictions),
            "coco_eval_pth": str(eval_pth),
            "coco_eval_sha256": eval_pth_sha,
            "coco_metrics": coco_metrics,
            "train_log_epoch7_metrics": {
                k: train_log_ep7[idx] for idx, k in enumerate(["mAP", "AP50", "AP75", "AP_s", "AP_m", "AP_l"])
            } if train_log_ep7 else None,
            "diff_vs_training_log": diff_vs_log,
            "seconds": elapsed_total,
            "evaluated_image_count": len(seen_images)
        }

        manifest["runs"][run_name] = run_record
        metrics_summary[run_name] = {
            "coco_metrics": coco_metrics,
            "train_log_epoch7_metrics": run_record["train_log_epoch7_metrics"],
            "diff_vs_training_log": diff_vs_log,
            "seconds": elapsed_total
        }

        print(f"[{run_name}] Done in {elapsed_total:.1f}s.")
        print(f"  COCO mAP={coco_metrics['mAP']:.6f}, AP50={coco_metrics['AP50']:.6f}, AP75={coco_metrics['AP75']:.6f}, AP_s={coco_metrics['AP_s']:.6f}")
        if diff_vs_log:
            print(f"  Diff vs Train Log: ΔmAP={diff_vs_log['mAP']:+.6f}, ΔAP50={diff_vs_log['AP50']:+.6f}, ΔAP75={diff_vs_log['AP75']:+.6f}, ΔAP_s={diff_vs_log['AP_s']:+.6f}")

        # Cleanup memory
        del model, post_nms, pre_nms, cfg, val_loader, ck, state, clean_state
        gc.collect()
        torch.cuda.empty_cache()

    dump_json(out_dir / "manifest.json", manifest)
    dump_json(out_dir / "metrics_summary.json", metrics_summary)
    print("\n==========================================")
    print("ALL 4 RUNS SUCCESSFULLY EVALUATED (ATTEMPT 2)!")
    print(f"Manifest written to: {out_dir / 'manifest.json'}")
    print(f"Metrics summary written to: {out_dir / 'metrics_summary.json'}")
    print("==========================================")


if __name__ == "__main__":
    main()
