#!/usr/bin/env python3
"""Dump per-image val200 detections for a trained checkpoint.

Why: COCOeval only reports a point estimate of mAP. On a 200-image val split a
0.4-pt gap between two models can be pure noise, and we cannot tell without a
confidence interval -- which requires the raw detections, not just the scalar.

This mirrors engine/solver/det_solver.py::val() exactly (same val_dataloader,
same postprocessor, same EMA module) so the mAP recomputed from the dump
reproduces the training log to the last digit.

Usage (from repo root):
  /home/constant/miniconda3/envs/deimv2/bin/python scripts/analysis/dump_val_dets.py \
      --config <cfg.yml> --ckpt <xxx.pth> --out outputs/analysis/_dets/<name>.pkl
"""

import argparse
import os
import pickle
import sys

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VENDOR = os.path.join(ROOT, "vendor", "DEIMv2")

# nightify_aic / prep_night_real are imported by the transform stack as top-level
# module names, so scripts/lib must be on the path *before* DEIMv2 builds the
# dataloader. The chdir is what train.py effectively relies on too.
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
sys.path.insert(0, ROOT)
os.chdir(VENDOR)
sys.path.insert(0, VENDOR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine.core import YAMLConfig  # noqa: E402
from engine.solver import TASKS  # noqa: E402
from engine.misc import dist_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dist_utils.setup_distributed(0, "builtin", seed=0)

    cfg = YAMLConfig(args.config, resume=args.ckpt)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()  # setup only: builds val_dataloader + loads resume ckpt

    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    dev = solver.device

    n_img = 0
    dets = {}
    with torch.no_grad():
        for i, (samples, targets) in enumerate(solver.val_dataloader):
            samples = samples.to(dev)
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

            outputs = module(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = solver.postprocessor(outputs, orig_sizes, samples)

            for t, r in zip(targets, results):
                iid = int(t["image_id"].item())
                dets[iid] = {
                    "boxes": r["boxes"].detach().cpu().numpy().astype(np.float64),
                    "scores": r["scores"].detach().cpu().numpy().astype(np.float64),
                    "labels": r["labels"].detach().cpu().numpy().astype(np.int64),
                }
                n_img += 1
            if (i + 1) % 10 == 0:
                print(f"  batch {i + 1}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"tag": args.tag or args.out, "dets": dets}, f)
    print(f"OK tag={args.tag} images={n_img} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
