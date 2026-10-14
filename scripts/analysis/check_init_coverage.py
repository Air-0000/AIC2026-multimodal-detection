#!/usr/bin/env python3
"""Quantify how much of the vitl800_clean model started from RANDOM init.

Compares the warm-start checkpoint (ckpts/dinov3/vitl16_rdttrack_init.pth) against
the current model state_dict (outputs/.../vitl800_clean/last.pth):
  * missing      : keys the warm-start does not contain at all      -> random init
  * shape_mismatch: same key but different tensor shape             -> random init
  * inherited    : name + shape match                               -> pretrained
Reports parameter counts per group, split by top-level module.
"""
import collections
import os
import sys

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
INIT = os.path.join(ROOT, "vendor/DEIMv2/ckpts/dinov3/vitl16_rdttrack_init.pth")
CUR = os.path.join(ROOT, "vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_vitl800_clean/last.pth")


def sd_of(path, key):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and key in ck:
        return ck[key]
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"]
    return ck


def strip(sd):
    out = {}
    for k, v in sd.items():
        k = k[len("module."):] if k.startswith("module.") else k
        out[k] = v
    return out


def main():
    init = strip(sd_of(INIT, "model"))
    cur = strip(sd_of(CUR, "ema" if False else "model"))

    g_inherit = collections.Counter()
    g_missing = collections.Counter()
    g_mis = collections.Counter()
    n_inherit = n_missing = n_mis = 0
    examples = collections.defaultdict(list)

    for k, v in cur.items():
        if k.startswith("criterion.") or k.startswith("postprocessor."):
            continue
        n = v.numel()
        top = ".".join(k.split(".")[:3])
        if k not in init:
            g_missing[top] += n
            n_missing += n
            if len(examples["missing"]) < 12:
                examples["missing"].append(k)
        elif tuple(init[k].shape) != tuple(v.shape):
            g_mis[top] += n
            n_mis += n
            if len(examples["mismatch"]) < 12:
                examples["mismatch"].append(f"{k}: ckpt{tuple(init[k].shape)} -> model{tuple(v.shape)}")
        else:
            g_inherit[top] += n
            n_inherit += n

    tot = max(n_inherit + n_missing + n_mis, 1)
    print(f"total params (model, excl. criterion/postproc): {tot/1e6:.1f}M")
    print(f"  inherited      : {n_inherit/1e6:7.1f}M  ({100*n_inherit/tot:5.1f}%)")
    print(f"  MISSING in ckpt: {n_missing/1e6:7.1f}M  ({100*n_missing/tot:5.1f}%)")
    print(f"  SHAPE MISMATCH : {n_mis/1e6:7.1f}M  ({100*n_mis/tot:5.1f}%)")
    print(f"  --> random-init start: {(n_missing+n_mis)/1e6:.1f}M "
          f"({100*(n_missing+n_mis)/tot:.1f}%)")

    print("\n--- top modules by RANDOM-init size ---")
    rand = collections.Counter()
    for c in (g_missing, g_mis):
        for k, v in c.items():
            rand[k] += v
    for k, v in rand.most_common(18):
        print(f"  {v/1e6:8.2f}M  {k}")
    print("\n--- example MISSING keys ---")
    for e in examples["missing"]:
        print("  ", e)
    print("\n--- example SHAPE-MISMATCH keys ---")
    for e in examples["mismatch"]:
        print("  ", e)


if __name__ == "__main__":
    main()
