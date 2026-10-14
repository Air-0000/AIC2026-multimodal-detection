#!/usr/bin/env python
"""追查 vitl16_rdttrack_init.pth 的真实构成: 里面每一块究竟继承自哪个模型。

方法: 对候选源模型做逐键 shape+tensor 指纹比对。若某模块的 tensor 与源模型
逐字节相等 -> 它就是拷贝来的 (即"用训练好的权重"), 否则是随机初始化。
"""
import os
import sys

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIM = os.path.join(ROOT, "vendor", "DEIMv2")

INIT = os.path.join(DEIM, "ckpts/dinov3/vitl16_rdttrack_init.pth")
CANDIDATES = [
    ("COCO detector (deimv2_dinov3_l_coco.pth)", os.path.join(ROOT, "deimv2_dinov3_l_coco.pth")),
    ("midboost_combo best", os.path.join(
        DEIM, "outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo",
        "best_stg1.pth")),
    ("midboost_l4u3 best", os.path.join(DEIM, "outputs/*l4u3*/best_stg1.pth")),
    ("deimv2_x_for_rdttrack_deepir.pth", os.path.join(DEIM, "ckpts/deimv2_x_for_rdttrack_deepir.pth")),
]

PREFIXES = [
    "encoder.", "decoder.",
    "backbone.rgb_dinov3.", "backbone.ir_dinov3.", "backbone.depth_dinov3.",
]


def strip(ck):
    for k in ("model", "state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict) and len(ck[k]):
            return strip(ck[k])
    if isinstance(ck, dict) and "ema" in ck and isinstance(ck["ema"], dict):
        e = ck["ema"]
        return strip(e.get("module", e))
    return ck


def load(p):
    p = os.path.realpath(p)
    if not os.path.exists(p):
        return None
    return strip(torch.load(p, map_location="cpu"))


def stats(sd):
    """per-prefix: (n_keys, n_params)"""
    out = {}
    for k, v in sd.items():
        top = None
        for p in PREFIXES:
            if k.startswith(p):
                top = p
                break
        if top is None:
            top = "other"
        n, c = out.get(top, (0, 0))
        out[top] = (n + 1, c + v.numel())
    return out


def main():
    if not os.path.exists(INIT):
        print("missing init:", INIT)
        return
    init = load(INIT)
    print(f"=== {os.path.basename(INIT)} 构成 ===")
    tot = sum(v.numel() for v in init.values())
    print(f"total params = {tot/1e6:.1f}M, keys = {len(init)}\n")
    for p, (n, c) in sorted(stats(init).items(), key=lambda x: -x[1][1]):
        print(f"  {p:28s} keys={n:6d}  params={c/1e6:8.2f}M  ({100*c/tot:5.1f}%)")

    import glob
    cands = []
    for name, pat in CANDIDATES:
        m = glob.glob(pat) if "*" in pat else [pat]
        for f in m:
            cands.append((os.path.basename(f), f))
    print("\n=== 逐模块溯源 (与候选源做 tensor 指纹比对) ===")
    cache = {}
    for name, path in cands:
        sd = load(path)
        if sd is None:
            print(f"  [skip] {name}: not found")
            continue
        cache[name] = sd
        print(f"\n  --- 源: {name} ---")
        for pref in ["encoder.", "decoder.", "backbone.ir_dinov3.", "backbone.depth_dinov3."]:
            keys = [k for k in init if k.startswith(pref)]
            if not keys:
                continue
            hit = miss = nokey = 0
            for k in keys:
                if k not in sd:
                    nokey += 1
                    continue
                a, b = init[k], sd[k]
                if a.shape != b.shape:
                    nokey += 1
                    continue
                if torch.equal(a.float(), b.float()):
                    hit += 1
                else:
                    miss += 1
            tag = "== 完全继承" if miss == 0 and hit else \
                  ("-- 无关" if hit == 0 else "~~ 部分继承")
            print(f"    {pref:26s} hit={hit:5d} diff={miss:5d} n/a={nokey:5d}  {tag}")


if __name__ == "__main__":
    sys.exit(main())
