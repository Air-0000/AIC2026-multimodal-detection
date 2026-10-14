#!/usr/bin/env python
"""确认 vitl16_rdttrack_init.pth 的继承源头 (带完整路径输出)。"""
import glob
import os

import torch

DEIM = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2"


def strip(ck):
    for k in ("model", "state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict) and len(ck[k]):
            return strip(ck[k])
    if isinstance(ck, dict) and "ema" in ck and isinstance(ck["ema"], dict):
        e = ck["ema"]
        return strip(e.get("module", e))
    return ck


def main():
    init = strip(torch.load(os.path.join(DEIM, "ckpts/dinov3/vitl16_rdttrack_init.pth"),
                            map_location="cpu"))
    paths = (glob.glob(os.path.join(DEIM, "outputs/*l4u3*/best_stg1.pth")) +
             glob.glob(os.path.join(DEIM, "outputs/*midboost_combo*/best_stg1.pth")) +
             [os.path.join(DEIM, "ckpts/deimv2_x_for_rdttrack_deepir.pth")])
    groups = ["encoder.", "decoder.", "backbone.ir_dinov3.", "backbone.depth_dinov3."]
    for p in paths:
        if not os.path.exists(p):
            continue
        sd = strip(torch.load(p, map_location="cpu"))
        tot = hit = 0
        for grp in groups:
            for kk in [x for x in init if x.startswith(grp)]:
                tot += 1
                if kk in sd and init[kk].shape == sd[kk].shape and \
                        torch.equal(init[kk].float(), sd[kk].float()):
                    hit += 1
        verdict = "完全继承 (逐字节相同)" if hit == tot and tot else \
                  ("无关" if hit == 0 else "部分继承")
        print(f"{p}\n   {hit}/{tot} 相同   {verdict}\n")


if __name__ == "__main__":
    main()
