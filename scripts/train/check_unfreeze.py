#!/usr/bin/env python3
"""检查 RDTTrack 的 "部分解冻 RGB 主干" 是否真的生效。

check_configs.py 只验证配置能加载, 不验证解冻链路。这里补三件事:
  1. 逐 executor(顶层子模块)打印 可训练/总 参数量, 确认 RGB 主干真的有一部分是 trainable
  2. 按 block 粒度打印 requires_grad, 确认解冻的是 *最后 N 个* block (不是随机几个)
  3. 用 det_solver 里同一套正则把参数分到优化器组, 统计每组参数量与实际 lr
     —— 解冻出的参数若被 base lr(3e-4) 接住就是灾难: 预训练先验会被几步打散

用法:
    conda run -n deimv2 python scripts/train/check_unfreeze.py \
        --cfg vendor/DEIMv2/configs/deimv2/xxx.yml
"""
import argparse
import re
import sys
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIM = os.path.join(ROOT, "vendor/DEIMv2")
sys.path.insert(0, DEIM)
sys.path.insert(0, os.path.join(ROOT, "scripts/lib"))  # nightify_aic

import torch  # noqa: E402
from engine.core import YAMLConfig  # noqa: E402


def human(n):
    return f"{n/1e6:8.2f}M" if n >= 1e6 else f"{n/1e3:8.1f}K"


def count_groups(named_params, groups, base_lr, base_wd):
    """复刻 det_solver 的组路由: 先匹配到的专用组优先, 其余落到底。"""
    compiled = [(g.get("params"), g) for g in groups]
    stat = {}
    used = set()
    for name, p in named_params:
        if not p.requires_grad:
            continue
        hit = None
        for pat, g in compiled:
            if isinstance(pat, str) and re.match(pat, name):
                hit = g
                break
        if hit is None:
            lr, wd, tag = base_lr, base_wd, "(fallthrough/base)"
        else:
            lr = hit.get("lr", base_lr)
            wd = hit.get("weight_decay", base_wd)
            tag = hit.get("params")[:46]
        used.add(name)
        k = (tag, lr, wd)
        s = stat.setdefault(k, [0, 0])
        s[0] += p.numel()
        s[1] += 1
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.path.join(
        DEIM, "configs/deimv2/deimv2_dinov3_l_aic_vitl800_rdttrack_deepir_aug.yml"))
    args = ap.parse_args()

    cfg = YAMLConfig(args.cfg)
    n_unfreeze = cfg.yaml_cfg.get("DINOv3STAsRDTTrack", {}).get("unfreeze_rgb_stages", 0)
    print(f"config: {os.path.basename(args.cfg)}")
    print(f"unfreeze_rgb_stages = {n_unfreeze}")
    print("=" * 78)

    model = cfg.model
    bb = None
    for m in model.modules():
        if m.__class__.__name__ == "DINOv3STAsRDTTrack":
            bb = m
            break
    if bb is None:
        print("!! 没找到 DINOv3STAsRDTTrack")
        return 1

    # ---- 1. 顶层可训练量 ----
    print("[1] 顶层模块可训练参数")
    tot_tr = 0
    for name, mod in bb.named_children():
        n = sum(p.numel() for p in mod.parameters())
        tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        tot_tr += tr
        if n == 0:
            continue
        flag = "" if tr == 0 else ("  <-- 部分解冻" if tr < n else "  <-- 全训")
        print(f"    {name:24s} trainable {human(tr):>10s} / {human(n):>10s}{flag}")

    n_all = sum(p.numel() for p in bb.parameters())
    print(f"    {'TOTAL':24s} trainable {human(tot_tr):>10s} / {human(n_all):>10s}"
          f"   ({100*tot_tr/n_all:.1f}%)")

    # ---- 2. block 粒度 ----
    print("\n[2] RGB 主干 block 的 requires_grad")
    blocks = bb.rgb_dinov3.blocks
    line = []
    for i, b in enumerate(blocks):
        on = any(p.requires_grad for p in b.parameters())
        line.append("X" if on else ".")
    print(f"    blocks {len(blocks)}: {''.join(line)}   (X=可训练, .=冻结)")
    frozen_tr = [i for i, b in enumerate(blocks)
                 if any(p.requires_grad for p in b.parameters())]
    print(f"    可训练 block 索引: {frozen_tr}")
    has_norm = False
    nrm = getattr(bb.rgb_dinov3, "norm", None)
    if nrm is not None:
        has_norm = any(p.requires_grad for p in nrm.parameters())
    print(f"    final norm 可训练: {has_norm}")

    # ---- 3. 优化器分组 (用真实 optimizer 构建, 权威) ----
    print("\n[3] 优化器分组 (由 YAMLConfig.optimizer 真实构建)")
    try:
        optim = cfg.optimizer
    except Exception as e:
        print(f"    [ERR] 构建 optimizer 失败: {e}")
        return 1
    opt_cfg = cfg.yaml_cfg.get("optimizer", {})
    base_wd = opt_cfg.get("weight_decay")
    rgb_in_base = False
    for gi, g in enumerate(optim.param_groups):
        lr = g.get("lr")
        wd = g.get("weight_decay", base_wd)
        n = sum(p.numel() for p in g["params"])
        sample = None
        gset = set(g["params"])
        for nm, p in bb.named_parameters():
            if sample is None and p in gset and "rgb_dinov3" in nm:
                sample = nm
                break
        tag = sample if sample else "(无 rgb 参数)"
        if "rgb_dinov3" in tag:
            rgb_in_base = True
        print(f"    g{gi:>2d} {human(n):>10s}  lr={lr:<10.2e}  wd={wd:<10g}  {tag[:52]}")

    # ---- 判定 ----
    print("\n" + "=" * 78)
    rgb_tr = sum(p.numel() for p in bb.rgb_dinov3.parameters() if p.requires_grad)
    if rgb_tr == 0:
        print("[FAIL] RGB 主干仍然全冻结 —— 部分解冻没生效")
        return 1
    if frozen_tr != list(range(len(blocks) - min(n_unfreeze, len(blocks)), len(blocks))):
        print("[WARN] 可训练 block 不是连续的最后 N 个")
        return 1
    # 解冻出的 RGB 参数必须被 LLRD 低-LR 组接住: 其所在组 lr 必须 <= 2e-5,
    # 不能吃 base lr(3e-4) 或遗留的 2e-5 兜底组。
    gset_of = {}
    for g in optim.param_groups:
        gset_of.setdefault(id(g), set(g["params"]))
    rgb_lr = {}
    bad = []
    for nm, p in bb.named_parameters():
        if not p.requires_grad or "rgb_dinov3" not in nm:
            continue
        grp = next((g for g in optim.param_groups if p in gset_of[id(g)]), None)
        if grp is None:
            bad.append((nm, None)); continue
        lrv = grp.get("lr")
        rgb_lr[nm] = lrv
        if lrv is None or lrv > 2e-5 + 1e-12:
            bad.append((nm, lrv))
    if bad:
        print("[FAIL] 有 RGB 解冻参数未被 LLRD 低-LR 组接住:")
        for nm, lrv in bad[:6]:
            print(f"    {nm}  lr={lrv}")
        return 1
    lrs = sorted(set(round(v, 9) for v in rgb_lr.values()))
    print(f"[OK] RGB 主干已解冻 {human(rgb_tr)} 个参数, 全部由 LLRD 接住")
    print(f"     解冻 RGB 参数的实际 lr 集合 = {[f'{x:.2e}' for x in lrs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
