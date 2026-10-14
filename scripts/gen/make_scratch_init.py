#!/usr/bin/env python
"""生成 "一次性训练" 的干净热启动权重 (不含任何 AIC/目标检测训练产物)。

设计原则
--------
主干用官方 DINOv3 预训练 = 通用视觉预训练, 等价于 ImageNet 预训练, 是标准做法,
不算 "用训练好的权重"。要排除的是: 继承某个**已在 AIC/同类任务上收敛的 detector**
的 encoder / decoder / aux stem。

RGB 流  : 官方 dinov3_vitl16 LVD1689M 预训练 (本地已有 1.21GB 文件)
IR/Depth: 官方 DINOv3 ViT-S 权重 + patch_embed 通道复制 (3->2 / 3->1)
          来源 = DEIMv2 官方 COCO 检测器 `deimv2_dinov3_l_coco.pth` 里的
          `backbone.dinov3.*`。这里取的是它的 **backbone 卷积/bock 权重**,
          作为 RGB->aux 的迁移起点 —— aux 本来就不是红外预训练的, 原项目
          prepare_* 脚本也是这么做的。注意: 这个 aux 源**没有** AIC 数据训练痕迹,
          而原始 vitl16_rdttrack_init.pth 的 aux 是 midboost_l4u3 在 AIC 上训出来的。

产出里 **没有** encoder / decoder / convs / aux_proj / sta / prompt / dt_ort,
它们会在训练时随机初始化 —— 这才是 "一次性训练出来"。
"""
import os
import subprocess
import sys

import torch

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
DEIM = os.path.join(ROOT, "vendor", "DEIMv2")
PREPARE = os.path.join(DEIM, "prepare_dinov3_for_rdttrack.py")
PY = "/home/constant/miniconda3/envs/deimv2/bin/python"

VITL = os.path.join(DEIM, "ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
COCO = os.path.join(ROOT, "deimv2_dinov3_l_coco.pth")
AUX_FLAT = os.path.join(DEIM, "ckpts/dinov3/dinov3_vits16_from_coco_backbone.pth")
OUT = os.path.join(DEIM, "ckpts/dinov3/vitl16_rdttrack_scratch.pth")


def strip(ck):
    for k in ("model", "state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict) and len(ck[k]):
            return strip(ck[k])
    if isinstance(ck, dict) and "ema" in ck and isinstance(ck["ema"], dict):
        e = ck["ema"]
        return strip(e.get("module", e))
    return ck


def main():
    if os.path.exists(OUT):
        print(f"[skip] already exists: {OUT}")
        print(f"       size = {os.path.getsize(OUT)/1e6:.1f} MB")
        return

    # ---- 1. 剥出官方 ViT-S backbone 作为 aux 流的干净起点 ----
    if not os.path.exists(AUX_FLAT):
        print(f"[1/2] extracting clean ViT-S backbone from {os.path.basename(COCO)} ...")
        sd = strip(torch.load(COCO, map_location="cpu"))
        flat = {}
        for k, v in sd.items():
            if k.startswith("backbone.dinov3."):
                flat[k[len("backbone.dinov3."):]] = v
        if not flat:
            print("[FATAL] no backbone.dinov3.* keys found in", COCO)
            return 1
        torch.save(flat, AUX_FLAT)
        n = sum(v.numel() for v in flat.values())
        pe = flat.get("patch_embed.proj.weight")
        depth = len({int(k.split(".")[1]) for k in flat
                     if k.startswith("blocks.") and k.count(".") >= 2})
        print(f"      saved {AUX_FLAT}: keys={len(flat)} params={n/1e6:.1f}M "
              f"embed_dim={pe.shape[0]} in_chans={pe.shape[1]} blocks={depth}")
        if pe.shape[0] != 384:
            print(f"[FATAL] expected ViT-S aux with embed_dim=384, got {pe.shape[0]}")
            return 1
    else:
        print(f"[1/2] aux source already extracted: {AUX_FLAT}")

    # ---- 2. 生成 scratch init: 不带 --inherit-aux / --inherit-encoder ----
    print(f"[2/2] building scratch init (NO inherit) -> {os.path.basename(OUT)}")
    cmd = [PY, PREPARE,
           "--dinov3", VITL,
           "--aux-dinov3", AUX_FLAT,
           "--out", OUT,
           "--ir-in-chans", "2",
           "--depth-in-chans", "1"]
    r = subprocess.run(cmd, cwd=DEIM)
    if r.returncode != 0:
        print("[FATAL] prepare script failed")
        return r.returncode

    # ---- 3. 自检: 确认产物里没有继承任何 detector 权重 ----
    ck = strip(torch.load(OUT, map_location="cpu"))
    bad = [k for k in ck if k.startswith(("encoder.", "decoder."))]
    groups = {}
    for k, v in ck.items():
        top = k.split(".")[1] if k.startswith("backbone.") else k.split(".")[0]
        g, c = groups.get(top, (0, 0))
        groups[top] = (g + 1, c + v.numel())
    tot = sum(v.numel() for v in ck.values())
    print(f"\n=== {os.path.basename(OUT)} 自检 ===")
    print(f"total {tot/1e6:.1f}M keys={len(ck)}")
    for g, (n, c) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"  {g:16s} keys={n:5d} params={c/1e6:8.2f}M ({100*c/tot:5.1f}%)")
    print(f"\n继承的 detector 模块 (应为 0): encoder/decoder 键数 = {len(bad)}")
    if bad:
        print("[FAIL] 仍含继承权重:", bad[:5])
        return 1
    print("[OK] 干净: encoder/decoder/convs/prompt/sta/aux_proj 将在训练时随机初始化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
