#!/usr/bin/env python3
"""Checkpoint 空间的「路径几何」：训练到底在动什么、动多远、方向是否连续。

为什么需要它：mAP 曲线只能告诉你 g(θ) 这个标量怎么变，不能告诉你 θ 走到哪了。
把 θ 的距离和方向一致性算出来，才能区分三种情况：
  (a) 正在稳定下降（位移大、方向一致 cos 高）
  (b) 在宽平盆地里来回晃（位移大但方向 cos ≈ 0，且 loss/指标几乎不变）
  (c) 已经停住（位移趋近 0）

实现要点：用 mmap=True 只把 'model' 部分按需调页，避免 4.2GB/ckpt 全量驻留。
用法:
  python checkpoint_geometry.py <out_dir> [--files a.pth b.pth ...] [--json out.json]
"""
import argparse
import glob
import gc
import json
import os
import re
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GROUPS = [
    ("rgb_vit",   lambda k: k.startswith("backbone.rgb_dinov3")),
    ("ir_vit",    lambda k: k.startswith("backbone.ir_dinov3") or k.startswith("backbone.ir_aux_proj")
                            or k.startswith("backbone.ir_norms")),
    ("depth_vit", lambda k: k.startswith("backbone.depth_dinov3") or k.startswith("backbone.depth_aux_proj")
                            or k.startswith("backbone.depth_norms")),
    ("fusion",    lambda k: k.startswith("backbone.dt_ort") or k.startswith("backbone.prompt")
                            or k.startswith("backbone.sta")),
    ("encoder",   lambda k: k.startswith("encoder.")),
    ("decoder",   lambda k: k.startswith("decoder.")),
    ("other",     lambda k: True),
]


def group_of(k):
    for name, f in GROUPS:
        if f(k):
            return name
    return "other"


def load_model_sd(path):
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = d["model"]
    le = d.get("last_epoch", None)
    del d
    gc.collect()
    # 全部 materialize 成连续 fp32（mmap 张量做减法会触发整页读入，先复制更可控）
    return {k: v for k, v in sd.items()}, le


def pair_norms(sdA, sdB):
    """返回 (总 L2, 总 ||θ||, 逐组 L2, 位移 dict, 逐组相对位移)。

    只统计浮点且全有限的参数：`decoder.anchors` 是 anchor 网格 buffer（含 inf 哨兵），
    `decoder.valid_mask` 是 bool，都不是可学习参数，必须排除，否则总距离直接变 nan/inf。
    """
    tot2 = 0.0
    base2 = 0.0
    g2 = {}
    gb2 = {}
    disp = {}
    skipped = []
    for k, va in sdA.items():
        vb = sdB.get(k)
        if vb is None or not torch.is_floating_point(va) or not torch.is_floating_point(vb):
            skipped.append(k)
            continue
        fa = va.float()
        if not torch.isfinite(fa).all():
            skipped.append(k)
            continue
        d = vb.float() - fa
        n = float(d.pow(2).sum())
        b = float(fa.pow(2).sum())
        tot2 += n
        base2 += b
        g = group_of(k)
        g2[g] = g2.get(g, 0.0) + n
        gb2[g] = gb2.get(g, 0.0) + b
        disp[k] = d
    rel = {k: (v ** 0.5) / (gb2.get(k, 0.0) ** 0.5) if gb2.get(k, 0.0) > 0 else float("nan")
           for k, v in g2.items()}
    return (tot2 ** 0.5, base2 ** 0.5,
            {k: v ** 0.5 for k, v in g2.items()}, disp, rel, skipped)


def dot(a, b):
    s = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        s += float((va * vb).sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="例如 outputs/vitl800_scratch_stageC_fuse")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(ROOT, "vendor", "DEIMv2", a.out_dir)
    if a.files:
        paths = [f if os.path.isabs(f) else os.path.join(d, f) for f in a.files]
    else:
        paths = sorted(glob.glob(os.path.join(d, "checkpoint*.pth")))
        for extra in sorted(glob.glob(os.path.join(d, "best_stg1_ep*.pth"))):
            paths.append(extra)
        paths.append(os.path.join(d, "last.pth"))
    paths = [p for p in paths if os.path.exists(p)]

    # 逐个读 last_epoch 用于排序（只读头部，mmap 不落盘全量）
    meta = []
    for p in paths:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            le = int(ck.get("last_epoch", -1))
            del ck
        except Exception as e:
            print("跳过 %s (%s)" % (p, e))
            continue
        meta.append((le, p))
    meta.sort()
    print("按 last_epoch 排序的 checkpoint：")
    for le, p in meta:
        print("   ep%-3d  %-52s %.2f GB" % (le, os.path.basename(p), os.path.getsize(p) / 2 ** 30))

    res = {"meta": [{"epoch": le, "file": os.path.basename(p)} for le, p in meta],
           "consecutive": [], "cosine": [], "extra_pairs": []}

    # ---- 滑窗：3 个 ckpt 算一次位移与方向余弦，避免全量驻留 ----
    cur_sd, cur_le = load_model_sd(meta[0][1])
    prev_disp = None
    prev_le = cur_le
    for le, p in meta[1:]:
        nxt_sd, _ = load_model_sd(p)
        tot, base, g, disp, rel, skipped = pair_norms(cur_sd, nxt_sd)
        de = le - cur_le
        rec = dict(from_ep=cur_le, to_ep=le, de=de, l2=tot, theta_norm=base,
                   rel=tot / base, per_epoch=tot / de,
                   per_group=g, per_group_rel=rel,
                   per_group_per_epoch={k: v / de for k, v in g.items()},
                   n_skipped=len(skipped), skipped=skipped[:5])
        if prev_disp is not None:
            na = dot(prev_disp, prev_disp) ** 0.5
            nb = dot(disp, disp) ** 0.5
            cos = dot(prev_disp, disp) / (na * nb) if na > 0 and nb > 0 else float("nan")
            rec["cos_prev"] = cos
            res["cosine"].append(dict(prev=f"ep{prev_le}", cur=f"ep{cur_le}", nxt=f"ep{le}", cos=cos))
        res["consecutive"].append(rec)
        prev_disp = disp
        prev_le = cur_le
        del cur_sd
        gc.collect()
        cur_sd, cur_le = nxt_sd, le

    # ---- 额外关心的配对：ep18(最佳) vs 之后 ----
    ep_best = None
    for le, p in meta:
        if "ep18" in os.path.basename(p):
            ep_best = (le, p)
    pairs = []
    if ep_best:
        last = meta[-1]
        second_last = meta[-2]
        pairs = [(os.path.basename(ep_best[1]), os.path.basename(second_last[1]), second_last[1]),
                 (os.path.basename(ep_best[1]), os.path.basename(last[1]), last[1])]
    for a1, b1, bp in pairs:
        ap_sd, _ = load_model_sd(os.path.join(d, a1))
        bp_sd, _ = load_model_sd(bp)
        tot, base, g, _, rel, sk = pair_norms(ap_sd, bp_sd)
        res["extra_pairs"].append(dict(a=a1, b=b1, l2=tot, rel=tot / base,
                                       per_group=g, per_group_rel=rel, n_skipped=len(sk)))
        del ap_sd, bp_sd
        gc.collect()

    # ---- 打印 ----
    print("\n" + "=" * 100)
    print("① 相邻 checkpoint 的位移（θ 空间走了多远）")
    if res["consecutive"]:
        print("   （已排除 %d 个不可学习/含 inf 的 buffer，例如 %s）"
              % (res["consecutive"][0]["n_skipped"], res["consecutive"][0]["skipped"]))
    print("%-9s %4s %12s %12s %10s %12s" % ("区间", "Δep", "||Δθ||", "||θ||", "相对", "||Δθ||/ep"))
    for r in res["consecutive"]:
        print("%-9s %4d %12.4f %12.4f %9.2e %12.4f"
              % ("ep%d→%d" % (r["from_ep"], r["to_ep"]), r["de"], r["l2"],
                 r["theta_norm"], r["rel"], r["per_epoch"]))

    gs = [n for n, _ in GROUPS]
    print("\n② 分组位移（||Δθ_group|| / ||θ_group||，= 该组「走了自身的百分之几」）")
    print("%-9s" % "区间" + "".join("%13s" % g for g in gs if g != "other"))
    for r in res["consecutive"]:
        line = "%-9s" % ("ep%d→%d" % (r["from_ep"], r["to_ep"]))
        for g in gs:
            if g == "other":
                continue
            line += "%13.3e" % r["per_group_rel"].get(g, float("nan"))
        print(line)

    print("\n③ 方向一致性 cos(D_t, D_{t-1})  —— 越接近 1 越像「沿一条路走」，≈0 像「随机晃」")
    for c in res["cosine"]:
        print("   %s→%s 与 %s→%s : cos = %+.3f" % (c["prev"], c["cur"], c["cur"], c["nxt"], c["cos"]))

    if res["extra_pairs"]:
        print("\n④ 关键配对：最佳点(ep18) 到后面各点的距离")
        for r in res["extra_pairs"]:
            print("   %s → %s : ||Δθ|| = %.4f   相对 = %.3e" % (r["a"], r["b"], r["l2"], r["rel"]))
            print("        分组: %s" % {k: round(v, 3) for k, v in sorted(r["per_group"].items())})

    if a.json:
        out = a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("\n写出 %s" % out)


if __name__ == "__main__":
    main()
