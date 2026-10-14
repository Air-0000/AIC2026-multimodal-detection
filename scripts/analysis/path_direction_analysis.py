#!/usr/bin/env python3
"""训练轨迹的「方向一致性」分析：逐模块看位移向量是在前进还是在来回横跳。

为什么需要它：
  checkpoint_geometry.py 只算位移的**长度**（||Δθ||），长度大不能区分
  「沿一个方向稳步走」和「在原地来回横跳」——两者都能给出同样的 ||Δθ||。
  本脚本补上**方向**：

    cos(Δθ_t, Δθ_{t+1})   相邻两个区间的位移夹角余弦
        +1  → 一直在同一条直线上前进（良态、稳定下降）
         0  → 位移方向互不相关（随机游走 / 噪声主导）
        <0  → 掉头，来回横跳（峡谷里震荡，或该模块 lr 相对曲率过大）

    straightness = |Σ Δθ| / Σ |Δθ|     净位移 / 路径总长
        1.0 = 全程走直线；0.0 = 绕回原点。

逐模块（rgb_vit / ir_vit / depth_vit / fusion / encoder / decoder）算，
因为不同模块的自由度差异巨大（fusion 只占 1.7M，rgb_vit 占 300M），
混在一起的平均会被最大的模块吃掉。

内存策略：三个 checkpoint 用 mmap 打开，逐个 key 做 float 转换后立即丢弃，
峰值只有「一个 key 的两份 float 副本」，不复制整个 state_dict。

用法：
  python scripts/analysis/path_direction_analysis.py <ckpt_dir> --json outputs/path_direction.json
"""
import argparse
import gc
import json
import os
import sys

import torch

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
]
GNAMES = [g for g, _ in GROUPS] + ["ALL"]


def group_of(k):
    for name, f in GROUPS:
        if f(k):
            return name
    return None


def load_sd(path):
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return d["model"], d.get("last_epoch", None)


def is_param(k, v):
    """排除 buffer：anchors 含 inf 哨兵、valid_mask 是 bool。"""
    if not hasattr(v, "dtype"):
        return False
    if v.dtype == torch.bool:
        return False
    if "anchors" in k or "valid_mask" in k:
        return False
    return v.dtype.is_floating_point


def triple_dirs(pa, pb, pc):
    """返回 {group: dict(dot, n1, n2, nb, n1_1, n2_1)}；n? 为平方量。

    d1 = θ_b − θ_a, d2 = θ_c − θ_b。
    n1/n2 = |d1|²/|d2|²；nb = |θ_b|²（用于算相对步长）。
    n1_1 / n2_1 = 把 d 内积拆成两份（d·d = |da|²+|db|²−2·da·db），
    供「净位移 / 路径长」直度指标使用。
    """
    acc = {g: dict(dot=0.0, n1=0.0, n2=0.0, nb=0.0, s1=0.0, s2=0.0) for g in GNAMES}
    sda, sdb, sdc = pa, pb, pc
    for k, va in sda.items():
        vb = sdb.get(k)
        vc = sdc.get(k)
        if vb is None or vc is None or not is_param(k, va):
            continue
        g = group_of(k) or "ALL"
        fa = va.float()
        fb = vb.float()
        fc = vc.float()
        d1 = fb - fa
        d2 = fc - fb
        # 逐 key 的平方量
        n1 = float(d1.pow(2).sum())
        n2 = float(d2.pow(2).sum())
        nb = float(fb.pow(2).sum())
        dt = float((d1 * d2).sum())
        for gg in (g, "ALL"):
            a = acc[gg]
            a["dot"] += dt
            a["n1"] += n1
            a["n2"] += n2
            a["nb"] += nb
        del fa, fb, fc, d1, d2
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--files", nargs="*", default=None,
                    help="显式指定顺序的文件名；默认用常见 checkpoint 排序")
    ap.add_argument("--theta-ref", default=None,
                    help="算相对步长时用的 ||θ|| 来源（默认用每个区间的起点）")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = a.ckpt_dir
    if a.files:
        spec = [(os.path.join(d, f), None) for f in a.files]
    else:
        names = [f for f in os.listdir(d) if f.endswith(".pth")]
        # ⚠ 关键：必须按 checkpoint 自己记的 last_epoch 排序，不能按文件名。
        #   目录里同时有 checkpoint0029.pth / last.pth / best_stg1_ep*.pth，
        #   best_* 是**旧 epoch 的副本**，按文件名排会把时间轴打乱，算出
        #   cos 为 -1、相对步长为负的伪结果。
        spec = [(os.path.join(d, f), None) for f in names]

    # 读真实 epoch 并按 epoch 排序；同 epoch 保留一个（优先 last.pth → checkpointNNNN）
    keyed = []
    for p, e in spec:
        if e is None:
            _, e = load_sd(p)
        keyed.append((e if e is not None else -1, p))
    prio = lambda p: (0 if os.path.basename(p) == "last.pth" else
                      1 if os.path.basename(p).startswith("checkpoint") else 2)
    best_at = {}
    for e, p in keyed:
        cur = best_at.get(e)
        if cur is None or prio(p) < prio(cur):
            best_at[e] = p
    spec = sorted(((best_at[e], e) for e in best_at), key=lambda t: t[1])

    if len(spec) < 3:
        print("需要 ≥3 个 checkpoint，实际 %d" % len(spec))
        return 1

    print("轨迹点（按真实 last_epoch 排序）：")
    for p, e in spec:
        print("   ep%-4s %s" % (e, os.path.basename(p)))

    print("=" * 100)
    print("轨迹方向一致性：cos(Δθ_t, Δθ_{t+1})  (逐模块)")
    print("=" * 100)
    print("  +1 = 稳定同向前进   ~0 = 方向随机（噪声主导）   <0 = 掉头横跳")
    print("-" * 100)

    res = {"spec": [{"epoch": e, "file": os.path.basename(p)} for p, e in spec],
           "triples": []}
    for idx in range(len(spec) - 2):
        (pa, ea), (pb, eb), (pc, ec) = spec[idx], spec[idx + 1], spec[idx + 2]
        try:
            sda, lea = load_sd(pa)
            sdb, leb = load_sd(pb)
            sdc, lec = load_sd(pc)
        except Exception as ex:  # noqa: BLE001
            print("加载失败 %s: %s" % (pa, ex))
            continue
        ea = lea if lea is not None else ea
        eb = leb if leb is not None else eb
        ec = lec if lec is not None else ec
        acc = triple_dirs(sda, sdb, sdc)
        del sda, sdb, sdc
        gc.collect()

        tag = "ep%s→%s→%s" % (ea, eb, ec)
        rec = {"tag": tag, "eps": [ea, eb, ec], "groups": {}}
        for g in GNAMES:
            x = acc[g]
            if x["n1"] <= 0 or x["n2"] <= 0:
                continue
            cosd = x["dot"] / ((x["n1"] ** 0.5) * (x["n2"] ** 0.5))
            # 直度：|d1+d2| / (|d1|+|d2|)
            net = (x["n1"] + x["n2"] + 2 * x["dot"]) ** 0.5
            straight = net / ((x["n1"] ** 0.5) + (x["n2"] ** 0.5))
            de = (eb - ea + ec - eb) or 1
            rec["groups"][g] = dict(cos=cosd, straight=straight,
                                    rel_step=((x["n1"] + x["n2"]) ** 0.5) / de
                                    / (x["nb"] ** 0.5 if x["nb"] > 0 else float("nan")))
        res["triples"].append(rec)

        hdr = "%-18s" % tag
        print("\n" + hdr)
        print("  %-10s %9s %11s %14s" % ("group", "cos", "直度", "相对步长/ep"))
        for g in GNAMES:
            v = rec["groups"].get(g)
            if not v:
                continue
            print("  %-10s %+9.3f %11.3f %14.3e"
                  % (g, v["cos"], v["straight"], v["rel_step"]))

    print("\n" + "=" * 100)
    if res["triples"]:
        print("汇总（各区间 cos 的均值；<0 的模块就是横跳的那部分）")
        print("  %-10s %10s %10s" % ("group", "mean cos", "mean 直度"))
        for g in GNAMES:
            vals = [t["groups"][g]["cos"] for t in res["triples"] if g in t["groups"]]
            strs_ = [t["groups"][g]["straight"] for t in res["triples"] if g in t["groups"]]
            if not vals:
                continue
            print("  %-10s %+10.3f %10.3f" % (g, sum(vals) / len(vals), sum(strs_) / len(strs_)))
    print("=" * 100)

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("已写 %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
