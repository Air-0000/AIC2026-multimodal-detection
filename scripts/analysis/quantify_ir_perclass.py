#!/usr/bin/env python
"""逐类别量化 IR(heat/rawIR/depth) 的贡献 —— 直接回答"IR 到底帮了谁、在什么光照下帮"。

对照设计（同一权重 checkpoint0049.pth + 同一 val 配置，仅推理时遮挡一个通道）:
    midboost_combo_{cond}.log                 完整 6ch
    midboost_combo_ablate_heat_{cond}.log     遮挡 heat 通道 (填 0.5 -> 归一化后恰为 0)
    midboost_combo_ablate_rawir_{cond}.log    遮挡 rawIR 通道
    midboost_combo_ablate_depth_{cond}.log    遮挡 depth 通道

关键指标: Δ = 完整 − 遮挡。Δ>0 表示该通道对此类别有正贡献。
"""
import os
import re
import sys

AIC = "/home/constant/CodeBuddy/Dev/AIC"
LOGDIR = os.path.join(AIC, "logs")

CATS = ["person", "boat", "animal", "seat", "sign", "bicycle", "car",
        "ball", "light", "garbage can", "uav", "tricycle"]
ABL = ["heat", "rawir", "depth"]
COND = ["day", "night50", "night25", "night12", "night05",
        "night01", "night005", "night002", "night001", "night0005"]
LABEL = {"day": "day", "night50": "0.50", "night25": "0.25", "night12": "0.12",
         "night05": "0.05", "night01": "0.01", "night005": "0.005",
         "night002": "0.002", "night001": "0.001", "night0005": "0.0005"}

TOT = re.compile(r"Average Precision.*?IoU=0\.50:0\.95.*?area=\s+all"
                 r".*?maxDets=100\s*\]\s*=\s*([0-9.]+)")


def parse(path):
    """-> (per_class_ap dict, mAP) or None"""
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    m = re.search(r"per-category.*?\n((?:\s+\S.*\n){3,20})", txt)
    aps = {}
    if m:
        for ln in m.group(1).splitlines():
            mm = re.match(r"\s+(.+?)\s+AP=([0-9.]+)\s+AP50=([0-9.]+)", ln)
            if mm:
                aps[mm.group(1).strip()] = float(mm.group(2))
    t = TOT.search(txt)
    return aps, (float(t.group(1)) if t else None)


def main():
    full = {c: parse(os.path.join(LOGDIR, f"midboost_combo_{c}.log")) for c in COND}
    abl = {ab: {c: parse(os.path.join(LOGDIR, f"midboost_combo_ablate_{ab}_{c}.log"))
                for c in COND} for ab in ABL}

    have = [c for c in COND if full[c] and all(abl[ab][c] for ab in ABL)]
    print(f"可配对条件: {len(have)}/10  ({', '.join(LABEL[c] for c in have)})\n")

    # ---------- 1. 全局 mAP ----------
    print("=" * 78)
    print("【1】整体 mAP：逐光照档位，遮挡各通道后的变化")
    print("=" * 78)
    hdr = f"{'亮度':>7s} {'完整':>7s}"
    for ab in ABL:
        hdr += f" {'无'+ab:>8s}{'Δ':>8s}"
    print(hdr)
    print("-" * 78)
    for c in have:
        f = full[c][1]
        row = f"{LABEL[c]:>7s} {f:7.4f}"
        for ab in ABL:
            v = abl[ab][c][1]
            row += f" {v:8.4f}{f - v:+8.4f}"
        print(row)

    # ---------- 2. 白天逐类别 ----------
    for cond, tag in (("day", "白天 / 原图"), ("night001", "极暗 0.001")):
        print()
        print("=" * 78)
        print(f"【2】{tag}：逐类别的 IR 贡献  Δ = 完整 − 遮挡")
        print("=" * 78)
        print(f"{'类别':>12s} {'完整AP':>8s} | " +
              " ".join(f"{'无'+ab:>9s}" for ab in ABL) + " | " +
              " ".join(f"{'Δ'+ab:>8s}" for ab in ABL))
        print("-" * 78)
        rows = []
        for cat in CATS:
            fv = full[cond][0].get(cat)
            if fv is None:
                continue
            cells = []
            ds = []
            for ab in ABL:
                av = abl[ab][cond][0].get(cat)
                cells.append(av)
                ds.append(fv - av if av is not None else float("nan"))
            rows.append((cat, fv, cells, ds))
        for cat, fv, cells, ds in rows:
            cs = " ".join(f"{v:9.3f}" for v in cells)
            dsv = " ".join(f"{d:+8.3f}" for d in ds)
            print(f"{cat:>12s} {fv:8.3f} | {cs} | {dsv}")
        # 平均
        import statistics
        mean_d = [statistics.fmean([r[3][i] for r in rows]) for i in range(len(ABL))]
        print("-" * 78)
        print(f"{'类别均值':>12s} {'':8s} | {'':>9s} {'':>9s} {'':>9s} | " +
              " ".join(f"{d:+8.3f}" for d in mean_d))

    # ---------- 3. 结论 ----------
    print()
    print("=" * 78)
    print("【3】person / car 随光照的 IR(heat) 贡献")
    print("=" * 78)
    print(f"{'亮度':>7s} {'person完整':>11s} {'person无IR':>11s} {'Δ':>8s} | "
          f"{'car完整':>9s} {'car无IR':>9s} {'Δ':>8s}")
    print("-" * 78)
    for c in have:
        for cat in ("person",):
            pass
        pv = full[c][0].get("person", 0); p0 = abl["heat"][c][0].get("person", 0)
        cv = full[c][0].get("car", 0); c0 = abl["heat"][c][0].get("car", 0)
        print(f"{LABEL[c]:>7s} {pv:11.3f} {p0:11.3f} {pv - p0:+8.3f} | "
              f"{cv:9.3f} {c0:9.3f} {cv - c0:+8.3f}")

    print()
    print("=" * 78)
    print("【4】按白天 IR 贡献排序（哪些类别真的靠 IR）")
    print("=" * 78)
    day = []
    for cat in CATS:
        fv = full["day"][0].get(cat)
        if fv is None:
            continue
        day.append((cat, fv, fv - abl["heat"]["day"][0].get(cat, fv)))
    for cat, fv, d in sorted(day, key=lambda x: -x[2]):
        flag = "  <-- IR 有帮助" if d > 0.02 else ("  <-- IR 反而有害" if d < -0.005 else "")
        print(f"  {cat:14s} 完整 {fv:.3f}   Δheat {d:+.4f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
