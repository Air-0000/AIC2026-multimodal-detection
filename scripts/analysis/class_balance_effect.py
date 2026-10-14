#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化「类别不均」的两层含义, 用已有产物回答:
  (A) 指标层: COCO mAP 是 macro (每类 1/12)。若改成 micro (每框等权) 分数会变成多少?
      -> 衡量当前分数里有多少来自长尾类。
  (B) 损失层: 当前 num_boxes 归一化 = micro, 即 person 的梯度权重是 tricycle 的 199 倍。
      若改成 macro (类别均衡), 各类权重如何变? 代价是什么?
  (C) 可行性: 每个长尾类出现在多少张图上 (决定重采样/加权的可行性与过拟合风险)。
  (D) 信号 vs 噪声: 4 个模型跨模型的 per-class 离散度 vs bootstrap sd。
输出 outputs/metrics/class_balance_effect.json
用法: python scripts/analysis/class_balance_effect.py
"""
import json, os
import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VARJ = os.path.join(ROOT, "outputs/metrics/perclass_variance.json")

d = json.load(open(VARJ))
classes = d["classes"]                 # 12 类, 顺序 = sorted category id
n_gt = d["n_gt"]                       # val 上的 GT 框数
point = d["point"]                     # {model: {class: AP}}
pc = d["per_class"]                    # 以 fuse_l4u3 为参考的 bootstrap 统计
ref = d["mAP"]["reference_model"]

# ---- 训练集类别分布 + 每类出现在多少张图里 ----
tr = json.load(open(os.path.join(ROOT, "work_dir/deimv2_dataset/train/annotations/instances_train.json")))
cn = {c["id"]: c["name"] for c in tr["categories"]}
tr_n, img_set = {}, {}
for a in tr["annotations"]:
    nm = cn[a["category_id"]]
    tr_n[nm] = tr_n.get(nm, 0) + 1
    img_set.setdefault(nm, set()).add(a["image_id"])
N_tr = sum(tr_n.values())

def w(style):
    """返回 12 类权重 (已归一化到 sum=1), 顺序同 classes"""
    n = np.array([n_gt[c] for c in classes], float)
    if style == "macro":          # COCO 官方: 每类等权
        v = np.ones_like(n)
    elif style == "micro":        # 每框等权 = 按 val 支撑度加权
        v = n
    elif style == "sqrt":         # 折中
        v = np.sqrt(n)
    elif style == "inverse":      # 1/n  (越稀有越重)
        v = 1.0 / n
    else:
        raise ValueError(style)
    return v / v.sum()

print("=" * 92)
print("(A) 指标层: 同一批 per-class AP, 换不同加权方式得到的『总分』")
print("=" * 92)
models = list(point.keys())
wmacro, wmicro, wsq, winv = w("macro"), w("micro"), w("sqrt"), w("inverse")
hdr = "%-12s %9s %9s %9s %9s %9s" % ("model", "macro", "micro", "sqrt", "inverse", "macro-micro")
print(hdr)
res_A = {}
for m in models:
    ap = np.array([point[m][c] for c in classes], float)
    a = float((ap * wmacro).sum())
    b = float((ap * wmicro).sum())
    c_ = float((ap * wsq).sum())
    e = float((ap * winv).sum())
    res_A[m] = {"macro": a, "micro": b, "sqrt": c_, "inverse": e, "macro_minus_micro": a - b}
    print("%-12s %9.4f %9.4f %9.4f %9.4f %+9.4f" % (m, a, b, c_, e, a - b))

print()
print("解读: macro 是官方口径; micro 是『每框等权』。若 micro 明显更高, 说明当前分数靠头部类撑,")
print("      长尾类在拖后腿 —— 而它们在官方指标里各占 1/12。")

# ---- 各类对总分(macro)的贡献 ----
print()
print("=" * 92)
print("(A2) macro 口径下, 每类各占 1/12 = 8.333%%。按参考模型 %s 的 AP 排序:" % ref)
print("=" * 92)
rows = []
for c in classes:
    ap = point[ref][c]
    rows.append((c, n_gt[c], ap, ap / 12 * 100, 1 - ap))
rows.sort(key=lambda r: r[2])
print("%-14s %7s %8s %10s %10s" % ("class", "val_n", "AP", "贡献到mAP", "剩余空间"))
for c, n, ap, contrib, room in rows:
    print("%-14s %7d %8.4f %9.2f%% %10.4f" % (c, n, ap, contrib, room))
tail5 = {"tricycle", "ball", "boat", "uav", "garbage can"}
print()
print("5 个最少标注类合计贡献 macro mAP = %.4f  (占 %.1f%%)"
      % (sum(point[ref][c] / 12 for c in tail5), 100 * sum(point[ref][c] for c in tail5) / sum(point[ref][c] for c in classes)))

# ---- 损失层权重对比 ----
print()
print("=" * 92)
print("(B) 损失层: 当前 num_boxes 归一化 => 每类的实际梯度权重 = 训练框数占比")
print("=" * 92)
print("%-14s %8s %10s %12s %12s" % ("class", "train_n", "现状(micro)", "类别均衡", "均衡/现状"))
res_B = {}
tot = 0.0
for c in sorted(classes, key=lambda x: tr_n.get(x, 0)):
    n = tr_n.get(c, 0)
    cur = n / N_tr
    bal = 1 / 12
    ratio = bal / cur if cur > 0 else float("inf")
    res_B[c] = {"train_n": n, "n_images": len(img_set.get(c, ())),
                "cur_weight": cur, "balanced_weight": bal, "ratio": ratio}
    print("%-14s %8d %10.5f %12.5f %12.2fx" % (c, n, cur, bal, ratio))
print()
print("=> 现状下 person 的梯度权重是 tricycle 的 %.0f 倍, 而指标里两者各占 1/12。" % (tr_n["person"] / tr_n["tricycle"]))

# ---- 可行性: 每类出现在多少张图上 ----
print()
print("=" * 92)
print("(C) 可行性: 重采样 (RFS) / 类别加权 的过拟合风险")
print("=" * 92)
print("%-14s %8s %9s %12s %14s" % ("class", "train_n", "n_images", "框/图", "重复倍率上限"))
res_C = {}
for c in sorted(classes, key=lambda x: tr_n.get(x, 0)):
    n = tr_n.get(c, 0)
    ni = len(img_set.get(c, ()))
    mult = (N_tr / 12) / n if n else float("inf")
    res_C[c] = {"n_images": ni, "boxes_per_image": n / ni if ni else 0, "repeat_mult": mult}
    print("%-14s %8d %9d %12.2f %13.1fx" % (c, n, ni, n / ni if ni else 0, mult))
print()
print("=> 『重复倍率上限』= 把该类提到与平均类同等权重时, 其图像要被多看的倍数。")
print("   倍数很高 + n_images 很小 => 等价于反复看同一批图, 过拟合风险大。")

# ---- 信号 vs 噪声 ----
print()
print("=" * 92)
print("(D) 跨模型离散度 vs bootstrap 噪声 (判断哪一类的差异是真信号)")
print("=" * 92)
print("%-14s %7s %9s %11s %11s %8s" % ("class", "val_n", "AP", "跨模型sd", "bootstrap", "信号/噪声"))
res_D = {}
for c in sorted(classes, key=lambda x: -n_gt[x]):
    aps = np.array([point[m][c] for m in models], float)
    sd_cross = float(aps.std(ddof=1))
    sd_boot = float(pc[c]["boot_sd"])
    ratio = sd_cross / sd_boot if sd_boot > 0 else float("nan")
    res_D[c] = {"cross_model_sd": sd_cross, "boot_sd": sd_boot,
                "ratio": ratio, "spread": float(aps.max() - aps.min())}
    print("%-14s %7d %9.4f %11.4f %11.4f %8.2f" % (c, n_gt[c], point[ref][c], sd_cross, sd_boot, ratio))

# ---- oracle: 每类取最好模型 (仅供量化上限, 注意规则禁止集成) ----
print()
print("=" * 92)
print("(D2) 上限参考: 若每类单独取 4 个模型中最好的 (注意: 属集成, 规则第十一条禁止提交)")
print("=" * 92)
best_per = {}
for c in classes:
    vals = {m: point[m][c] for m in models}
    bm = max(vals, key=vals.get)
    best_per[c] = (bm, vals[bm])
oracle = float(np.mean([v for _, v in best_per.values()]))
for m in models:
    print("  %-12s mAP=%.4f  (距 oracle -%.4f)" % (m, res_A[m]["macro"], oracle - res_A[m]["macro"]))
print("  %-12s mAP=%.4f" % ("ORACLE", oracle))
print()
print("  单类归属: " + ", ".join("%s->%s" % (c, best_per[c][0]) for c in classes))

out = {
    "note": "macro = COCO 官方(每类等权); micro = 每框等权(当前损失归一化的口径)",
    "metric_reweighting": res_A,
    "loss_weight_current_vs_balanced": res_B,
    "repeat_sampling_feasibility": res_C,
    "signal_vs_noise": res_D,
    "oracle_per_class": {"mAP": oracle, "per_class_model": {c: best_per[c][0] for c in classes}},
    "tail5_macro_contribution": float(sum(point[ref][c] / 12 for c in tail5)),
}
with open(os.path.join(ROOT, "outputs/metrics/class_balance_effect.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print()
print("saved -> outputs/metrics/class_balance_effect.json")
