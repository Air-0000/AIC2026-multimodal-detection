#!/usr/bin/env python3
"""评测方差治理：量化「扩评测集」与「平滑被评估对象」各自的收益。

两个必须分开的东西（混淆它们会得出完全错误的优化方向）：

  sigma_A 抽样方差   = 换一批评测图带来的不确定度。只有它随 N 增大而下降。
                      决定「这个分数能不能代表它在测试集上的真实水平」。
  sigma_B 轨迹抖动   = 固定同一批图、权重在训练中变化带来的 mAP 变动。
                      与图的数量无关（图固定 => 读数是确定的）。决定「曲线抖不抖」。

本脚本实测 sigma_A(N)：从 val200 里做**无放回**子抽样，用
pycocotools/ faster_coco_eval 重算 COCO mAP，得到 Var(N)。

有限总体抽样理论（不放回，总体 M=200）：
    Var_sub(N) = sigma_pop^2 * (1/N - 1/M)
=>  sigma_pop^2 = Var_sub(N) / (1/N - 1/M)
=>  对 N' 张独立图（无限总体）： Var(N') = sigma_pop^2 / N'

用多个 N 交叉验证这条 1/sqrt(N) 律是否成立。若不成立（长尾 + AP 是排序统计量，
很可能偏离），那「扩图就能降方差」的直觉就要打折。

用法（需要 deimv2 环境，无需 GPU）：
  <deimv2 python> scripts/analysis/val_variance_reduction.py --iters 80
"""
import argparse
import contextlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
VENDOR = os.path.join(ROOT, "vendor", "DEIMv2")
sys.path.insert(0, VENDOR)

from faster_coco_eval import COCO, COCOeval_faster  # noqa: E402

ANN = os.path.join(ROOT, "work_dir/deimv2_dataset/valid/annotations/instances_val.json")
DETS = os.path.join(ROOT, "outputs/analysis/_dets")
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gt():
    base = json.load(open(ANN))
    imgs = base["images"]
    cats = base["categories"]
    anns_by_img = {}
    for a in base["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    return imgs, cats, anns_by_img


def load_prep(dets_dir, score_floor):
    prep = {}
    for p in sorted(os.path.join(dets_dir, x) for x in os.listdir(dets_dir)
                    if x.endswith(".pkl")):
        with open(p, "rb") as f:
            dets = pickle.load(f)["dets"]
        per_img = {}
        for iid, d in dets.items():
            s = d["scores"]
            m = s >= score_floor
            if not m.any():
                per_img[iid] = np.zeros((0, 6))
                continue
            b = d["boxes"][m]
            xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            per_img[iid] = np.concatenate(
                [xywh, s[m, None], d["labels"][m, None].astype(np.float64)], 1)
        prep[os.path.basename(p)[:-4]] = per_img
    return prep


class Evaluator(object):
    def __init__(self, imgs, cats, anns_by_img):
        self.imgs, self.cats, self.anns_by_img = imgs, cats, anns_by_img
        self.devnull = open(os.devnull, "w")

    def gt_dict(self, ids):
        images, annotations = [], []
        aid = 1
        for k, old in enumerate(ids):
            src = self.imgs[old]
            nid = k + 1
            images.append({"id": nid, "width": src["width"], "height": src["height"],
                           "file_name": src["file_name"]})
            for a in self.anns_by_img.get(src["id"], []):
                annotations.append({"id": aid, "image_id": nid,
                                    "category_id": a["category_id"],
                                    "bbox": list(a["bbox"]), "area": a["area"],
                                    "iscrowd": 0})
                aid += 1
        return {"images": images, "annotations": annotations, "categories": self.cats}

    @staticmethod
    def dt_list(per_img, imgs, ids):
        out = []
        for k, old in enumerate(ids):
            arr = per_img.get(imgs[old]["id"])
            if arr is None or len(arr) == 0:
                continue
            nid = k + 1
            for row in arr:
                out.append({"image_id": nid, "category_id": int(row[5]),
                            "bbox": [float(row[0]), float(row[1]),
                                     float(row[2]), float(row[3])],
                            "score": float(row[4])})
        return out

    def score(self, per_img, ids, want_per_class=False):
        gt = COCO(self.gt_dict(ids))
        dt = self.dt_list(per_img, self.imgs, ids)
        n_eff = len({a["category_id"] for a in gt.dataset["annotations"]})
        if not dt:
            return 0.0, n_eff, [np.nan] * len(self.cats)
        cd = gt.loadRes(dt)
        ev = COCOeval_faster(gt, cd, iouType="bbox", separate_eval=True)
        with contextlib.redirect_stdout(self.devnull):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        pcs = []
        if want_per_class:
            # area=all(0), maxDets=100(index 2)；无 GT 的类为 NaN（不可记 0）
            P = ev.eval["precision"]  # [T, R, K, A, M]
            for k in range(P.shape[2]):
                col = P[:, :, k, 0, 2]
                ok = col >= 0
                pcs.append(float(col[ok].mean()) if ok.any() else np.nan)
        return float(ev.stats[0]), n_eff, pcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-floor", type=float, default=0.005)
    ap.add_argument("--sizes", default="25,50,100,150")
    ap.add_argument("--predict", default="200,400,600,1000,1800,2000")
    ap.add_argument("--out-json", default=os.path.join(ROOT, "outputs/metrics/val_variance_reduction.json"))
    a = ap.parse_args()

    imgs, cats, anns_by_img = load_gt()
    M = len(imgs)
    K = len(cats)
    prep = load_prep(DETS, a.score_floor)
    names = sorted(prep.keys())
    ev_ = Evaluator(imgs, cats, anns_by_img)
    print("models = %s   M(val images) = %d   K = %d" % (names, M, K))

    # ---- point estimate on full val200 ----
    full = np.arange(M)
    point = {}
    for nm in names:
        v, n_eff, pcs = ev_.score(prep[nm], full, want_per_class=True)
        point[nm] = dict(mAP=v, n_cat_eff=n_eff, per_class=pcs)
        print("  %-14s mAP=%.4f  (有GT类 %d/%d)" % (nm, v, n_eff, K))

    sizes = [int(x) for x in a.sizes.split(",") if int(x) < M]
    predict = [int(x) for x in a.predict.split(",")]

    # ---- subsample experiment ----
    rng = np.random.default_rng(a.seed)
    rows = []
    print("\n=== 无放回子抽样：Var(N) 实测  (每格 %d 次) ===" % a.iters)
    print("%-14s %5s %8s %8s %10s %10s %8s"
          % ("model", "N", "mean", "sd", "sd(pt)", "Var", "有GT类"))
    for N in sizes:
        for nm in names:
            vals = np.zeros(a.iters)
            cats_eff = np.zeros(a.iters)
            pc_mat = np.full((a.iters, K), np.nan)
            for it in range(a.iters):
                idx = np.sort(rng.choice(M, size=N, replace=False))
                v, n_eff, pcs = ev_.score(prep[nm], idx, want_per_class=True)
                vals[it] = v
                cats_eff[it] = n_eff
                pc_mat[it] = pcs
            sd = float(vals.std(ddof=1))
            var = sd * sd
            # sigma_pop^2 = Var_sub / (1/N - 1/M)
            s2_pop = var / (1.0 / N - 1.0 / M)
            rows.append(dict(model=nm, N=N, mean=float(vals.mean()), sd=sd,
                             var=var, s2_pop=s2_pop, cat_eff=float(cats_eff.mean()),
                             pc_sd=[float(np.nanstd(pc_mat[:, k], ddof=1))
                                    for k in range(K)]))
            print("%-14s %5d %8.4f %8.4f %10.2f %10.6f %8.1f"
                  % (nm, N, vals.mean(), sd, sd * 100, var, cats_eff.mean()))
        print("-" * 66)

    # ---- 验证 1/sqrt(N) 律 ----
    print("\n=== sigma_pop^2 的一致性（若 1/sqrt(N) 律成立，同一 model 各行应相同）===")
    print("%-14s %6s %12s %12s" % ("model", "N", "sd_pop(pt)", "相对N=25"))
    ref = {}
    for nm in names:
        base_p = None
        for r in rows:
            if r["model"] != nm:
                continue
            sp = r["s2_pop"] ** 0.5 * 100
            if base_p is None:
                base_p = sp
                ref[nm] = r["s2_pop"]
            print("%-14s %6d %12.2f %12s" % (nm, r["N"], sp,
                                             "1.000" if base_p else "%.3f" % (sp / base_p)))
        print("-" * 48)

    # ---- 两件独立的验证 ----
    # (1) Var_sub(N) 应 = sigma_pop^2 * (1/N - 1/M)：做线性回归，看是否过原点
    print("\n=== 检验1：Var_sub(N) 是否正比于 (1/N - 1/M)（原点直线 = 1/sqrt(N) 律成立）===")
    print("%-14s %12s %12s %10s" % ("model", "slope=s2_pop", "intercept", "R^2"))
    fit_tbl = {}
    for nm in names:
        xs, ys = [], []
        for r in rows:
            if r["model"] != nm:
                continue
            xs.append(1.0 / r["N"] - 1.0 / M)
            ys.append(r["var"])
        xs = np.array(xs)
        ys = np.array(ys)
        sl, ic = np.polyfit(xs, ys, 1)
        pred = sl * xs + ic
        ss = float(((ys - pred) ** 2).sum())
        tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss / tot if tot > 0 else float("nan")
        fit_tbl[nm] = dict(slope=float(sl), intercept=float(ic), r2=float(r2))
        print("%-14s %12.6f %12.6f %10.4f" % (nm, sl, ic, r2))

    # (2) sigma_pop 的独立来源：有放回 bootstrap（200 iters）的 sd
    #     bootstrap 的样本均值方差 = sigma_pop^2 / M  => sigma_pop = sd * sqrt(M)
    npz_path = os.path.join(ROOT, "outputs", "metrics/val_ci_bootstrap.npz")
    boot_pop = {}
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        print("\n=== 检验2：与独立来源（有放回 bootstrap, N=M）交叉验证 sigma_pop ===")
        print("%-14s %12s %12s %10s" % ("model", "bootstrap", "subsample", "比值"))
        for nm in names:
            key = "boot_" + nm
            if key not in z:
                continue
            sd_b = float(z[key].std(ddof=1))
            boot_pop[nm] = sd_b * (M ** 0.5)
            sb = None
            for r in rows:
                if r["model"] == nm and (sb is None or r["N"] > sb["N"]):
                    sb = r
            sp = sb["s2_pop"] ** 0.5
            print("%-14s %12.4f %12.4f %10.3f"
                  % (nm, boot_pop[nm], sp, sp / boot_pop[nm] if boot_pop[nm] else float("nan")))

    # ---- 预测：优先用统计最稳的估计（bootstrap 200 iters > subsample） ----
    print("\n=== 评测集扩到 N' 张后的抽样不确定度  sigma(N') = sigma(200)*sqrt(200/N') ===")
    base_sd = {}
    for nm in names:
        if nm in boot_pop:
            base_sd[nm] = boot_pop[nm] / (M ** 0.5)      # = sigma(200)
        else:
            sb = None
            for r in rows:
                if r["model"] == nm and (sb is None or r["N"] > sb["N"]):
                    sb = r
            base_sd[nm] = (sb["s2_pop"] / M) ** 0.5
    predict_tbl = {}
    print("%-14s" % "model" + "".join("%9s" % ("N=%d" % N) for N in predict))
    for nm in names:
        predict_tbl[nm] = {str(N): float(base_sd[nm] * (M / N) ** 0.5) for N in predict}
        print("%-14s" % nm + "".join("%9.2f" % (predict_tbl[nm][str(N)] * 100)
                                     for N in predict))
    print("%-14s" % "(意义)" + "".join("%9s" % ("±%.1f" % (1.96 * predict_tbl[names[0]][str(N)] * 100))
                                       for N in predict))
    print("(单位 pt；各行 = 各模型的 sd(N')；末行 = 首个模型的 95%CI 半宽)")

    # ---- 要从 train 划多少过来 ----
    print("\n=== 代价：valid 全集只有 %d 张，扩集必须从 train(1800) 划 ===" % M)
    print("%9s %12s %14s %16s" % ("目标 N'", "需从train划", "train剩余", "训练数据损失"))
    for N in predict:
        if N <= M:
            continue
        take = N - M
        print("%9d %12d %14d %15.1f%%" % (N, take, 1800 - take, 100 * take / 1800.0))

    # ---- 长尾类瓶颈：N'=1800 时各类还贡献多少方差 ----
    print("\n=== 长尾类是否仍是瓶颈（选定 model 的 per-class sd）===")
    nm0 = names[0]
    for N in sizes:
        r = [x for x in rows if x["model"] == nm0 and x["N"] == N][0]
        tot_full = sum((x / K) ** 2 for x in r["pc_sd"] if x == x)
        top = sorted([(cats[k]["name"], r["pc_sd"][k], (r["pc_sd"][k] / K) ** 2)
                      for k in range(K) if r["pc_sd"][k] == r["pc_sd"][k]],
                     key=lambda x: -x[2])[:5]
        share = 100 * sum(t[2] for t in top) / tot_full if tot_full else float("nan")
        print("  N=%-4d  top5 类占 per-class 方差 %.1f%%   %s"
              % (N, share, ", ".join("%s(sd=%.3f)" % (t[0], t[1]) for t in top[:3])))

    out = dict(M=M, K=K, models=names, iters=a.iters, seed=a.seed,
               score_floor=a.score_floor, point=point, rows=rows,
               predict={k: v for k, v in predict_tbl.items()},
               predicted_sizes=predict)
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", a.out_json)


if __name__ == "__main__":
    main()
