"""统计数据集规模与类别分布，用于评估"扩评测集"能降低多少方差。"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "work_dir", "deimv2_dataset")


def stat(split, name):
    p = os.path.join(BASE, split, "annotations",
                     "instances_%s.json" % ("train" if split == "train" else "val"))
    d = json.load(open(p))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    cnt = Counter(a["category_id"] for a in d["annotations"])
    tot = sum(cnt.values())
    per = Counter(a["image_id"] for a in d["annotations"])
    imgs = {i["id"] for i in d["images"]}
    print("=" * 64)
    print("%s : images=%d  annotations=%d  有标注图=%d  无框图=%d"
          % (name, len(d["images"]), tot, len(per), len(imgs) - len(per)))
    print("%-4s %-16s %7s %7s %10s" % ("id", "name", "n_box", "pct", "出现图数"))
    for cid, n in sorted(cnt.items(), key=lambda x: -x[1]):
        nc = len({a["image_id"] for a in d["annotations"] if a["category_id"] == cid})
        print("%-4d %-16s %7d %6.1f%% %10d" % (cid, cats.get(cid, "?"), n, 100 * n / tot, nc))
    return dict(n_img=len(d["images"]), n_box=tot, per_class=cnt, cats=cats,
                per_class_img={cid: len({a["image_id"] for a in d["annotations"]
                                         if a["category_id"] == cid}) for cid in cnt})


def main():
    tr = stat("train", "TRAIN")
    va = stat("valid", "VALID")

    print("\n" + "=" * 64)
    print("若把全部 valid 用作评测集，各类 GT 框数变化：")
    print("%-16s %10s %10s %8s" % ("class", "val用200(估)", "val全量", "倍数"))
    # val200 的 per-class 20 GT 数来自 perclass_variance.json
    pv = os.path.join(ROOT, "outputs", "metrics/perclass_variance.json")
    if os.path.exists(pv):
        d = json.load(open(pv))
        pc = d.get("per_class", {})
        for k, v in sorted(pc.items(), key=lambda x: -x[1]["var_share_pct"]):
            n200 = v.get("n_gt", 0)
            print("%-16s %10d %10s %8s"
                  % (k, n200, "?", "?"))
    # 推测：valid 总量
    print("\nvalid 全集 = %d 张 / %d 框；当前只用了其中 200 张" % (va["n_img"], va["n_box"]))
    print("train 全集 = %d 张 / %d 框（训练占用，不能当 val，除非留出不训练的子集）"
          % (tr["n_img"], tr["n_box"]))


if __name__ == "__main__":
    main()
