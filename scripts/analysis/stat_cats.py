import json
import collections
import numpy as np

ll = json.load(open("/home/constant/CodeBuddy/Dev/AIC/work_dir/LLVIP/instances_llvip.json"))
ai = json.load(open("/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/train/annotations/instances_train.json"))

lc = {c["id"]: c["name"] for c in ll["categories"]}
ac = {c["id"]: c["name"] for c in ai["categories"]}
print("LLVIP cats:", lc)
print("AIC cats  :", ac)
print("交集:", sorted(set(lc.values()) & set(ac.values())))

per_img = collections.Counter(a["image_id"] for a in ll["annotations"])
n_img = len(ll["images"])
print(f"LLVIP: {n_img} imgs, {len(ll['annotations'])} boxes, "
      f"avg {len(ll['annotations'])/n_img:.2f} person/img, 无标注图片 {n_img - len(per_img)}")

ar = np.array([a["bbox"][2] * a["bbox"][3] for a in ll["annotations"]])
print(f"box 面积: 小(<32^2)={int((ar < 1024).sum())}  "
      f"中(<96^2)={int(((ar >= 1024) & (ar < 9216)).sum())}  大={int((ar >= 9216).sum())}")
print("面积中位数:", int(np.median(ar)),
      " 原图尺寸:", (ll["images"][0]["width"], ll["images"][0]["height"]))
# after resize to 512 (scale factor on the long side)
w0, h0 = ll["images"][0]["width"], ll["images"][0]["height"]
s = 512 / max(w0, h0)
print(f"resize到512后: 小目标数={int((ar * s * s < 1024).sum())}, 中位面积={int(np.median(ar) * s * s)}")

pc = collections.Counter(a["category_id"] for a in ai["annotations"])
print("AIC 各类标注数:", {ac[k]: v for k, v in sorted(pc.items())})
