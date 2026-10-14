#!/usr/bin/env python3
import zipfile
import os
import glob
from collections import Counter, defaultdict

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
    6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle",
}

zips = {
    "adaptive (51.685)": "/home/constant/CodeBuddy/Dev/AIC/work_dir/submissions/submissions_v22_cycle1/submission_v22_cycle1_adaptive.zip",
    "thr25 (52.615)": "/home/constant/CodeBuddy/Dev/AIC/work_dir/submissions/submissions_v22_cycle1/submission_v22_cycle1_thr25.zip",
    "c1_thr45 (单模黄金)": "/home/constant/CodeBuddy/Dev/AIC/work_dir/submissions/submissions_v22_cycle1/submission_v22_cycle1_thr45.zip",
    "soup_thr45 (参数汤王牌)": "/home/constant/CodeBuddy/Dev/AIC/work_dir/submissions/submissions_v22_soup/submission_v22_soup_thr45.zip",
}

stats = {}

for name, zpath in zips.items():
    c_counts = Counter()
    c_confs = defaultdict(list)
    empty_cnt = 0
    total_boxes = 0
    with zipfile.ZipFile(zpath) as zf:
        for fname in zf.namelist():
            lines = [ln.strip() for ln in zf.read(fname).decode().splitlines() if ln.strip()]
            if not lines:
                empty_cnt += 1
                continue
            total_boxes += len(lines)
            for ln in lines:
                parts = ln.split()
                cid = int(parts[0])
                conf = float(parts[5])
                c_counts[cid] += 1
                c_confs[cid].append(conf)
    stats[name] = {
        "counts": c_counts,
        "confs": {c: sum(c_confs[c])/max(len(c_confs[c]),1) for c in range(12)},
        "total": total_boxes,
        "empty": empty_cnt,
    }

header = f"{'类别名称':<16} | {'adaptive (51.685)':<18} | {'thr25 (52.615)':<16} | {'c1_thr45 (单模黄金)':<18} | {'soup_thr45 (参数汤王牌)':<20}"
print("=" * len(header))
print(header)
print("=" * len(header))

for cid in range(12):
    cname = f"[{cid:2d}] {CLASS_NAMES[cid]}"
    c_adp = stats["adaptive (51.685)"]["counts"][cid]
    c_t25 = stats["thr25 (52.615)"]["counts"][cid]
    c_c1_45 = stats["c1_thr45 (单模黄金)"]["counts"][cid]
    c_sp_45 = stats["soup_thr45 (参数汤王牌)"]["counts"][cid]
    print(f"{cname:<16} | {c_adp:<18d} | {c_t25:<16d} | {c_c1_45:<18d} | {c_sp_45:<20d}")

print("-" * len(header))
tot_adp = stats["adaptive (51.685)"]["total"]
tot_t25 = stats["thr25 (52.615)"]["total"]
tot_c1_45 = stats["c1_thr45 (单模黄金)"]["total"]
tot_sp_45 = stats["soup_thr45 (参数汤王牌)"]["total"]
print(f"{'【总预测框数】':<16} | {tot_adp:<18d} | {tot_t25:<16d} | {tot_c1_45:<18d} | {tot_sp_45:<20d}")
print(f"{'【单图均值】':<16} | {tot_adp/1000:<18.2f} | {tot_t25/1000:<16.2f} | {tot_c1_45/1000:<18.2f} | {tot_sp_45/1000:<20.2f}")
print(f"{'【空文件数】':<16} | {stats['adaptive (51.685)']['empty']:<18d} | {stats['thr25 (52.615)']['empty']:<16d} | {stats['c1_thr45 (单模黄金)']['empty']:<18d} | {stats['soup_thr45 (参数汤王牌)']['empty']:<20d}")
print("=" * len(header))
