#!/usr/bin/env python3
import os, re, json, glob

OUT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"

def modality(name):
    if "rdttrack_heat" in name: return "RGB+Thermal(多模态)"
    if "nightcut_inorm" in name: return "RGB+nightcut(4ch)"
    if name == "deimv2_dinov3_l_aic" or name.startswith("deimv2_dinov3_l_aic__"): return "RGB(3ch)"
    if name.startswith("deimv2_dinov3_l_heat"): return "IR(Thermal)"
    if "rgb3ch" in name: return "RGB(3ch)"
    if "nightsim" in name: return "night-sim(4ch)"
    if "diffuse" in name: return "diffuse IR(4ch)"
    if "fused" in name: return "fused(多模态)"
    if "csd" in name: return "CSD"
    if "smoke" in name: return "smoke-test"
    if "step" in name: return "step-pipeline"
    return "other"

def classify_path(p):
    p = (p or "").lower()
    if not p or p == "aic": return "AIC"
    if "bdd100k/rgb" in p or "bdd100k" in p:
        if "night" in p: return "BDD-night"
        if "day" in p: return "BDD-day"
        return "BDD"
    if "flir" in p: return "FLIR"
    if "llvip" in p: return "LLVIP"
    if "deimv2_dataset" in p: return "AIC"
    return "?"

rows = []
for name in sorted(os.listdir(OUT)):
    d = os.path.join(OUT, name)
    if not os.path.isdir(d): continue
    log = os.path.join(d, "log.txt")
    if not os.path.exists(log): continue
    txt = open(log, "r", errors="ignore").read()
    # last test_coco_eval_bbox
    ms = list(re.finditer(r"test_coco_eval_bbox\":\s*\[([0-9.,\s]+)\]", txt))
    if not ms:
        continue
    arr = [float(x) for x in ms[-1].group(1).split(",") if x.strip()!=""]
    mAP, AP50 = arr[0], arr[1]
    # train/val dataset from config yaml
    cfg = os.path.join("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2", name + ".yml")
    train_path = val_path = ""
    if os.path.exists(cfg):
        cy = open(cfg, "r", errors="ignore").read()
        imgs = re.findall(r"img_folder:\s*(\S+)", cy)
        anns = re.findall(r"ann_file:\s*(\S+)", cy)
        train_path = imgs[0] if imgs else ""
        val_path = imgs[1] if len(imgs)>1 else train_path
        # refine with ann_file (more specific)
        if anns:
            # ann_file order: train, val
            if len(anns)>1:
                val_path = anns[1]
            else:
                val_path = anns[0]
    if not train_path:
        # name-based fallback
        if "bdd_day" in name: train_path="bdd_day"
        elif "bdd_night" in name: train_path="bdd_night"
        elif "flir" in name: train_path="flir"
        elif "llvip" in name: train_path="llvip"
        else: train_path="aic"
        val_path = train_path
    rows.append({
        "name": name, "modality": modality(name),
        "train_data": classify_path(train_path),
        "val_set": classify_path(val_path),
        "mAP": mAP, "AP50": AP50,
    })

rows.sort(key=lambda r:(r["modality"], r["train_data"], r["name"]))

md = ["# DEIMv2 各训练记录 具体 mAP（取自 log.txt 的 test_coco_eval_bbox）", ""]
md.append(f"共 {len(rows)} 条有具体 mAP 的记录（其余 ~205 个为仅评测目录，mAP 未落盘，需重跑评测才能取）。")
md.append("")
md.append("mAP = AP@IoU=0.50:0.95（val 集）；AP50 = AP@IoU=0.50。")
md.append("")
md.append("| 模态 | 训练数据 | 验证集 | 名称 | mAP | AP50 |")
md.append("|---|---|---|---|---|---|")
for r in rows:
    md.append(f"| {r['modality']} | {r['train_data']} | {r['val_set']} | `{r['name']}` | {r['mAP']:.3f} | {r['AP50']:.3f} |")

with open("/home/constant/CodeBuddy/Dev/AIC/docs/training_records_mAP.md","w") as f:
    f.write("\n".join(md))

# also print
for r in rows:
    print(f"{r['modality']:22s} {r['train_data']:9s} {r['val_set']:9s} {r['mAP']:.3f} {r['AP50']:.3f}  {r['name']}")
print("TOTAL", len(rows))
