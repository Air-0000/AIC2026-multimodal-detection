#!/usr/bin/env python3
import os, re, subprocess, json

OUT = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
rows = []

def grep1(path, pat):
    try:
        r = subprocess.run(["grep","-m1","-E", pat, path], capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return ""

def modality(name):
    if "rdttrack_heat" in name: return "RGB+Thermal(多模态)"
    if "nightcut_inorm" in name: return "RGB+nightcut(4ch)"
    if name.startswith("deimv2_dinov3_l_heat"): return "IR(Thermal)"
    if name == "deimv2_dinov3_l_aic" or name.startswith("deimv2_dinov3_l_aic__"): return "RGB(3ch)"
    if "rgb3ch" in name: return "RGB(3ch)"
    if "nightsim" in name: return "night-sim(4ch)"
    if "diffuse" in name: return "diffuse IR(4ch)"
    if "fused" in name: return "fused(多模态)"
    if "csd" in name: return "CSD"
    if "smoke" in name: return "smoke-test"
    if "step" in name: return "step-pipeline"
    return "other"

def train_data(name, train_path):
    p = (train_path or "").lower()
    if "bdd100k" in p or "bdd_day" in name or "bdd_night" in name:
        if "day" in name: return "BDD-day"
        if "night" in name: return "BDD-night"
        return "BDD"
    if "flir" in p or "flir" in name: return "FLIR"
    if "llvip" in p or "llvip" in name: return "LLVIP"
    if "deimv2_dataset" in p or "aic" in name: return "AIC(day)"
    return "?"

for name in sorted(os.listdir(OUT)):
    d = os.path.join(OUT, name)
    if not os.path.isdir(d): continue
    files = os.listdir(d)
    has_ckpt = any(f.startswith(("best_stg","last","checkpoint")) and f.endswith(".pth") for f in files)
    has_eval = any(f in ("eval.pth",) or f.startswith("eval") for f in files)
    kind = "train" if has_ckpt else ("eval-only" if has_eval else "empty")
    log = os.path.join(d, "log.txt")
    mAP = train_path = None
    if os.path.exists(log):
        line = grep1(log, r"img_folder")
        # first img_folder is train dataloader
        train_path = line
        ap = grep1(log, r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all")
        if ap:
            m = re.search(r"= ([0-9.]+)", ap)
            if m: mAP = float(m.group(1))
    if mAP is None:
        # eval-only: try summary
        pass
    rows.append({
        "name": name, "modality": modality(name),
        "train_data": train_data(name, train_path),
        "kind": kind, "mAP": mAP,
        "train_path": train_path or "",
    })

# group
groups = {}
for r in rows:
    k = (r["modality"], r["train_data"])
    groups.setdefault(k, []).append(r)

# write markdown
md = ["# DEIMv2 训练记录总表(按 模态 × 训练数据 分类)", ""]
md.append(f"共 {len(rows)} 个输出目录。kind: train=有训练ckpt; eval-only=仅评测产物; empty=空目录。")
md.append("")
md.append("## 分类汇总(计数)")
md.append("")
md.append("| 模态 | 训练数据 | 数量 | 训练 / 仅评测 |")
md.append("|---|---|---|---|")
for k in sorted(groups):
    g = groups[k]
    nt = sum(1 for r in g if r["kind"]=="train")
    ne = sum(1 for r in g if r["kind"]=="eval-only")
    md.append(f"| {k[0]} | {k[1]} | {len(g)} | {nt} / {ne} |")

md.append("")
md.append("## 明细(按 模态 > 训练数据 > 名称)")
md.append("")
md.append("| 模态 | 训练数据 | 名称 | 类型 | val mAP |")
md.append("|---|---|---|---|---|")
for k in sorted(groups):
    for r in sorted(groups[k], key=lambda x:x["name"]):
        mp = f"{r['mAP']:.3f}" if r["mAP"] is not None else "-"
        md.append(f"| {r['modality']} | {r['train_data']} | `{r['name']}` | {r['kind']} | {mp} |")

with open("/home/constant/CodeBuddy/Dev/AIC/docs/training_records.md","w") as f:
    f.write("\n".join(md))

# also print category counts
print("TOTAL", len(rows))
for k in sorted(groups):
    g = groups[k]
    nt = sum(1 for r in g if r["kind"]=="train")
    ne = sum(1 for r in g if r["kind"]=="eval-only")
    print(f"{k[0]:24s} | {k[1]:10s} | n={len(g):3d} (train={nt}, eval={ne})")
