#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型记录收集器 —— 单一真相源。

读取所有训练/评测 log 与配置，重建 MODEL_RECORDS.md：
  - 表A: 训练模型总览 (orig val AP/AP50/epoch/params/权重)
  - 表B: night / 暗光评测索引 (来自 logs/ 的 COCO AP 文本)

用法:
  python3 scripts/collect_model_records.py
之后新实验跑完, 重跑本脚本即可重建 (或手动在文件末尾追加)。

注意: 所有结果只进 MODEL_RECORDS.md 一个文件, 不要再散落别处。
"""
import os
import re
import json
import glob

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
OUT = os.path.join(ROOT, "vendor/DEIMv2/outputs")
LOGS = os.path.join(ROOT, "logs")
RECORDS = os.path.join(ROOT, "MODEL_RECORDS.md")

# 训练目录名 -> 系列/模态推断 (用于对比分组)
SERIES_HINTS = [
    ("x_deepir", "X+DeepIR"),
    ("deepir", "DeepIR"),
    ("midboost", "MidBoost"),
    ("ufall_x", "RDTTrack-X"),
    ("ufall", "RDTTrack-UFALL"),
    ("rdttrack", "RDTTrack"),
    ("rdtdiffuse", "RDTDiffuse"),
    ("nightsim", "NightSim"),
    ("bidir", "BiDir"),
    ("csd", "CSD"),
    ("cssa", "CSSA门控"),
    ("envgate", "Env门控"),
    ("geo", "几何(HHA)"),
    ("depthrior", "DepthPrior"),
    ("pureir", "纯IR"),
    ("diffuse", "Diffuse"),
    ("completion", "Depth补全(RGD)"),
    ("rgb3ch", "RGB(3ch)"),
    ("rgb", "RGB"),
]


def infer_series(name):
    n = name.lower()
    for key, label in SERIES_HINTS:
        if key in n:
            return label
    return "其他"


def infer_modality(name):
    """由目录名推断模态组合 (通道含义)。"""
    n = name.lower()
    if any(k in n for k in ("pureir", "pure_ir", "rawir", "raw_ir")):
        return "IR(纯红外)"
    if "completion" in n or ("depth" in n and "ir" not in n and "heat" not in n):
        return "RGB+Depth"
    if "x_deepir" in n or "ufall_x" in n:
        return "RGB+IR+Depth"
    if any(k in n for k in ("rdttrack", "fused", "tardal", "bidir", "csd")):
        return "RGB+IR"
    if "nightsim" in n or "rgb3ch" in n or "baseline" in n or n == "deimv2_dinov3_l_aic":
        return "RGB"
    if "heat" in n:
        return "RGB+IR"
    return "RGB+IR"


def infer_dataset(name):
    """由目录名推断训练数据集。"""
    n = name.lower()
    if "bdd" in n:
        return "BDD"
    if "flir" in n:
        return "FLIR"
    if "llvip" in n:
        return "LLVIP"
    return "AIC"


def infer_bdataset(bn, lv):
    """表B 专用: night*/DNTD 等暗化档位视同独立'数据集'。

    orig 评测 / 未识别档位回退到训练数据集 (AIC/BDD/FLIR/LLVIP)。
    """
    LEVEL_DS = ("day", "night50", "night25", "night12", "night05", "night01",
                "night005", "night002", "night001", "night0005",
                "dark50", "dark25", "dark12", "dark05", "dntd",
                "flir", "llvip", "bdd")
    if lv in LEVEL_DS:
        return lv
    return infer_dataset(bn)


def last_test(logp):
    """返回 (epoch, (AP, AP50), params) 取最后一条 val/test 行。"""
    epoch = None
    params = None
    last = None
    with open(logp, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if "epoch" in o:
                epoch = o["epoch"]
            if "n_parameters" in o:
                params = o["n_parameters"]
            if "test_coco_eval_bbox" in o:
                a = o["test_coco_eval_bbox"]
                last = (a[0], a[1])
    return epoch, last, params


def best_weight(d):
    for cand in ("best_stg2.pth", "best_stg1.pth", "best.pth"):
        p = os.path.join(OUT, d, cand)
        if os.path.exists(p):
            return cand
    return "—"


def parse_coco_ap(logp):
    """从 COCO 评测文本提取整体 AP / AP50。

    注意: COCO 行形如 '... maxDets=100 ] = 0.095'。必须用 ']' 锚定,
    否则会误匹配到 'maxDets=100' 里的 100 (旧版 bug, 导致全表 100.0000)。
    """
    ap = ap50 = None
    with open(logp, errors="ignore") as f:
        for line in f:
            if ap is None:
                m = re.search(
                    r"Average Precision\s+\(AP\)\s+@\[[^\]]*\]\s*=\s*([0-9.]+)", line
                )
                if m:
                    ap = float(m.group(1))
            if ap50 is None:
                m2 = re.search(
                    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|[^\]]*\]"
                    r"\s*=\s*([0-9.]+)", line
                )
                if m2:
                    ap50 = float(m2.group(1))
            if ap is not None and ap50 is not None:
                break
    return ap, ap50


# 评测 log 文件名排除 (训练/准备/管道类, 不是 night 评测)
EXCLUDE = ("_train.log", "_run.log", "_prep.log", "_gencfg.log",
           "_pipeline.log", "all_run.log", "bfusion_run.log",
           "build_bdd_nightify.log", "completion_gencfg.log")


def infer_level(name):
    pats = ["night0005", "night005", "night001", "night002", "night01",
            "night05", "night12", "night25", "night50", "day", "dark05", "dark12",
            "dark25", "dark50", "orig", "abbothblk", "abirblk", "abdepblk",
            "abrgbblk", "smoke", "featnoir", "featnoaux", "featnodepth",
            "featnoir", "llvip", "flir", "bdd", "dndt"]
    for p in pats:
        if p in name:
            return p
    return "?"


def main():
    # ---- 表A: 训练模型 ----
    rowsA = []
    for d in sorted(os.listdir(OUT)):
        if "__night" in d:
            continue
        lp = os.path.join(OUT, d, "log.txt")
        if not os.path.isfile(lp):
            continue
        ep, t, params = last_test(lp)
        if t is None:
            continue
        rowsA.append((d, ep, t[0], t[1], params, best_weight(d)))

    # ---- 表B: night 评测 ----
    rowsB = []
    for lp in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        bn = os.path.basename(lp)
        if any(bn.endswith(e) or bn == e for e in EXCLUDE):
            continue
        ap, ap50 = parse_coco_ap(lp)
        if ap is None:
            continue
        rowsB.append((bn, infer_level(bn.lower()), ap, ap50))

    # ---- 生成 markdown ----
    L = []
    L.append("# AIC 多模态检测 — 统一模型记录 (MODEL_RECORDS)")
    L.append("")
    L.append("> **单一真相源**。所有训练(orig)结果与 night/暗光评测结果集中于此, "
             "不再散落各处 log / txt / png。")
    L.append(">")
    L.append("> **维护约定**: 新实验跑完后 —")
    L.append("> 1. orig val 数字会自动进入表A (重跑 `scripts/collect_model_records.py` 重建);")
    L.append("> 2. night/暗光评测的 COCO log 放入 `logs/`, 重跑脚本即进表B;")
    L.append("> 3. 或直接在文件末尾手动追加一行, 保持表格格式。")
    L.append("> 任何结果只进本文件, 不要另建分散的记录。")
    L.append("")
    L.append(f"_生成自 {len(rowsA)} 个训练模型 + {len(rowsB)} 个 night 评测 log。_")
    L.append("")
    L.append("## 表A：训练模型总览（按 模态 / 数据集 分组，组内按 AP 降序）")
    L.append("")
    L.append("> 模态与数据集由 outputs 目录名推断；权重路径见末列。")
    L.append("> 同模态同数据集的模型才直接可比，跨组仅作参考。")
    L.append("")
    from collections import defaultdict
    groups = defaultdict(list)
    for d, ep, ap, ap50, params, w in rowsA:
        groups[(infer_modality(d), infer_dataset(d))].append(
            (d, ep, ap, ap50, params, w))
    MOD_ORDER = ["RGB", "IR(纯红外)", "RGB+IR", "RGB+Depth", "RGB+IR+Depth", "其他"]
    DS_ORDER = ["AIC", "BDD", "FLIR", "LLVIP", "orig",
                "day", "night50", "night25", "night12", "night05",
                "night01", "night005", "night002", "night001", "night0005",
                "dark50", "dark25", "dark12", "dark05", "dntd", "?"]

    def mkey(m):
        return MOD_ORDER.index(m) if m in MOD_ORDER else len(MOD_ORDER)

    def dkey(ds):
        u = ds.upper()
        idx = DS_ORDER.index(u) if u in DS_ORDER else (DS_ORDER.index(ds) if ds in DS_ORDER else len(DS_ORDER))
        return (idx, ds)

    for ds in sorted({k[1] for k in groups}, key=dkey):
        for mod in sorted({k[0] for k in groups if k[1] == ds}, key=mkey):
            items = sorted(groups[(mod, ds)], key=lambda r: -r[2])
            L.append(f"### {ds} / {mod}  （n={len(items)}）")
            L.append("")
            L.append("| # | 模型 (outputs 目录) | epoch | AP | AP50 | params | 权重 |")
            L.append("|---|---|---|---|---|---|---|")
            for i, (d, ep, ap, ap50, params, w) in enumerate(items, 1):
                L.append(f"| {i} | `{d}` | {ep} | {ap:.4f} | {ap50:.4f} | {params} | {w} |")
            L.append("")
    L.append("---")
    L.append("")
    L.append("## 表B：night / 暗光评测索引（按 模态 / 数据集 分组，组内按 AP 降序）")
    L.append("")
    L.append("> 每行 = 一个评测 log (logs/)；档位由文件名推断。")
    L.append("> 模态/数据集由日志文件名推断。")
    L.append("")
    groupsB = defaultdict(list)
    for bn, lv, ap, ap50 in rowsB:
        groupsB[(infer_modality(bn), infer_bdataset(bn, lv))].append(
            (bn, lv, ap, ap50))
    for ds in sorted({k[1] for k in groupsB}, key=dkey):
        for mod in sorted({k[0] for k in groupsB if k[1] == ds}, key=mkey):
            items = sorted(groupsB[(mod, ds)], key=lambda r: -r[2])
            L.append(f"### {ds} / {mod}  （n={len(items)}）")
            L.append("")
            L.append("| 评测 log | 档位 | AP | AP50 |")
            L.append("|---|---|---|---|")
            for bn, lv, ap, ap50 in items:
                L.append(f"| `{bn}` | {lv} | {ap:.4f} | {ap50:.4f} |")
            L.append("")
    L.append("---")
    L.append("")
    L.append("<!-- AUTO_GENERATED_TABLES_END -->")
    L.append("")

    auto_content = "\n".join(L).strip()

    # 安全保护：提取现存文件中的手写演进总账部分，严禁全量抹杀！
    manual_tail = ""
    target_path = os.path.realpath(RECORDS) if os.path.islink(RECORDS) else RECORDS
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            old_content = f.read()

        # 备份原文件
        import time
        backup_dir = os.path.join(ROOT, "archive/backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, f"MODEL_RECORDS_backup_{int(time.time())}.md")
        with open(backup_file, "w", encoding="utf-8") as bf:
            bf.write(old_content)
        print(f"安全备份已创建: {backup_file}")

        # 寻找手写演进章节切分点
        if "<!-- MANUAL_SECTIONS_START -->" in old_content:
            manual_tail = old_content.split("<!-- MANUAL_SECTIONS_START -->", 1)[1].strip()
        elif "<!-- AUTO_GENERATED_TABLES_END -->" in old_content:
            manual_tail = old_content.split("<!-- AUTO_GENERATED_TABLES_END -->", 1)[1].strip()
        elif "\n## 表C" in old_content:
            idx = old_content.find("\n## 表C")
            manual_tail = old_content[idx:].strip()
        elif "\n## 十" in old_content:
            # 找到首个演进章节（如 ## 十一 或 ## 十四）
            idx = old_content.find("\n## 十")
            manual_tail = old_content[idx:].strip()

    final_content = auto_content
    if manual_tail:
        final_content = (
            auto_content
            + "\n\n<!-- MANUAL_SECTIONS_START -->\n\n"
            + manual_tail
            + "\n"
        )

    # 关键防丢失断言验证 (Sanity check)
    if os.path.exists(target_path) and ("## 十" in old_content):
        if "## 十" not in final_content:
            raise RuntimeError(
                "CRITICAL ERROR: 手写演进总账章节在合并后丢失！写入已中止以保护真实数据。"
            )

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"written: {target_path} (via {RECORDS})")
    print(f"  表A 训练模型: {len(rowsA)}")
    print(f"  表B night评测: {len(rowsB)}")
    if manual_tail:
        print(f"  手写演进总账成功保留 (长度: {len(manual_tail)} 字符)")


if __name__ == "__main__":
    main()

