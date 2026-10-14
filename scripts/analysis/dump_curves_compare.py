import re, json, sys

EXPS = {
    "D-FINE-X 微调 (best 0.5550@47)": "deim_hgnetv2_x_aic_obj2coco_ft",
    "Stage A 纯RGB ViT-L/800 scratch": "vitl800_scratch_stageA_rgb",
    "Stage B 模态注入": "vitl800_scratch_stageB_mod",
    "Stage C 融合微调": "vitl800_scratch_stageC_fuse",
    "combo 基线 (50ep)": "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_combo",
    "l4u3 调优成品": "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3",
    "vitl16_l4like 640 scratch": "vitl16_l4like_640_scratch",
}

BASE = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/"
pat = re.compile(r'"test_coco_eval_bbox": \[([0-9.eE+, -]+)\]')

out = {}
for label, d in EXPS.items():
    vals = []
    try:
        for line in open(BASE + d + "/log.txt", errors="ignore"):
            m = pat.search(line)
            if m:
                vals.append(round(float(m.group(1).split(",")[0]), 4))
    except FileNotFoundError:
        print("MISS", d, file=sys.stderr)
    out[label] = vals
    best = max(vals) if vals else None
    print(f"{label}: n={len(vals)} last={vals[-1] if vals else '-'} best={best}@ep{vals.index(best) if best else '-'}")

json.dump(out, open("/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/model_curves_compare.json", "w"), ensure_ascii=False, indent=1)
