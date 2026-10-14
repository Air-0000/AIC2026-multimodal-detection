import json, re, glob, os

LOG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/vitl800_scratch_stageC_fuse/log.txt"
EVAL_DIR = "/home/constant/CodeBuddy/Dev/AIC/logs"

# 1) per-epoch val mAP from training log (test_coco_eval_bbox = [mAP, AP50, AP75, ...])
curve = []
with open(LOG) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "epoch" in d and "test_coco_eval_bbox" in d:
            ev = d["test_coco_eval_bbox"]
            curve.append({
                "epoch": d["epoch"],
                "mAP": round(ev[0], 4),
                "AP50": round(ev[1], 4),
                "AP75": round(ev[2], 4),
                "lr": d.get("train_lr"),
            })
curve.sort(key=lambda x: x["epoch"])
print(f"[curve] {len(curve)} epochs, best mAP={max(c['mAP'] for c in curve):.4f} @ep{max(curve,key=lambda x:x['mAP'])['epoch']}")

# 2) night-robustness table from eval logs
levels = ["day","night50","night25","night12","night05","night01","night005","night002","night001","night0005"]
night = []
pat = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \]\s*=\s*([0-9.]+)")
for lv in levels:
    p = f"{EVAL_DIR}/stageC_fuse_eval_{lv}.log"
    if not os.path.exists(p):
        night.append({"level": lv, "mAP": None}); continue
    txt = open(p).read()
    m = pat.findall(txt)
    night.append({"level": lv, "mAP": round(float(m[-1]), 4) if m else None})
print("[night]", night)

out = {"curve": curve, "night": night,
       "swa_ckpt": "swa_stgC_ep24_39.pth"}
with open("/home/constant/CodeBuddy/Dev/AIC/outputs/metrics/stageC_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("[ok] wrote outputs/metrics/stageC_results.json")
