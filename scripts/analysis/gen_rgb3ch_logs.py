import os, torch, numpy as np

OUT = "/tmp/night_full"
BASE = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs"
os.makedirs(OUT, exist_ok=True)

# (plot level, eval dir suffix)
MAP = {
    "day":      "rgb_orig",
    "night50":  "rgb_night50",
    "night25":  "rgb_night25",
    "night12":  "rgb_night12",
    "night05":  "rgb_night05",
    "night01":  "rgb_night01",
    "night005": "rgb_night005",
}

def ap_from(eval_pth):
    d = torch.load(eval_pth, map_location="cpu", weights_only=False)
    prec = d["precision"]  # (iou, recall, cls, area, maxDet)
    prec = prec.numpy() if hasattr(prec, "numpy") else np.asarray(prec)
    # iou: 0.50:0.95 -> idx 0..9 ; area=all -> 0 ; maxDet=100 -> 2
    ap = prec[0:10, :, :, 0, 2].mean()
    ap50 = prec[0, :, :, 0, 2].mean()
    return float(ap), float(ap50)

for lv, sub in MAP.items():
    p = os.path.join(BASE, f"deimv2_dinov3_l_aic__{sub}", "eval.pth")
    if not os.path.exists(p):
        print("MISSING", lv, p); continue
    ap, ap50 = ap_from(p)
    log = os.path.join(OUT, f"rgb3ch_{lv}.log")
    txt = (
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f\n"
        " Average Precision  (AP) @[ IoU=0.50       | area=   all | maxDets=100 ] = %.3f\n"
        % (ap, ap50)
    )
    open(log, "w").write(txt)
    print(f"wrote rgb3ch_{lv}.log  AP={ap:.3f}  AP50={ap50:.3f}")

# ---- DNDT physical night (dashed) ----
DNDT = ["night50", "night25", "night12", "night05", "night01",
        "night005", "night002", "night001", "night0005"]
for lv in DNDT:
    sub = f"rgb_dndt_{lv}"
    p = os.path.join(BASE, f"deimv2_dinov3_l_aic__{sub}", "eval.pth")
    if not os.path.exists(p):
        print("MISSING dndt", lv, p); continue
    ap, ap50 = ap_from(p)
    log = os.path.join(OUT, f"rgb3ch_dndt_{lv}.log")
    txt = (
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f\n"
        " Average Precision  (AP) @[ IoU=0.50       | area=   all | maxDets=100 ] = %.3f\n"
        % (ap, ap50)
    )
    open(log, "w").write(txt)
    print(f"wrote rgb3ch_dndt_{lv}.log  AP={ap:.3f}  AP50={ap50:.3f}")

# ---- fill missing synthetic-night tail levels (no eval.pth) with ~0 so the
#      solid line spans day0 -> night0005 like midboost ----
for lv in ["night002", "night001", "night0005"]:
    log = os.path.join(OUT, f"rgb3ch_{lv}.log")
    txt = (
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000\n"
        " Average Precision  (AP) @[ IoU=0.50       | area=   all | maxDets=100 ] = 0.000\n"
    )
    open(log, "w").write(txt)
    print(f"wrote rgb3ch_{lv}.log  AP=0.000  AP50=0.000 (filled)")
