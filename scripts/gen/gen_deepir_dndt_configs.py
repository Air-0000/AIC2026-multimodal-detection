import os
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFG = os.path.join(ROOT, "vendor/DEIMv2/configs/deimv2")
LV = ["night50","night25","night12","night05","night01",
      "night005","night002","night001","night0005"]
PREF = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_"
for lv in LV:
    src = os.path.join(CFG, f"{PREF}{lv}.yml")          # 合成夜间 config（已生成）
    dst = os.path.join(CFG, f"{PREF}dndt_{lv}.yml")
    t = open(src).read()
    # val RGB: aic_valid_rgb_nightXX -> aic_dndt_nightXX/images（物理夜间只改可见光；图片在 images/ 子目录）
    t = t.replace(f"/data_aux/aic_valid_rgb_{lv}", f"/work_dir/aic_dndt_{lv}/images")
    # output_dir 去重，避免与合成夜间评测冲突
    t = t.replace(f"/val_{lv}", f"/val_dndt_{lv}")
    open(dst, "w").write(t)
    print("wrote", os.path.basename(dst),
          "rgb->", "aic_dndt_" + lv in t)
