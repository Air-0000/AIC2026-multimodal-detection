import os

SRC = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir.yml"
OUTDIR = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
W = "/home/constant/CodeBuddy/Dev/AIC/work_dir"

# level -> RGB img_folder (IR/rawIR/depth/ann are shared from deepir config)
LEVELS = {
    "orig":     f"{W}/deimv2_dataset/valid/images",
    "night0005": f"{W}/aic_valid_rgb_night0005",
    "night001":  f"{W}/aic_valid_rgb_night001",
    "night002":  f"{W}/aic_valid_rgb_night002",
    "night005":  f"{W}/aic_valid_rgb_night005",
    "night01":   f"{W}/aic_valid_rgb_night01",
    "night05":   f"{W}/aic_valid_rgb_night05",
    "night12":   f"{W}/aic_valid_rgb_night12",
    "night25":   f"{W}/aic_valid_rgb_night25",
    "night50":   f"{W}/aic_valid_rgb_night50",
    "nightify":  f"{W}/aic_nightify_valid",
}

os.makedirs(OUTDIR, exist_ok=True)
txt = open(SRC).read()
assert "deimv2_dataset/valid/images" in txt, "val img_folder anchor not found"

for lv, img in LEVELS.items():
    new = txt.replace(
        "img_folder: /home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/valid/images",
        f"img_folder: {img}",
        1,
    )
    # unique output_dir so eval results/logs don't collide
    new = new.replace(
        "output_dir: ./outputs/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir",
        f"output_dir: {OUTDIR}/{lv}_out",
        1,
    )
    dst = os.path.join(OUTDIR, f"deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_val_{lv}.yml")
    open(dst, "w").write(new)
    print("wrote", dst, "img_folder=", img)
print("TOTAL", len(LEVELS))
