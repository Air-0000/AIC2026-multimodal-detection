import os, re

# Generate BiDirPaper 4ch (RGB3 + IR-heat1) night eval configs (9 darkness levels).
# Source: BiDirPaper TRAIN config, but swap val RGB folder to the night-augmented
# RGB (data_aux/aic_valid_rgb_{lv}); IR heat stays the NORMAL deimv2_heat/valid
# (IR is illumination-invariant, not darkened); depth dropped (4ch); val batch=1.
CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/deimv2_dinov3_l_aic_bidir_paper_4ch.yml"
OUTDIR = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
LEVELS = ["night50", "night25", "night12", "night05", "night01",
          "night005", "night002", "night001", "night0005"]
BASE_IMG = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset/valid/images"

src = open(CFG).read()
for lv in LEVELS:
    t = src
    t = t.replace(BASE_IMG, f"/home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_rgb_{lv}")
    # val batch 4 -> 1 (smoke protocol + memory), only inside val_dataloader block
    t = re.sub(r"(val_dataloader:.*?total_batch_size: )4", r"\g<1>1", t, flags=re.S)
    t = t.replace("output_dir: ./outputs/deimv2_dinov3_l_aic_bidir_paper_4ch",
                  f"output_dir: ./outputs/deimv2_dinov3_l_aic_bidir_paper_4ch/val_{lv}")
    out = os.path.join(OUTDIR, f"deimv2_dinov3_l_aic_bidir_paper_4ch_val_{lv}.yml")
    open(out, "w").write(t)
    print("wrote", out)
