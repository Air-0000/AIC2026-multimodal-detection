"""4ch (DiffFuse) IR ablation config on AIC val, same protocol as the 5ch one."""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
AIC = "/home/constant/CodeBuddy/Dev/AIC"
IR_REAL = os.path.join(AIC, "work_dir/deimv2_ir/valid")
IR_BLACK = os.path.join(AIC, "data_aux/aic_valid_ir_black")

src = os.path.join(CFG, "deimv2_dinov3_l_aic_diffuse_4ch.yml")
text = open(src).read()
head, val = text.split("val_dataloader:", 1)
head = head.replace(
    [ln for ln in head.splitlines() if ln.startswith("output_dir:")][0],
    "output_dir: ./outputs/deimv2_dinov3_l_aic_diffuse_4ch_val_noir",
)
val = val.replace(IR_REAL, IR_BLACK)
out = os.path.join(CFG, "deimv2_dinov3_l_aic_diffuse_4ch_val_noir.yml")
open(out, "w").write(head + "val_dataloader:" + val)
print("wrote", out)
for ln in open(out).read().splitlines():
    if "ir_folder" in ln or "output_dir" in ln:
        print("   ", ln.strip())
