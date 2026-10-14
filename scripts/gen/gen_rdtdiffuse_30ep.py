"""Create a 30-epoch quick-validation config for the hybrid backbone.

The 68-epoch schedule is scaled down proportionally: warmup 2000 -> 1000
(otherwise warmup alone eats ~30% of a 30-epoch run), flat 34 -> 15,
no_aug 8 -> 4, augmentation windows and the matcher switch scaled likewise.
"""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
SRC = "deimv2_dinov3_l_aic_rdtdiffuse.yml"
DST = "deimv2_dinov3_l_aic_rdtdiffuse_30ep.yml"
DST_NOIR = "deimv2_dinov3_l_aic_rdtdiffuse_30ep_val_noir.yml"

REPL = [
    ("epoches: 68", "epoches: 30"),
    ("warmup_iter: 2000", "warmup_iter: 1000"),
    ("flat_epoch: 34", "flat_epoch: 15"),
    ("no_aug_epoch: 8", "no_aug_epoch: 4"),
    ("mixup_epochs: [4, 34]", "mixup_epochs: [2, 15]"),
    ("copyblend_epochs: [4, 60]", "copyblend_epochs: [2, 25]"),
    ("stop_epoch: 60", "stop_epoch: 25"),
    ("matcher_change_epoch: 50", "matcher_change_epoch: 20"),
]

text = open(os.path.join(CFG, SRC)).read()
for old, new in REPL:
    if old not in text:
        print("!! not found:", old)
    text = text.replace(old, new)
text = text.replace("output_dir: ./outputs/deimv2_dinov3_l_aic_rdtdiffuse",
                    "output_dir: ./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep")
open(os.path.join(CFG, DST), "w").write(text)
print("wrote", DST)

# IR-ablation twin: same 30ep config, val IR replaced by all-black
IR_REAL = "/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_heat/valid"
IR_BLACK = "/home/constant/CodeBuddy/Dev/AIC/data_aux/aic_valid_ir_black"
head, val = text.split("val_dataloader:", 1)
head = head.replace("output_dir: ./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep",
                    "output_dir: ./outputs/deimv2_dinov3_l_aic_rdtdiffuse_30ep_val_noir")
val = val.replace(IR_REAL, IR_BLACK)
open(os.path.join(CFG, DST_NOIR), "w").write(head + "val_dataloader:" + val)
print("wrote", DST_NOIR)

for ln in text.splitlines():
    if ln.strip().startswith(("output_dir:", "epoches:", "warmup_iter:", "flat_epoch:",
                              "no_aug_epoch:", "inject:", "backbone:")):
        print("   ", ln.strip())
