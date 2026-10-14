"""Generate 5ch modality-ablation eval configs by swapping val aux folders."""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
AIC = "/home/constant/CodeBuddy/Dev/AIC"

IR_REAL = os.path.join(AIC, "work_dir/deimv2_heat/valid")
DEPTH_REAL = os.path.join(AIC, "dataset/train/depth")
IR_BLACK = os.path.join(AIC, "data_aux/aic_valid_ir_black")
DEPTH_BLACK = os.path.join(AIC, "data_aux/aic_valid_depth_black")


def swap(src, dst_name, out_dir, ir=None, depth=None, base_split="val_dataloader:"):
    text = open(os.path.join(CFG, src)).read()
    head, val = text.split(base_split, 1)
    head = head.replace(
        [ln for ln in head.splitlines() if ln.startswith("output_dir:")][0],
        f"output_dir: ./outputs/{out_dir}",
    )
    if ir:
        val = val.replace(IR_REAL, ir)
    if depth:
        val = val.replace(DEPTH_REAL, depth)
    open(os.path.join(CFG, dst_name), "w").write(head + base_split + val)
    print("wrote", dst_name, "->", out_dir)


# --- AIC val (real depth available) ---
swap("deimv2_dinov3_l_aic_rdttrack_heat.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_val_noir.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_val_noir", ir=IR_BLACK)

swap("deimv2_dinov3_l_aic_rdttrack_heat.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_val_nodepth.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_val_nodepth", depth=DEPTH_BLACK)

swap("deimv2_dinov3_l_aic_rdttrack_heat.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_val_noaux.yml",
     "deimv2_dinov3_l_aic_rdttrack_heat_val_noaux", ir=IR_BLACK, depth=DEPTH_BLACK)

# --- LLVIP: IR blacked out (depth is already null there) ---
llvip_ir = os.path.join(AIC, "work_dir/LLVIP_heat")
llvip_black = os.path.join(AIC, "work_dir/LLVIP_ir_black")
text = open(os.path.join(CFG, "deimv2_dinov3_l_aic_rdttrack_heat_llvip.yml")).read()
head, val = text.split("val_dataloader:", 1)
head = head.replace(
    [ln for ln in head.splitlines() if ln.startswith("output_dir:")][0],
    "output_dir: ./outputs/deimv2_dinov3_l_aic_rdttrack_heat_llvip_noir",
)
val = val.replace(llvip_ir, llvip_black)
open(os.path.join(CFG, "deimv2_dinov3_l_aic_rdttrack_heat_llvip_noir.yml"), "w").write(
    head + "val_dataloader:" + val)
print("wrote deimv2_dinov3_l_aic_rdttrack_heat_llvip_noir.yml")
