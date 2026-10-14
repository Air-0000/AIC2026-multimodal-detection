"""Generate config for the hybrid backbone (orthogonal de-redundancy + diffuse injection)."""
import os

CFG = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2"
SRC = "deimv2_dinov3_l_aic_rdttrack_heat.yml"

text = open(os.path.join(CFG, SRC)).read()

# 1) swap the backbone name in DEIM:
text = text.replace("backbone: DINOv3STAsRDTTrack", "backbone: DINOv3STAsRDTDiffuse")

# 2) rename the backbone block and add the `inject` switch
text = text.replace("DINOv3STAsRDTTrack:\n", "DINOv3STAsRDTDiffuse:\n")
text = text.replace("  unfreeze_rgb_stages: 0\n",
                    "  unfreeze_rgb_stages: 0\n  inject: diffuse\n")

# 3) new output dir
text = text.replace("output_dir: ./outputs/deimv2_dinov3_l_aic_rdttrack_heat",
                    "output_dir: ./outputs/deimv2_dinov3_l_aic_rdtdiffuse")

out = os.path.join(CFG, "deimv2_dinov3_l_aic_rdtdiffuse.yml")
open(out, "w").write(text)

print("wrote", out)
for ln in text.splitlines():
    if ("backbone:" in ln or "output_dir:" in ln or "inject:" in ln
            or ln.startswith("DINOv3STAsRDTDiffuse")):
        print("   ", ln.strip())
