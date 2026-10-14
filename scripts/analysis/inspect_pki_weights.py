import torch

ck = torch.load("/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deim_hgnetv2_x_aic_obj2coco_ft_pki/last.pth",
                map_location="cpu", weights_only=False)
sd = ck.get("model", ck)
print("last_epoch:", ck.get("epoch"), "| keys in sd:", len(sd))

fuses = {k: v for k, v in sd.items() if "pki_fuse" in k}
dws = {k: v for k, v in sd.items() if "pki_dw5" in k}
print(f"n_fuse={len(fuses)}")
import math
fnorms = [v.norm().item() for v in fuses.values()]
dnorms = [v.norm().item() for v in dws.values()]
fnorms.sort()
print("fuse |W| norm: min %.3f med %.3f max %.3f" % (fnorms[0], fnorms[len(fnorms)//2], fnorms[-1]))
dnorms.sort()
print("dw5  |W| norm: min %.3f med %.3f max %.3f" % (dnorms[0], dnorms[len(dnorms)//2], dnorms[-1]))
# 对照: 一个典型 encoder bottleneck conv 的范数量级
ref = {k: v.norm().item() for k, v in sd.items() if "fpn_blocks.0.cv2.0.bottlenecks.0.conv1.conv.weight" in k}
print("ref bottleneck conv1 |W|:", ref)
# fuse 输出量级估计: |fuse(dw(y))| ~ |fuse| * |dw(y)|, 看 fuse 相对量级
ema = ck.get("ema")
if ema is not None and hasattr(ema, "module"):
    print("has ema.module")
