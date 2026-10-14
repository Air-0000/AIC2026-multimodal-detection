import torch

P = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/deimv2_dinov3_l_aic_fused_tardal/best_stg1.pth"
ck = torch.load(P, map_location="cpu", weights_only=False)
print("ckpt type:", type(ck))
if isinstance(ck, dict):
    print("ckpt top keys:", list(ck.keys())[:30])
    sd = ck
    for cand in ("model", "state_dict"):
        if cand in sd and isinstance(sd[cand], dict):
            sd = sd[cand]
            print("unwrapped via", cand)
            break
    if "ema" in ck and isinstance(ck["ema"], dict) and "module" in ck["ema"]:
        sd = ck["ema"]["module"]
        print("unwrapped via ema.module")
# strip DDP/module prefix
sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
print("state num tensors:", len(sd))
prefixes = {}
for k in sd.keys():
    pre = k.split(".")[0]
    prefixes[pre] = prefixes.get(pre, 0) + 1
print("top prefixes (first segment):", prefixes)
# find input conv weight (4D) to read in-channels
for k, v in sd.items():
    kl = k.lower()
    if ("stem" in kl or "patch_embed" in kl or "conv1" in kl or "in_proj" in kl) and v.dim() == 4:
        print("INPUT CONV:", k, "shape", tuple(v.shape), "-> in_ch", v.shape[1])
        break
print("---- first 15 keys ----")
for i, k in enumerate(sd.keys()):
    if i < 15:
        print("  ", k, tuple(sd[k].shape))
