import torch
base = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/vitl800_scratch_stageC_fuse"
c = f"{base}/checkpoint0039.pth"
d = torch.load(c, map_location="cpu", weights_only=False)
print("TOP KEYS:", list(d.keys()))
for k in d:
    v = d[k]
    if isinstance(v, dict):
        ks = list(v.keys())
        print(f"  {k}: dict with {len(ks)} keys, sample={ks[:3]}")
    else:
        try:
            print(f"  {k}: {type(v)} shape={tuple(v.shape)}" if hasattr(v,'shape') else f"  {k}: {type(v)} = {v}")
        except Exception as e:
            print(f"  {k}: {type(v)}")
