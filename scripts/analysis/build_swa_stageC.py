import torch, sys, time

BASE = "/home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/vitl800_scratch_stageC_fuse"
# Converged-orbit window: last available snapshots covering ep24..ep39
# (flatcosine cosine-decay tail is ep20..ep37, no-aug ep37..ep39 -> all in converged region)
SRC = ["checkpoint0024.pth", "checkpoint0029.pth", "checkpoint0034.pth", "checkpoint0039.pth"]
paths = [f"{BASE}/{s}" for s in SRC]

t0 = time.time()
states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
print(f"[swa] loaded {len(states)} ckpts in {time.time()-t0:.1f}s")
keys = list(states[0]["model"].keys())
assert all(list(s["model"].keys()) == keys for s in states), "key mismatch"

avg = {}
for k in keys:
    stacked = sum(s["model"][k].float() for s in states) / len(states)
    avg[k] = stacked

out = {
    "model": avg,
    "last_epoch": 39,
    "swa": True,
    "swa_sources": SRC,
    "note": "SWA over converged-orbit checkpoints ep24/29/34/39 (see MEMORY §5.6)",
}
out_path = f"{BASE}/swa_stgC_ep24_39.pth"
torch.save(out, out_path)
print(f"[swa] saved {out_path}  ({sum(s.numel() for s in avg.values())/1e6:.1f}M params avg)")
