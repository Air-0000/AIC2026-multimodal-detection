import os, sys
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
os.chdir(os.path.join(ROOT, "vendor", "DEIMv2"))

import engine.backbone  # trigger registration
from engine.core.workspace import GLOBAL_CONFIG

key = "DINOv3STAsRDTTrack"
print("in GLOBAL_CONFIG:", key in GLOBAL_CONFIG)
if key in GLOBAL_CONFIG:
    e = GLOBAL_CONFIG[key]
    print("top keys:", [k for k in e][:10])
    print("_name:", e.get("_name"), "| _pymodule:", bool(e.get("_pymodule")))
else:
    print("registered backbone keys sample:", [k for k in GLOBAL_CONFIG if "DINOv3" in k or "HGNet" in k][:8])
