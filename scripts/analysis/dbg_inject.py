import os, sys
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "lib"))
os.chdir(os.path.join(ROOT, "vendor", "DEIMv2"))

from engine.misc import dist_utils
from engine.solver import TASKS
from engine.core import YAMLConfig
from engine.core.workspace import GLOBAL_CONFIG

CFG = "configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir_midboost_l4u3_sedcn.yml"
cfg = YAMLConfig(CFG)
m = cfg.yaml_cfg["model"]
print("[model] value:", m if isinstance(m, str) else list(m.keys()))
gc = cfg.global_cfg
name = m if isinstance(m, str) else None
if name:
    sch = gc[name]
    print(f"[{name}] _inject:", sch.get("_inject"), "| _pymodule:", bool(sch.get("_pymodule")))
    for k in sch.get("_inject", []):
        _k = sch.get(k)
        print(f"  inject {k} -> {type(_k).__name__}:", _k if not isinstance(_k, dict) else _k)
        if isinstance(_k, str) and _k in gc:
            print(f"    global_cfg[{_k}] _name:", gc[_k].get("_name"), "| top keys:", [x for x in gc[_k]][:6])
        elif isinstance(_k, str):
            print(f"    global_cfg[{_k}] MISSING")
