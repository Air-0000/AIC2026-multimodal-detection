#!/usr/bin/env python3
"""生成 l4u3_na 基线 + l4u3_na_sedcn 对齐实验的 config 对 + 20 个 _na val cfg。

背景（见 docs/AIC_IR融合攻坚整合报告.html §5/§6/§11）:
  * RGBGuidedDCN（RGB 引导 DCNv1 可变形对齐, 零初始化残差）+ SE 通道门控
    已在 multimodal_backbones.py 实现（aux_dcn/aux_se 开关）, 但从未训练过。
  * 本对照在干净 _na 数据上从头训练, 单变量 = aux_se/aux_dcn 两个对齐模块。
  * 数据: heat_na / ir_na（跳过 align_ir() 假平移的干净版）。

产出:
  vendor/DEIMv2/configs/deimv2/..._midboost_l4u3_na.yml          (基线)
  vendor/DEIMv2/configs/deimv2/..._midboost_l4u3_na_sedcn.yml    (SE+DCN)
  vendor/DEIMv2/configs/deimv2/..._midboost_l4u3_na_val_{lv}.yml        x10
  vendor/DEIMv2/configs/deimv2/..._midboost_l4u3_na_sedcn_val_{lv}.yml  x10
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
CFG = os.path.join(ROOT, "vendor", "DEIMv2", "configs", "deimv2")
PREFIX = "deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x_deepir"
LV = ["day", "night50", "night25", "night12", "night05",
      "night01", "night005", "night002", "night001", "night0005"]

with open(os.path.join(CFG, PREFIX + "_midboost_l4u3_sedcn.yml")) as f:
    sedcn_src = f.read()


def swap_data(text):
    """所有数据路径换成干净 _na 版本。"""
    text = text.replace(
        "work_dir/deepir_cache/train", "work_dir/deepir_cache_na/train")
    text = text.replace(
        "work_dir/deimv2_heat/images", "work_dir/deimv2_heat_na/images")
    text = text.replace(
        "work_dir/deimv2_ir/images", "work_dir/deimv2_ir_na/images")
    text = text.replace(
        "work_dir/deimv2_heat/valid", "work_dir/deimv2_heat_na/valid")
    text = text.replace(
        "work_dir/deimv2_ir/valid", "work_dir/deimv2_ir_na/valid")
    return text


# ---- 训练 cfg: sedcn_na（保留 aux_se/aux_dcn） ------------------------------
sedcn_na = swap_data(sedcn_src)
sedcn_na = sedcn_na.replace(
    "output_dir: ./outputs/" + PREFIX + "_midboost_l4u3_sedcn",
    "output_dir: ./outputs/" + PREFIX + "_midboost_l4u3_na_sedcn")
header_sedcn = (
    "# =============================================================================\n"
    "# l4u3_na_sedcn —— IR 对齐实验（对齐臂）: l4u3 配方 + aux_se + aux_dcn, 干净 _na 数据\n"
    "#   * aux_dcn: RGBGuidedDCN —— RGB 引导 DCNv1 可变形对齐 fused aux, 零初始化残差\n"
    "#     (init=恒等)。靶点: 报告 §6.1 语义错配（逐目标 11px/p90 43px, 单一变换修不了）\n"
    "#   * aux_se : dt_ort 输出的 SE 逐通道信任门控\n"
    "#   * 单变量对照 = l4u3_na（同 config 去掉 aux_se/aux_dcn 两行, 其余逐字节相同）\n"
    "#   * 两者都 weights_path: null 纯 from-scratch, 无 -t 热启动\n"
    "# =============================================================================\n")
i = sedcn_na.index("output_dir:")
sedcn_na = sedcn_na[:i] + header_sedcn + sedcn_na[i:]
with open(os.path.join(CFG, PREFIX + "_midboost_l4u3_na_sedcn.yml"), "w") as f:
    f.write(sedcn_na)
print("wrote", PREFIX + "_midboost_l4u3_na_sedcn.yml")

# ---- 训练 cfg: l4u3_na（去掉对齐模块） --------------------------------------
base_na = swap_data(sedcn_src)
base_na = base_na.replace(
    "output_dir: ./outputs/" + PREFIX + "_midboost_l4u3_sedcn",
    "output_dir: ./outputs/" + PREFIX + "_midboost_l4u3_na")
base_na = base_na.replace(
    "  aux_se: true                # cv-improve: SE 通道门控 (dt_ort 输出)\n", "")
base_na = base_na.replace(
    "  aux_dcn: true               # cv-improve: RGB 引导 DCNv1 对齐 (零初始化残差)\n", "")
header_base = (
    "# =============================================================================\n"
    "# l4u3_na —— IR 对齐实验（基线臂）: l4u3 配方原样, 干净 _na 数据, 无对齐模块\n"
    "#   与 l4u3_na_sedcn 唯一 diff = aux_se / aux_dcn 两行。from-scratch, 无 -t。\n"
    "# =============================================================================\n")
i = base_na.index("output_dir:")
base_na = base_na[:i] + header_base + base_na[i:]
with open(os.path.join(CFG, PREFIX + "_midboost_l4u3_na.yml"), "w") as f:
    f.write(base_na)
print("wrote", PREFIX + "_midboost_l4u3_na.yml")

# ---- 20 个 val cfg ----------------------------------------------------------
for lv in LV:
    src_name = PREFIX + ("_val_orig.yml" if lv == "day" else "_val_" + lv + ".yml")
    with open(os.path.join(CFG, src_name)) as f:
        val_src = f.read()
    val_na = swap_data(val_src)
    for tag, with_aux in (("na", False), ("na_sedcn", True)):
        t = val_na.replace(
            "# =============================================================================",
            "# _na 评测 cfg（" + ("sedcn 臂: aux_se+aux_dcn=true" if with_aux else "基线臂: 无对齐模块") +
            "）—— 由 " + src_name + " 派生, 数据换 _na\n"
            "# =============================================================================",
            1)
        t = t.replace(
            "output_dir: /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/configs/deimv2/night05_out",
            "output_dir: /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2/outputs/_val_cfg_dummy_" + tag + "_" + lv)
        if with_aux:
            t = t.replace(
                "  input_norm: instance",
                "  input_norm: instance\n"
                "  aux_se: true                # 与 l4u3_na_sedcn 训练侧一致\n"
                "  aux_dcn: true               # 与 l4u3_na_sedcn 训练侧一致")
        # input_norm 后面可能带注释后缀, 兜底再试一次
        if "aux_se: true" not in t and with_aux:
            t = t.replace(
                "  unfreeze_rgb_stages:",
                "  aux_se: true\n  aux_dcn: true\n  unfreeze_rgb_stages:")
        out_name = PREFIX + "_midboost_l4u3_" + tag + "_val_" + lv + ".yml"
        with open(os.path.join(CFG, out_name), "w") as f:
            f.write(t)
        print("wrote", out_name)

print("ALL CONFIGS DONE")
