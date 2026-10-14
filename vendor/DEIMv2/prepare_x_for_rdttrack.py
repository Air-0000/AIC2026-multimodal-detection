#!/usr/bin/env python
"""把 HuggingFace 完整 DEIMv2-X 检测权重转成 RDTTrack 训练可用的 resume 权重。

DEIMv2-X 完整权重 (Intellindust/DEIMv2_DINOv3_X_COCO) 是单 RGB 主干检测模型,
权重为 model.safetensors (51.3M 参数, 约 205MB)。其 key 形如 (可能嵌套在
'model' / 'state_dict' / 'ema' 下):
    backbone.dinov3.*   (RGB DINOv3 主干, 3ch dinov3_vits16plus)
    encoder.*           (HybridEncoder)
    decoder.*           (DEIMTransformer)

RDTTrack 三流模型里 RGB 主干叫 backbone.rgb_dinov3.*; IR/Depth stem / prompt / sta
等都是 RDTTrack 独有 (X 权重里没有) -> resume 时 strict=False 随机初始化。

支持 .safetensors 与 .pth/.bin 两种输入。

用法:
    python prepare_x_for_rdttrack.py <deimv2_x_weight> <out.pth>
然后:
    train.py -c configs/deimv2/deimv2_dinov3_l_aic_rdttrack_heat_night_log_rnight_ufall_x.yml \
             -r <out.pth>

脚本会打印 RGB 主干 embed_dim (从 qkv.weight 倒数第二维读), 用于核对本代码
vision_transformer.py 里 dinov3_vits16plus 的 embed_dim 是否一致 (384 vs 256)。
"""
import sys
import torch

try:
    from safetensors.torch import load_file as _safe_load
    HAS_SAFE = True
except ImportError:
    HAS_SAFE = False


def load_ckpt(path):
    if path.endswith(".safetensors") and HAS_SAFE:
        return _safe_load(path, device="cpu")
    return torch.load(path, map_location="cpu")


def _strip(ck):
    """递归剥掉 'model' / 'state_dict' 嵌套; 优先用非 ema 权重。"""
    if isinstance(ck, dict):
        for key in ("model", "state_dict"):
            if key in ck and isinstance(ck[key], dict) and len(ck[key]):
                return _strip(ck[key])
        if "ema" in ck and isinstance(ck["ema"], dict):
            ema = ck["ema"]
            ema_sd = ema.get("module", ema) if isinstance(ema, dict) else ema
            others = [k for k in ck if k != "ema"]
            if isinstance(ema_sd, dict) and len(ema_sd) >= len(others):
                return _strip(ema_sd)
    return ck


def _channel_repeat_dinov3(v, in_chans):
    """Channel-repeat a DINOv3 patch_embed weight 3 -> in_chans (mean over RGB)."""
    if in_chans == 3:
        return v.clone()
    m = v.mean(dim=1, keepdim=True)            # [D, 1, K, K]
    return m.repeat(1, in_chans, 1, 1)


def main():
    if len(sys.argv) < 3:
        print("usage: python prepare_x_for_rdttrack.py <deimv2_x_weight> <out.pth> "
              "[--ir-in-chans 2] [--depth-in-chans 1]")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    ir_in_chans, depth_in_chans = 2, 1
    rest_args = sys.argv[3:]
    for i, a in enumerate(rest_args):
        if a == "--ir-in-chans" and i + 1 < len(rest_args):
            ir_in_chans = int(rest_args[i + 1])
        elif a == "--depth-in-chans" and i + 1 < len(rest_args):
            depth_in_chans = int(rest_args[i + 1])

    ck = _strip(load_ckpt(src))

    out = {}
    n_rgb = n_enc = n_dec = n_ir = n_dep = 0
    for k, v in ck.items():
        if k.startswith("backbone.dinov3."):
            rest = k[len("backbone.dinov3."):]
            out["backbone.rgb_dinov3." + rest] = v
            n_rgb += 1
            # IR DINOv3 (PureIR migration): full DINOv3, patch_embed 3 -> ir_in_chans
            out["backbone.ir_dinov3." + rest] = (
                _channel_repeat_dinov3(v, ir_in_chans)
                if k.endswith("patch_embed.proj.weight") else v.clone())
            n_ir += 1
            # Depth DINOv3 (depth optimisation): full DINOv3, patch_embed 3 -> depth_in_chans
            out["backbone.depth_dinov3." + rest] = (
                _channel_repeat_dinov3(v, depth_in_chans)
                if k.endswith("patch_embed.proj.weight") else v.clone())
            n_dep += 1
        elif k.startswith("encoder."):
            out[k] = v
            n_enc += 1
        elif k.startswith("decoder."):
            out[k] = v
            n_dec += 1

    # 打印 RGB 主干 embed_dim 供核对 (qkv.weight shape: [3*embed, embed])
    for k, v in out.items():
        if k.endswith("backbone.rgb_dinov3.blocks.0.attn.qkv.weight"):
            print(f"[核对] RGB 主干 embed_dim = {v.shape[-1]}  (qkv.weight shape={tuple(v.shape)})")

    torch.save(out, dst)
    print(f"saved {dst}: rgb_dinov3={n_rgb}, ir_dinov3={n_ir}, depth_dinov3={n_dep}, encoder={n_enc}, decoder={n_dec}")
    if n_rgb == 0:
        print("WARNING: 未找到 backbone.dinov3.* —— 权重 key 与预期不符, "
              "请检查源权重 (可能需先 strip 其他前缀)。")


if __name__ == "__main__":
    main()
