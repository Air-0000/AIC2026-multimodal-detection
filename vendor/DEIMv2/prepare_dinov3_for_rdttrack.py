#!/usr/bin/env python
"""把官方 DINOv3 (ViT-B/16 或 ViT-L/16) 预训练权重转成 RDTTrack 三流热启动权重。

背景
----
本仓库 (vendor/DEIMv2) 只内置 DINOv3 的 S 系列配置 (vits16 / vits16plus /
vits16_deep, embed_dim=384)。仓库自带的 `deimv2_dinov3_l_coco.pth` 名字带 "l",
但实测 patch_embed=(384,3,16,16) + 12 blocks, 其实也是 ViT-S —— DEIMv2 的 "L/X"
指的是 encoder/decoder 规模, 不是 ViT 规模。

想要真正更大的 backbone, 必须有官方 ViT-B/16 (768) 或 ViT-L/16 (1024) 权重,
并在 engine/backbone/dinov3/vision_transformer.py 的 configs 里注册对应条目
(已加: 'dinov3_vitb16' / 'dinov3_vitl16')。

官方 facebook/dinov3-* 是 gated (401), dl.fbaipublicfiles.com 直链 403。
改用未受限镜像 Fanqi-Lin-IR/dinov3_vitb16 | dinov3_vitl16 —— 已 sha256 校验
与官方原文件逐字节一致 (DINOv3 文件名后缀即内容 sha256 前 8 位)。

本脚本解决三流的通道适配:
  * RGB  主干: in_chans=3 -> 直接吃官方权重
  * IR   主干: in_chans=2 (heat + 配准原始 IR) -> patch_embed 3->2 通道均值复制
  * Depth主干: in_chans=1 -> patch_embed 3->1 通道均值复制
(与 prepare_x_for_rdttrack.py 的 _channel_repeat_dinov3 同一套做法)

可选: 把旧 ViT-S 模型的 encoder.* / decoder.* 一起并进来。RDTTrack 的 convs 把
三条流的特征统一投影到 hidden_dim=256 再喂 HybridEncoder, 所以换大 backbone
后 encoder/decoder 的输入输出维度不变, 旧权重可以继承 (shape 不匹配的键会被
train.py 的 _matched_state 自动跳过, 不会报错)。

用法
----
    cd vendor/DEIMv2
    python prepare_dinov3_for_rdttrack.py \
        --dinov3 ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --out    ckpts/dinov3/vitl16_rdttrack_init.pth \
        --inherit-aux     outputs/<old_s_model>/best_stg1.pth \
        --inherit-encoder outputs/<old_s_model>/best_stg1.pth

然后训练时用 -t 热启动 (不要用 -r):
    python train.py -c configs/deimv2/<new>.yml -t ckpts/dinov3/vitl16_rdttrack_init.pth --use-amp

-t 走 engine/solver/_solver.py::_matched_state, 按 (key 存在 AND shape 相同) 过滤,
不匹配的一律跳过并保持随机初始化 —— 所以缺 encoder/decoder 也不会报错。
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _channel_repeat_dinov3(v, in_chans):
    """Channel-repeat a DINOv3 patch_embed weight 3 -> in_chans (mean over RGB)."""
    if v.dim() != 4:
        return v.clone()
    if v.shape[1] == in_chans:
        return v.clone()
    m = v.mean(dim=1, keepdim=True)          # [D, 1, K, K]
    return m.repeat(1, in_chans, 1, 1)


def _strip(ck):
    """剥掉 'model' / 'state_dict' / 'ema' 嵌套。"""
    if isinstance(ck, dict):
        for key in ("model", "state_dict"):
            if key in ck and isinstance(ck[key], dict) and len(ck[key]):
                return _strip(ck[key])
        if "ema" in ck and isinstance(ck["ema"], dict):
            ema = ck["ema"]
            return _strip(ema.get("module", ema))
    return ck


def load_ckpt(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return _strip(load_file(path, device="cpu"))
    return _strip(torch.load(path, map_location="cpu"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3", required=True, help="主 RGB 流权重 (如官方 ViT-B/16 .pth)")
    ap.add_argument("--aux-dinov3", default=None,
                    help="IR/Depth 辅助流权重。给一个更小的 DINOv3 (如 ViT-S) 以省显存; "
                         "省略则辅助流与主 RGB 流同尺寸。")
    ap.add_argument("--out", required=True, help="输出热启动权重路径")
    ap.add_argument("--inherit-encoder", default=None,
                    help="可选的旧模型 ckpt, 继承其 encoder.* / decoder.*")
    ap.add_argument("--inherit-aux", default=None,
                    help="旧 (ViT-S) 模型 ckpt, 直接继承其已训练的 backbone.ir_dinov3.* / "
                         "backbone.depth_dinov3.*。这比用 --aux-dinov3 从头通道复制更好: "
                         "旧权重已经带着训练过的 2ch/1ch patch_embed。")
    ap.add_argument("--ir-in-chans", type=int, default=2)
    ap.add_argument("--depth-in-chans", type=int, default=1)
    args = ap.parse_args()

    ck = load_ckpt(args.dinov3)

    # 打印从权重反推的超参, 与 vision_transformer.py 的注册条目核对
    pe = ck.get("patch_embed.proj.weight")
    qkv = ck.get("blocks.0.attn.qkv.weight")
    fc1 = ck.get("blocks.0.mlp.fc1.weight")
    if fc1 is None:
        fc1 = ck.get("blocks.0.ffn.fc1.weight")
    if pe is not None:
        print(f"[核对] patch_embed {tuple(pe.shape)} -> embed_dim={pe.shape[0]}, "
              f"in_chans={pe.shape[1]}, patch={pe.shape[-1]}")
    if qkv is not None:
        print(f"[核对] blocks.0.attn.qkv {tuple(qkv.shape)} -> qkv=3*embed_dim ok={qkv.shape[0] == 3 * qkv.shape[1]}")
    if fc1 is not None:
        print(f"[核对] blocks.0 mlp.fc1 {tuple(fc1.shape)} -> ffn_ratio={fc1.shape[0] / fc1.shape[1]:.1f}")
    nblk = len({int(k.split(".")[1]) for k in ck if k.startswith("blocks.") and k.count(".") >= 2})
    print(f"[核对] depth = {nblk}")

    # ---- resolve the aux-stream (IR / Depth) weight source -------------------
    # Priority: --inherit-aux  >  --aux-dinov3  >  same weights as the RGB stream.
    # --inherit-aux is the best option: the old ViT-S model's aux stems are already
    # trained AND already carry the real 2ch/1ch patch_embed, so nothing is guessed.
    ir_src, dep_src, aux_origin = ck, ck, "same as RGB"
    if args.inherit_aux:
        if not os.path.exists(args.inherit_aux):
            print(f"[warn] --inherit-aux 路径不存在: {args.inherit_aux}")
        else:
            _old = load_ckpt(args.inherit_aux)
            _ir = {k[len("backbone.ir_dinov3."):]: v for k, v in _old.items()
                   if k.startswith("backbone.ir_dinov3.")}
            _dep = {k[len("backbone.depth_dinov3."):]: v for k, v in _old.items()
                    if k.startswith("backbone.depth_dinov3.")}
            if _ir and _dep:
                ir_src, dep_src = _ir, _dep
                aux_origin = f"inherited from {args.inherit_aux}"
            else:
                print(f"[warn] --inherit-aux 未找到 ir/depth dinov3 键 "
                      f"(ir={len(_ir)}, depth={len(_dep)}), 回退")
    if aux_origin == "same as RGB" and args.aux_dinov3:
        ir_src = dep_src = load_ckpt(args.aux_dinov3)
        aux_origin = args.aux_dinov3
    _ape = ir_src.get("patch_embed.proj.weight")
    print(f"[辅助流] 来源: {aux_origin} | embed_dim="
          f"{_ape.shape[0] if _ape is not None else '?'}")
    # Inherited weights already have the right patch_embed width; only repeat when
    # we are deriving the aux stems from a 3-channel pretrained checkpoint.
    _inherited = aux_origin.startswith("inherited")

    out = {}
    n_rgb = n_ir = n_dep = 0
    for k, v in ck.items():
        out["backbone.rgb_dinov3." + k] = v
        n_rgb += 1
    for k, v in ir_src.items():
        is_pe = k.endswith("patch_embed.proj.weight")
        out["backbone.ir_dinov3." + k] = (
            v.clone() if _inherited else
            (_channel_repeat_dinov3(v, args.ir_in_chans) if is_pe else v.clone()))
        n_ir += 1
    for k, v in dep_src.items():
        is_pe = k.endswith("patch_embed.proj.weight")
        out["backbone.depth_dinov3." + k] = (
            v.clone() if _inherited else
            (_channel_repeat_dinov3(v, args.depth_in_chans) if is_pe else v.clone()))
        n_dep += 1

    n_enc = n_dec = 0
    if args.inherit_encoder and os.path.exists(args.inherit_encoder):
        old = load_ckpt(args.inherit_encoder)
        for k, v in old.items():
            if k.startswith("encoder."):
                out[k] = v
                n_enc += 1
            elif k.startswith("decoder."):
                out[k] = v
                n_dec += 1
    elif args.inherit_encoder:
        print(f"[warn] --inherit-encoder 路径不存在: {args.inherit_encoder}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": out}, args.out)
    print(f"saved {args.out}: rgb_dinov3={n_rgb}, ir_dinov3={n_ir}, depth_dinov3={n_dep}, "
          f"encoder={n_enc}, decoder={n_dec}")
    if n_rgb == 0:
        print("[FATAL] 未找到任何权重键, 请检查 --dinov3 文件格式")
        sys.exit(1)


if __name__ == "__main__":
    main()
