"""CPU contract tests of the Co-DETR ViT-L port (scripts/train/codino/aic_codino_vit.py) against the original code.

The references below are transcribed from Sense-X/Co-DETR mmdet/models/backbones/vit.py (einops-based RoPE,
explicit-softmax attention) and mmdet/core/optimizers/layer_decay_optimizer_constructor.py. Needs the codino env
(torch, mmcv 2.1, mmengine, mmdet 3.3) plus einops; run from the repo root with codino_env.sh sourced:
    CUDA_VISIBLE_DEVICES='' python scripts/diagnostics/test_codino_vit_port.py
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'codino'))
import aic_codino_vit as port  # noqa: E402

try:
    from einops import rearrange, repeat
except ImportError:  # the references need einops; the port does not
    rearrange = repeat = None


def ref_broadcat(tensors, dim=-1):
    shape_len = len(tensors[0].shape)
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1], ) * len(tensors)), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def ref_rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def ref_rope_win(dim, pt_seq_len, ft_seq_len):
    freqs = 1. / (10000**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
    freqs = torch.einsum('..., f -> ... f', t, freqs)
    freqs = repeat(freqs, '... n -> ... (n r)', r=2)
    freqs = ref_broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
    return freqs.cos().view(-1, freqs.shape[-1]), freqs.sin().view(-1, freqs.shape[-1])


def ref_get_rope(t, H, W):
    dim, pt_seq_len = 32, 16
    freqs = 1. / (10000**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    tH = torch.arange(H) / H * pt_seq_len
    tW = torch.arange(W) / W * pt_seq_len
    freqsH = repeat(torch.einsum('..., f -> ... f', tH, freqs), '... n -> ... (n r)', r=2)
    freqsW = repeat(torch.einsum('..., f -> ... f', tW, freqs), '... n -> ... (n r)', r=2)
    freqs = ref_broadcat((freqsH[:, None, :], freqsW[None, :, :]), dim=-1)
    return t * freqs.cos().view(-1, freqs.shape[-1]) + ref_rotate_half(t) * freqs.sin().view(-1, freqs.shape[-1])


def ref_attention(attn, x):
    """Upstream Attention.forward with xattn=False, using the port module's weights."""
    B, H, W, C = x.shape
    x = x.view(B, -1, C)
    N = H * W
    q = torch.nn.functional.linear(x, attn.q_proj.weight, attn.q_bias)
    k = torch.nn.functional.linear(x, attn.k_proj.weight, None)
    v = torch.nn.functional.linear(x, attn.v_proj.weight, attn.v_bias)
    q = q.reshape(B, N, attn.num_heads, -1).permute(0, 2, 1, 3)
    k = k.reshape(B, N, attn.num_heads, -1).permute(0, 2, 1, 3)
    v = v.reshape(B, N, attn.num_heads, -1).permute(0, 2, 1, 3)
    if attn.rope is not None:
        q, k = attn.rope(q).type_as(v), attn.rope(k).type_as(v)
    else:
        q, k = ref_get_rope(q, H, W).type_as(v), ref_get_rope(k, H, W).type_as(v)
    q = q * (q.shape[-1]**-0.5)
    a = (q @ k.transpose(-2, -1)).softmax(dim=-1).type_as(x)
    x = (a @ v).transpose(1, 2).reshape(B, N, -1)
    return attn.proj(x).view(B, H, W, C)


@unittest.skipIf(rearrange is None, 'einops not installed (needed by the upstream reference code)')
class RopeAndAttention(unittest.TestCase):

    def test_rotate_half(self):
        x = torch.randn(2, 4, 10, 64)
        self.assertTrue(torch.equal(port.rotate_half(x), ref_rotate_half(x)))

    def test_window_rope_tables(self):
        mod = port.VisionRotaryEmbeddingFast(32, 16, 24)
        cos, sin = ref_rope_win(32, 16, 24)
        torch.testing.assert_close(mod.freqs_cos, cos, rtol=0, atol=1e-6)
        torch.testing.assert_close(mod.freqs_sin, sin, rtol=0, atol=1e-6)

    def test_global_rope_non_square(self):
        attn = port.Attention(1024, 16, rope=None)
        t = torch.randn(1, 16, 7 * 13, 64)
        torch.testing.assert_close(attn._global_rope(t, 7, 13), ref_get_rope(t, 7, 13), rtol=0, atol=1e-5)

    def test_attention_window_and_global(self):
        torch.manual_seed(0)
        rope = port.VisionRotaryEmbeddingFast(32, 16, 24)
        for r, (h, w) in ((rope, (24, 24)), (None, (9, 14))):
            attn = port.Attention(1024, 16, rope=r)
            nn.init.normal_(attn.q_bias)
            nn.init.normal_(attn.v_bias)
            x = torch.randn(2, h, w, 1024)
            with torch.no_grad():
                torch.testing.assert_close(attn(x), ref_attention(attn, x), rtol=1e-4, atol=1e-4)


class StructureAndOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vit = port.CoDETRViT(pretrain_img_size=64, patch_size=16, embed_dim=128, depth=4, num_heads=4,
                                 window_size=4, window_block_indexes=[0, 1, 2])

    def test_shared_rope_buffers_and_output(self):
        keys = self.vit.state_dict().keys()
        self.assertIn('rope_win.freqs_cos', keys)
        self.assertIn('blocks.0.attn.rope.freqs_cos', keys)  # same module, saved per block as upstream does
        self.assertNotIn('blocks.3.attn.rope.freqs_cos', keys)  # global block
        out = self.vit(torch.randn(1, 3, 112, 160))
        self.assertEqual(out[0].shape, (1, 128, 7, 10))

    def test_sfp_levels(self):
        sfp = port.CoDETRSFP([128], 32, 5, use_p2=True)
        outs = sfp([torch.randn(1, 128, 8, 12)])
        self.assertEqual([tuple(o.shape[-2:]) for o in outs], [(32, 48), (16, 24), (8, 12), (4, 6), (2, 3)])

    def test_unported_options_fail(self):
        with self.assertRaises(NotImplementedError):
            port.CoDETRViT(use_lsj=True)

    def test_layer_decay_groups(self):
        model = nn.Module()
        model.backbone = self.vit
        model.head = nn.Linear(4, 4)
        ctor = port.CoDETRViTLayerDecayOptimizerConstructor(
            dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=1.0, weight_decay=0.01)),
            dict(num_layers=4, layer_decay_rate=0.5))
        params = []
        ctor.add_params(params, model)
        lr_of = {id(p): g['lr'] for g in params for p in g['params']}
        wd_of = {id(p): g['weight_decay'] for g in params for p in g['params']}
        # L = 6: patch_embed / pos_embed layer 0 -> 0.5**5, blocks.i -> 0.5**(4 - i), others 1.0
        self.assertAlmostEqual(lr_of[id(self.vit.patch_embed.proj.weight)], 0.5**5)
        self.assertAlmostEqual(lr_of[id(self.vit.pos_embed)], 0.5**5)
        self.assertAlmostEqual(lr_of[id(self.vit.blocks[0].mlp.w1.weight)], 0.5**4)
        self.assertAlmostEqual(lr_of[id(self.vit.blocks[3].mlp.w1.weight)], 0.5)
        self.assertAlmostEqual(lr_of[id(self.vit.out_norm.weight)], 1.0)
        self.assertAlmostEqual(lr_of[id(model.head.weight)], 1.0)
        self.assertEqual(wd_of[id(self.vit.pos_embed)], 0.01)  # upstream's exemption never matches full names
        self.assertEqual(wd_of[id(self.vit.blocks[0].attn.q_bias)], 0.)
        self.assertEqual(wd_of[id(model.head.bias)], 0.)
        self.assertEqual(sum(len(g['params']) for g in params), len(list(model.parameters())))


if __name__ == '__main__':
    unittest.main()
