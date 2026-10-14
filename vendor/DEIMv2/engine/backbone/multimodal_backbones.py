"""
Multimodal backbones for DEIMv2 (AIC) -- three staged experiments.

These implement the three reference-paper methods, staged so each can be tried
independently (per the user's "try one at a time" plan):

  Step 1 -- [1] CSD (Channel Switching + Spatial Attention, CVPR 2023)
            DINOv3STAsCSD: RGB backbone (pretrained) + light IR stem, fused by a
            learned spatial attention gate M(x,y) deciding per-location which
            modality to trust. Channel switching is applied at train time
            (randomly drop the IR branch with probability p).

  Step 2 -- [4] Bi-directional Fusion + Deformation-Sensitive Loss (Info Fusion 2025)
            DINOv3STAsBiDir: same two-stream layout but fuses via a bi-directional
            cross-attention (RGB->IR and IR->RGB) instead of a single gate, which
            is robust to the per-scene geometric misalignment (no hard registration).

  Step 3 -- [5] Collaborating Vision / Depth / Thermal (NeurIPS 2025 D&B)
            DINOv3STAsTriModal: RGB+heat share one DINOv3 stem (in_chans=4), depth
            has its own stem, and the three feature streams interact through a
            cross-modal attention block at each scale. Depth carries geometry,
            IR carries thermal cues, RGB carries appearance -- kept orthogonal at
            the feature level.

All three return (c2, c3, c4) exactly like the original DINOv3STAs so the
HybridEncoder / decoder need no changes.
"""
import os
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F

from functools import partial
from ..core import register
from .dinov3_adapter import DINOv3STAs, SpatialPriorModulev2
from .dinov3 import DinoVisionTransformer


# --------------------------------------------------------------------------- #
# Modality-dropout switch -- also usable as a *valid* feature-space ablation
# --------------------------------------------------------------------------- #
def _zero_stream(training, prob):
    """Whether an auxiliary stream should be zeroed on this forward pass.

    * training + 0 < prob < 1 : ordinary stochastic modality dropout.
    * prob >= 1.0             : always zero, in eval too. This is the switch
      used for a *feature-space* modality ablation.

    Feeding black *pixels* is NOT a valid ablation here: the aux streams are
    conv+BatchNorm stacks, so a constant image comes out of the stem as a
    large out-of-distribution activation (measured: zeroing the IR pixels
    *raises* the IR feature norm from 4451 to 8060), not as "no signal".
    Zeroing the feature is also exactly what the model sees during training
    whenever modality dropout fires.
    """
    if prob >= 1.0:
        return True
    return training and prob > 0.0 and torch.rand(1).item() < prob


# --------------------------------------------------------------------------- #
# Light modality stems (small CNNs producing 3 multi-scale feature maps)
# --------------------------------------------------------------------------- #
class ModalityStem(nn.Module):
    """A tiny CNN that turns a 1-channel modality (IR-heat or depth) into three
    multi-scale feature maps aligned with the DINOv3 output scales (1/8,1/16,1/32
    after the backbone's internal handling). We keep it deliberately small; the
    heavy semantics come from the pretrained RGB DINOv3 stream."""

    def __init__(self, in_chans=1, base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
        )

    def forward(self, x):
        c1 = self.stem(x)      # ~1/4
        c2 = self.conv2(c1)    # ~1/8
        c3 = self.conv3(c2)    # ~1/16
        return c2, c3


# --------------------------------------------------------------------------- #
# Fusion modules
# --------------------------------------------------------------------------- #
class SpatialGate(nn.Module):
    """CSD spatial attention: learn M in [0,1] from concatenated rgb/ir features
    and compute  out = M*rgb + (1-M)*ir."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            # bias=True so the gate can be initialised in favour of RGB
            nn.Conv2d(dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # Init the gate to a near-constant sigmoid(3) ~= 0.95 (trust RGB).
        # The IR stem is trained from scratch, so at step 0 its features are
        # noise; a 50/50 gate would corrupt the pretrained DINOv3 RGB stream.
        # The gate then learns to open up for IR only where it actually helps.
        nn.init.zeros_(self.conv[-2].weight)
        nn.init.constant_(self.conv[-2].bias, 3.0)

    def forward(self, rgb, ir):
        m = self.conv(torch.cat([rgb, ir], dim=1))  # [B,1,H,W]
        return m * rgb + (1.0 - m) * ir


class ECAGate(nn.Module):
    """ECA (Efficient Channel Attention, Wang et al. CVPR 2020) channel-weight
    estimator -- this is the *channel switching* mechanism of CSD (Cao et al.
    CVPR 2023, "Multimodal Object Detection by Channel Switching and Spatial
    Attention").

    Paper spec: per modality m, weight  w_m = sigmoid( f( GAP(X_m) ) )  where f
    is a 1D convolution over the channel dimension. The output is a per-channel
    weight map in [0,1] broadcast over H,W. Output shape [B,C,1,1].
    """

    def __init__(self, dim, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B,C,H,W]
        w = self.avg(x).squeeze(-1).transpose(1, 2)     # [B,1,C]
        w = self.conv(w).transpose(1, 2).unsqueeze(-1)  # [B,C,1,1]
        return self.sigmoid(w)


class CSSA(nn.Module):
    """Channel Switching and Spatial Attention (CSD, Cao et al. CVPR 2023).

    Faithful reproduction of the paper's CSSA fusion module -- the genuine
    method, NOT the hand-made scalar / env gate used before.

      (i) Channel Switching -- each modality gets a per-channel ECA weight; the
          two are made complementary (softmax over the pair) so every channel is
          a *soft switch* between RGB and IR. Where a RGB channel is
          uninformative it is replaced by the corresponding IR channel -- exactly
          the paper's "channels not important for the prediction are replaced by
          the other modality's channel".
      (ii) Spatial Attention -- parameter-free max+avg pooling along the channel
          dimension feeds a 7x7 conv + sigmoid to produce a per-location weight,
          re-weighting the switched feature (CBAM-style spatial attention).

    Unlike a single [B,1] global scalar (the old env gate), CSSA allocates
    modality trust per-channel AND per-location, which is why the paper's gating
    works while the scalar gate collapsed.
    """

    def __init__(self, dim, eca_k=3, sa_kernel=7):
        super().__init__()
        self.eca_rgb = ECAGate(dim, eca_k)
        self.eca_ir = ECAGate(dim, eca_k)
        # CBAM-style spatial attention: 2ch (max,avg) -> 7x7 conv -> sigmoid
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=sa_kernel,
                                 padding=sa_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, ir):
        wr = self.eca_rgb(rgb).float()                  # [B,C,1,1] in float32
        wi = self.eca_ir(ir).float()
        # complementary per-channel switch weight over the (rgb, ir) pair
        alpha = (wr / (wr + wi + 1e-4)).to(rgb.dtype)   # safe from FP16 underflow
        switched = alpha * rgb + (1.0 - alpha) * ir     # per-channel switching
        # spatial attention: max + avg pooling over the channel dimension
        maxp = switched.amax(dim=1, keepdim=True)
        avgp = switched.mean(dim=1, keepdim=True)
        sa = self.sigmoid(self.sa_conv(torch.cat([maxp, avgp], dim=1)))  # [B,1,H,W]
        return sa * switched


class BiDirFusion(nn.Module):
    """Bi-directional cross-attention fusion (RGB<->IR) at the feature level.
    Uses simple scaled dot-product attention over spatial tokens."""

    def __init__(self, dim, heads=4, sr_ratio=2):
        super().__init__()
        self.heads = heads
        self.sr_ratio = sr_ratio
        self.q_rgb = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_ir = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.q_ir = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_rgb = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def _attn(self, q_feat, kv_feat, q_proj, kv_proj, dim):
        # SDPA (flash / memory-efficient kernel) instead of materialising the
        # [B,H,N,N] attention matrix. Same maths as before -- scale=1.0 keeps
        # the original un-scaled q@k -- but memory drops from O(N^2) to O(N).
        # At 768 input the explicit matrix alone needed ~5 GiB and OOM'd.
        b, _, h, w = q_feat.shape
        e = dim // self.heads
        q = q_proj(q_feat).reshape(b, self.heads, e, h * w).transpose(2, 3)
        # Spatial-reduction on the key/value side (PVT-style SRA): the attention
        # matrix is [B,H,Nq,Nkv], so pooling kv by `sr_ratio` cuts that memory
        # by sr_ratio^2. At 768px input with batch 4 the full N x N matrix alone
        # is ~5 GiB in fp32 and OOMs a 16GB card; sr_ratio=2 brings it to ~1.3GB
        # regardless of which SDPA backend is selected.
        if self.sr_ratio > 1:
            kv_feat = F.avg_pool2d(kv_feat, self.sr_ratio, self.sr_ratio)
        n_kv = kv_feat.shape[-2] * kv_feat.shape[-1]
        k, v = kv_proj(kv_feat).chunk(2, dim=1)
        k = k.reshape(b, self.heads, e, n_kv).transpose(2, 3)
        v = v.reshape(b, self.heads, e, n_kv).transpose(2, 3)
        # NOTE: no custom `scale` here on purpose. The flash backend rejects a
        # custom scale and silently falls back to the math backend, which
        # materialises the [B,H,N,N] matrix (~5 GiB at 768px) and OOMs. With
        # the default 1/sqrt(head_dim) we get the O(N) flash kernel. This makes
        # the attention properly scaled (the previous explicit implementation
        # used un-scaled q@k).
        out = F.scaled_dot_product_attention(q, k, v)  # [B,H,N,E]
        return out.transpose(2, 3).reshape(b, dim, h, w)

    def forward(self, rgb, ir):
        f_rgb = self._attn(rgb, ir, self.q_rgb, self.kv_ir, rgb.shape[1])  # rgb queries ir
        f_ir = self._attn(ir, rgb, self.q_ir, self.kv_rgb, ir.shape[1])    # ir queries rgb
        out = self.norm(self.proj(torch.cat([f_rgb, f_ir], dim=1)))
        return out


class CmDRMFusion(nn.Module):
    """Cross-modal Differential Reinforcement Module (CmDRM, VideoFusion CVPR 2026).
    
    Tackles non-rigid / tree foliage parallax and unaligned background noise by:
    1. Mining cross-modal differential feature: F_diff = F_ir - F_rgb (mining thermal hotspots & non-redundancies).
    2. Parallax-Aware Cross-Attention (Soft Spatial Neighborhood Matching):
       Query from RGB (main stream), Key/Value from F_diff (differential stream).
       Uses F.scaled_dot_product_attention (with spatial reduction pooling sr_ratio=2)
       so queries softly attend to local thermal motor hotspots within a neighborhood,
       absorbing 2~4 pixel sub-pixel leaf alignment misalignments without rigid pixel subtraction noise!
    3. Dual Channel Attention (ECA) & Spatial Attention (CBAM-style SA) refinement:
       Refines the differential attention response to eliminate high-frequency fringe artifacts.
    4. Modality Integration Conv:
       Projects the reinforced differential stream together with IR features back to the main feature space.
    """

    def __init__(self, dim, heads=4, sr_ratio=2, sa_kernel=7):
        super().__init__()
        self.heads = heads
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, bias=False)

        self.ca = ECAGate(dim, k=3)
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=sa_kernel, padding=sa_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, rgb, ir):
        # 1. Cross-modal difference mining
        diff = ir - rgb

        # 2. Scaled Dot-Product Cross-Attention (RGB queries Diff)
        b, dim, h, w = rgb.shape
        e = dim // self.heads

        q = self.q_proj(rgb).reshape(b, self.heads, e, h * w).transpose(2, 3)

        diff_kv = diff
        if self.sr_ratio > 1:
            diff_kv = F.avg_pool2d(diff_kv, self.sr_ratio, self.sr_ratio)
        n_kv = diff_kv.shape[-2] * diff_kv.shape[-1]

        k, v = self.kv_proj(diff_kv).chunk(2, dim=1)
        k = k.reshape(b, self.heads, e, n_kv).transpose(2, 3)
        v = v.reshape(b, self.heads, e, n_kv).transpose(2, 3)

        # Flash / memory-efficient attention (O(N) complexity)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        reinforced_diff = attn_out.transpose(2, 3).reshape(b, dim, h, w)

        # 3. Channel & Spatial refinement to eliminate alignment fringe noise
        ca_w = self.ca(reinforced_diff)
        refined = reinforced_diff * ca_w + reinforced_diff

        maxp = refined.amax(dim=1, keepdim=True)
        avgp = refined.mean(dim=1, keepdim=True)
        sa_w = self.sigmoid(self.sa_conv(torch.cat([maxp, avgp], dim=1)))
        gated_diff = sa_w * refined

        # 4. Integrate reinforced differential feature with IR stream
        fused = self.fuse_conv(torch.cat([gated_diff, ir], dim=1))
        return self.norm(fused)


class DiffWeightedFusion(nn.Module):
    """Difference-weighted bi-directional cross-attention fusion.

    Inspired by DMFusion (Ma et al., 2025, "difference-aware modality fusion"):
    instead of averaging the two directional attention streams (50/50), we
    estimate a *per-spatial-location modality dominance* from the RGB/IR
    discrepancy and use it to dynamically weight the two streams. Where the IR
    agrees with RGB the two reinforce each other; where they disagree (e.g. an
    IR-only cold/hot target that RGB misses), the dominant modality is trusted
    more. This is the missing ingredient in the naive BiDirFusion of Step2.
    """

    def __init__(self, dim, heads=4, sr_ratio=2):
        super().__init__()
        self.heads = heads
        self.sr_ratio = sr_ratio
        self.q_rgb = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_ir = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.q_ir = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_rgb = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        # discrepancy -> dominance weights. Small conv on the absolute
        # difference of the two streams; output 2 logits (rgb, ir) -> softmax.
        self.dom = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 2, 1, bias=True),
        )
        # init near uniform dominance (logits ~0) so both streams start trusted
        nn.init.zeros_(self.dom[-1].weight)
        nn.init.zeros_(self.dom[-1].bias)

    def _attn(self, q_feat, kv_feat, q_proj, kv_proj, dim):
        # SDPA (flash / memory-efficient kernel) instead of materialising the
        # [B,H,N,N] attention matrix. Same maths as before -- scale=1.0 keeps
        # the original un-scaled q@k -- but memory drops from O(N^2) to O(N).
        # At 768 input the explicit matrix alone needed ~5 GiB and OOM'd.
        b, _, h, w = q_feat.shape
        e = dim // self.heads
        q = q_proj(q_feat).reshape(b, self.heads, e, h * w).transpose(2, 3)
        # Spatial-reduction on the key/value side (PVT-style SRA): the attention
        # matrix is [B,H,Nq,Nkv], so pooling kv by `sr_ratio` cuts that memory
        # by sr_ratio^2. At 768px input with batch 4 the full N x N matrix alone
        # is ~5 GiB in fp32 and OOMs a 16GB card; sr_ratio=2 brings it to ~1.3GB
        # regardless of which SDPA backend is selected.
        if self.sr_ratio > 1:
            kv_feat = F.avg_pool2d(kv_feat, self.sr_ratio, self.sr_ratio)
        n_kv = kv_feat.shape[-2] * kv_feat.shape[-1]
        k, v = kv_proj(kv_feat).chunk(2, dim=1)
        k = k.reshape(b, self.heads, e, n_kv).transpose(2, 3)
        v = v.reshape(b, self.heads, e, n_kv).transpose(2, 3)
        # NOTE: no custom `scale` here on purpose. The flash backend rejects a
        # custom scale and silently falls back to the math backend, which
        # materialises the [B,H,N,N] matrix (~5 GiB at 768px) and OOMs. With
        # the default 1/sqrt(head_dim) we get the O(N) flash kernel. This makes
        # the attention properly scaled (the previous explicit implementation
        # used un-scaled q@k).
        out = F.scaled_dot_product_attention(q, k, v)  # [B,H,N,E]
        return out.transpose(2, 3).reshape(b, dim, h, w)

    def forward(self, rgb, ir):
        f_rgb = self._attn(rgb, ir, self.q_rgb, self.kv_ir, rgb.shape[1])  # rgb queries ir
        f_ir = self._attn(ir, rgb, self.q_ir, self.kv_rgb, ir.shape[1])    # ir queries rgb
        # discrepancy-weighted dominance: [B,2,H,W] -> [B,2,1,1] global weight
        diff = torch.abs(rgb - ir)
        dom = self.dom(diff)                       # [B,2,H,W]
        w_rgb, w_ir = dom.softmax(dim=1).chunk(2, dim=1)   # each [B,1,H,W]
        fused = w_rgb * f_rgb + w_ir * f_ir       # dynamic per-location weighting
        out = self.norm(self.proj(torch.cat([f_rgb, f_ir], dim=1)))
        # The discrepancy-weighted sum is the real output; the concat branch is a
        # BatchNorm-friendly reference mixed in at a small fixed weight so the BN
        # running stats are not driven by a single modality early in training.
        return 0.5 * out + 0.5 * self.norm(self.proj(torch.cat([fused, fused], dim=1)))


class CrossModalAttention(nn.Module):
    """Cross-modal attention for three streams (rgb+heat, depth). Used by Step3.
    depth acts as the query source attending to the rgb+heat features."""

    def __init__(self, dim, heads=4, sr_ratio=2):
        super().__init__()
        self.heads = heads
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, query_feat, keyval_feat, dim):
        b, _, h, w = query_feat.shape
        q = self.q(query_feat).reshape(b, self.heads, dim // self.heads, h * w)
        k, v = self.kv(keyval_feat).chunk(2, dim=1)
        k = k.reshape(b, self.heads, dim // self.heads, h * w)
        v = v.reshape(b, self.heads, dim // self.heads, h * w)
        attn = (q.transpose(2, 3) @ k).softmax(dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3).reshape(b, dim, h, w)
        return self.norm(self.proj(out))


# --------------------------------------------------------------------------- #
# Step 6: [5] RDTTrack collaboration blocks (orthogonal projection constraint +
#         Fovea prompt) -- cloned from the official RDTTrack repo and adapted to
#         DEIMv2's feature-map backbone interface.
# --------------------------------------------------------------------------- #
class Fovea(nn.Module):
    """Spatial softmax attention (RDTTrack vit_ce_prompt.Fovea)."""

    def __init__(self, smooth=False):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)
        mask = self.softmax(x * self.smooth) if self.smooth else self.softmax(x)
        return (mask * x).contiguous().view(b, c, h, w)


class PromptBlock(nn.Module):
    """[5] RDTTrack Prompt_block: compresses a concatenated (RGB || aux-fused)
    feature map via two 1x1 projections, applies Fovea spatial attention on one
    branch and adds the other as residual, then 1x1 back to the input channel
    count. Input channels must be even (two equal halves)."""

    def __init__(self, inplanes, hide_channel=8, smooth=True, outplanes=None):
        super().__init__()
        # `inplanes` is the TOTAL channel count of the concatenated input
        # (e.g. RGB || aux); each half is therefore `inplanes // 2`. The prompt
        # can optionally be projected back to a different (usually smaller)
        # channel count than the concatenation via `outplanes`, so it can be
        # added onto the RGB feature map. Defaults to `inplanes`.
        self.outplanes = outplanes if outplanes is not None else inplanes
        self.conv0_0 = nn.Conv2d(inplanes // 2, hide_channel, 1, bias=False)
        self.conv0_1 = nn.Conv2d(inplanes // 2, hide_channel, 1, bias=False)
        self.conv1x1 = nn.Conv2d(hide_channel, self.outplanes, 1, bias=False)
        self.fovea = Fovea(smooth=smooth)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        return self.conv1x1(x0)


class OrthogonalProjectionFusion(nn.Module):
    """[5] RDTTrack DepthIR_ort_block: the orthogonal-projection constraint.

    In the original tri-modal (RGB+Depth+Thermal) setting this block is applied
    to the two *auxiliary* modalities (Depth, Thermal) to keep only their
    mutually-orthogonal (complementary, non-redundant) components. Here it is
    adapted to our bimodal (RGB+IR) detection setting by orthogonalising the
    two available streams against each other. Each stream is first projected to
    a low-dim space (1x1 conv); its projection onto the other stream is removed
    (Gram-Schmidt style, weighted by learnable alpha/beta); the two orthogonal
    residuals are concatenated and projected back to `inplanes`.
    """

    def __init__(self, inplanes, hide_channel=8):
        super().__init__()
        self.conv0_0 = nn.Conv2d(inplanes, hide_channel, 1, bias=False)
        self.conv0_1 = nn.Conv2d(inplanes, hide_channel, 1, bias=False)
        self.conv1_1 = nn.Conv2d(hide_channel * 2, inplanes, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, a, b):
        a0 = self.conv0_0(a)
        b0 = self.conv0_1(b)
        # True Gram-Schmidt projection: the projection coefficient is the
        # channel-wise inner product <a0, b0> (summed over C), normalised by the
        # SQUARED norm of the basis vector. The previous form
        #     (a0 * b0 / ||b0||) * b0
        # multiplied elementwise (not an inner product) and divided by ||b0||
        # rather than ||b0||^2, so the term grew like ||b0||^2/||b0|| = ||b0||,
        # i.e. quadratically in the input magnitude -- the fusion input at
        # scale0 had ||dt||/||rgb|| ~ 51x, which blew the projected term up.
        # Channels of a0/b0 are the reduced (hide_channel) dim, so the dot
        # product is taken over dim=1 and kept as a per-pixel scalar map.
        dot_ab = (a0 * b0).sum(dim=1, keepdim=True)          # <a0, b0>  [B,1,H,W]
        n_b2 = (b0 * b0).sum(dim=1, keepdim=True) + 1e-6     # ||b0||^2
        n_a2 = (a0 * a0).sum(dim=1, keepdim=True) + 1e-6     # ||a0||^2
        proj_a_on_b = (dot_ab / n_b2) * b0
        proj_b_on_a = (dot_ab / n_a2) * a0
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        a_orth = a0 - alpha * proj_a_on_b
        b_orth = b0 - beta * proj_b_on_a
        return self.conv1_1(torch.cat([a_orth, b_orth], dim=1))


# --------------------------------------------------------------------------- #
# Shared helper: extract DINOv3 multi-scale features from an RGB (3ch) input
# --------------------------------------------------------------------------- #
def _load_dinov3_sd(weights_path):
    """Load a DEIMv2 DINOv3 checkpoint and strip it down to the `dinov3` stem
    state dict (the checkpoints store everything under `model.backbone.dinov3.*`)."""
    ck = torch.load(weights_path, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    out = {}
    for k, v in sd.items():
        if k.startswith("backbone.dinov3."):
            out[k[len("backbone.dinov3."):]] = v
        elif isinstance(k, str) and not k.startswith(("backbone.", "encoder.", "decoder.")):
            out[k] = v
    return out


def _scale_interaction_indexes(indexes, src_depth, dst_depth):
    """把相对 `src_depth` 的 block 索引按比例映射到 `dst_depth`。

    三条流 (rgb / ir / depth) 共用一组 interaction_indexes, 但 aux 流可以是一个
    更小的 ViT (aux_name -> 不同 depth)。直接用主索引会越界: DINOv3 的
    get_intermediate_layers 只会命中仍存在的前几个 block 并 assert 失败
    ("only 1 / 3 blocks found")。ViT-S 主+aux 时两边都是 12 层所以看不出来,
    RGB 换成 24 层的 ViT-L 而 aux 仍是 12 层就必现。

    映射用 (depth-1) 归一化, 保证最后一个索引落在 dst 的最后一层。
    例如 24 层的 [11,17,23] -> 12 层的 [5,8,11]。
    """
    if src_depth == dst_depth:
        return list(indexes)
    return [min(max(int(round(i * (dst_depth - 1) / (src_depth - 1))), 0), dst_depth - 1)
            for i in indexes]


def _dinov3_features(dinov3, x, interaction_indexes, patch_size):
    H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
    H_toks, W_toks = x.shape[2] // patch_size, x.shape[3] // patch_size
    bs = x.shape[0]
    all_layers = dinov3.get_intermediate_layers(x, n=interaction_indexes, return_class_token=True)
    if len(all_layers) == 1:
        all_layers = [all_layers[0], all_layers[0], all_layers[0]]
    sem_feats = []
    num_scales = len(all_layers) - 2
    for i, sem_feat in enumerate(all_layers):
        feat, _ = sem_feat
        sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
        rh, rw = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
        sem_feat = F.interpolate(sem_feat, size=[rh, rw], mode="bilinear", align_corners=False)
        sem_feats.append(sem_feat)
    return sem_feats  # [c2_feat, c3_feat, c4_feat] before STA concat


def _unfreeze_rgb_with_llrd(rgb_dinov3, n_unfreeze, top_lr=2.0e-5, decay=0.85):
    """解冻 RGB 主干最后 n_unfreeze 个 block + 末层 norm, 并写入**层级衰减(Layer-wise
    LR Decay) lr**。

    为什么需要 LLRD (而不是给解冻块一个固定小 lr):
        ViT / DINOv3 是层次化表征 —— 浅层(block 0..)是边缘/纹理等通用先验, 深层
        (block 23)是语义。在小数据集(~2000 张)上解冻主干, 若所有 block 共用一个 lr,
        浅层极易把预训练好的通用先验一步打散。LLRD 的做法: 越靠前的层 lr 越低(保住
        预训练先验), 越靠后的层(语义层) lr 越高(允许为下游任务适配)。

    lr 在每个 block 的 Parameter 上以属性 `llrd_lr` 记录, 由 optimizer 分组函数
    (engine/core/yaml_config.py::get_optim_params) 读取并据此把该 block 拆成独立
    优化器组。这样无需在 yml 里写一长串易错的正则。

    Args:
        rgb_dinov3 : DinoVisionTransformer 实例 (RGB 主干)
        n_unfreeze : 解冻最后 N 个 block; 超过总 block 数时自动截断
        top_lr     : 最后(第 N 个)block 与末层 norm 的 lr —— 越深越高
        decay      : 每往前一个 block 乘的衰减系数 (0.85 => 6 层跨度约 2.25x)
    """
    blocks = rgb_dinov3.blocks
    n_unfreeze = min(int(n_unfreeze), len(blocks))
    unf_idx = list(range(len(blocks) - n_unfreeze, len(blocks)))
    for gi in unf_idx:
        lr = top_lr * (decay ** (unf_idx[-1] - gi))
        for p in blocks[gi].parameters():
            p.requires_grad_(True)
            p.llrd_lr = float(lr)
    nrm = getattr(rgb_dinov3, 'norm', None)
    if nrm is not None:
        for p in nrm.parameters():
            p.requires_grad_(True)
            p.llrd_lr = float(top_lr)
    rgb_dinov3.train()
    print(f"[unfreeze] RGB 主干解冻最后 {n_unfreeze} block(s) {unf_idx[0]}..{unf_idx[-1]} "
          f"+ norm | LLRD top_lr={top_lr:.1e} decay={decay} "
          f"(block {unf_idx[-1]}={top_lr:.1e} .. block {unf_idx[0]}"
          f"={top_lr * decay ** n_unfreeze:.1e})")
    return n_unfreeze, unf_idx


# --------------------------------------------------------------------------- #
# Step 1: CSD (spatial attention)
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsCSD(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.5,  # channel-switching: drop IR stream at train time
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        # RGB stream: pretrained DINOv3 (in_chans=3, weights preserved)
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[CSD] loaded DINOv3 weights from {weights_path}")
        # freeze RGB backbone for small-data robustness (matches RDTTrack recipe)
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        self.embed_dim = self.rgb_dinov3.embed_dim

        # IR heat stream
        self.ir_stem = ModalityStem(in_chans=1, base=64)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        # CSSA (Channel Switching + Spatial Attention, CSD CVPR2023) fuses rgb & ir
        # at the two finer scales -- the genuine paper module (replaces the ad-hoc
        # SpatialGate used before).
        self.csa = nn.ModuleList([CSSA(self.embed_dim), CSSA(self.embed_dim)])

        # detail (STA) for the rgb stream
        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        # x: [B,4,H,W] = RGB(3) + heat(1)
        rgb = x[:, :3]
        ir = x[:, 3:4]
        bs = x.shape[0]

        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        ir_c2, ir_c3 = self.ir_stem(ir)
        ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                    self.ir_norms[1](self.ir_proj[1](ir_c3))]

        # fuse at the two finer scales (c2, c3) with the genuine CSD CSSA module;
        # c4 uses rgb only. The per-channel ECA switch *inside* CSSA already learns
        # where to trust IR over RGB, so the stochastic IR-dropout below is kept
        # only as a *robustness* augmentation (force the model to cope when IR is
        # missing at test time), orthogonal to the learned channel switching.
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            fused = [rgb_feats[0], rgb_feats[1], rgb_feats[2]]
        else:
            fused = [self.csa[0](rgb_feats[0], ir_feats[0]),
                     self.csa[1](rgb_feats[1], ir_feats[1]),
                     rgb_feats[2]]

        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 2: Bi-directional fusion (+ deformation-sensitive loss handled in solver)
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsBiDir(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.5,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
        self.embed_dim = self.rgb_dinov3.embed_dim

        self.ir_stem = ModalityStem(in_chans=1, base=64)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        self.fuse = nn.ModuleList([BiDirFusion(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:4]
        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        ir_c2, ir_c3 = self.ir_stem(ir)
        ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                    self.ir_norms[1](self.ir_proj[1](ir_c3))]
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            ir_feats = [torch.zeros_like(f) for f in ir_feats]

        fused = [self.fuse[0](rgb_feats[0], ir_feats[0]),
                 self.fuse[1](rgb_feats[1], ir_feats[1]),
                 rgb_feats[2]]
        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 3: Difference-weighted fusion (DMFusion-style) on RAW IR grayscale
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsDiffFuse(nn.Module):
    """Step 3 backbone.

    Two novelties vs Step1/2 that our experiments showed were missing:
      * The 4th channel is the *aligned RAW infrared* (full structure), not the
        heat residual `max(IR-RGB,0)` that discarded ~65% of IR pixels.
      * Fusion is difference-weighted bi-directional cross-attention
        (DiffWeightedFusion): per-location modality dominance estimated from the
        RGB/IR discrepancy, so the network trusts whichever modality is more
        reliable where they disagree (e.g. IR-only targets RGB misses).
    """

    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.15,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
        # freeze RGB backbone for small-data robustness (matches RDTTrack recipe)
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        self.embed_dim = self.rgb_dinov3.embed_dim

        self.ir_stem = ModalityStem(in_chans=1, base=64)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        self.fuse = nn.ModuleList([DiffWeightedFusion(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        # x: [B,4,H,W] = RGB(3) + raw_ir(1)
        rgb = x[:, :3]
        ir = x[:, 3:4]
        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        ir_c2, ir_c3 = self.ir_stem(ir)
        ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                    self.ir_norms[1](self.ir_proj[1](ir_c3))]
        if _zero_stream(self.training, self.ir_drop_prob):
            ir_feats = [torch.zeros_like(f) for f in ir_feats]

        fused = [self.fuse[0](rgb_feats[0], ir_feats[0]),
                 self.fuse[1](rgb_feats[1], ir_feats[1]),
                 rgb_feats[2]]
        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 3: Tri-modal (RGB+heat stem + Depth stem + cross-modal attention)
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsTriModal(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.5,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob

        # RGB + heat share one DINOv3 stem (in_chans=4, expanded from pretrained)
        self.rgbheat_dinov3 = DinoVisionTransformer(name=name, in_chans=4)
        if weights_path and os.path.exists(weights_path):
            sd = _load_dinov3_sd(weights_path)
            w = sd.get("patch_embed.proj.weight")
            if w is not None and w.shape[1] == 3:
                mean_w = w.mean(dim=1, keepdim=True)
                extra = mean_w.expand(-1, 1, -1, -1)
                sd["patch_embed.proj.weight"] = torch.cat([w, extra], dim=1)
            self.rgbheat_dinov3.load_state_dict(sd)
        self.embed_dim = self.rgbheat_dinov3.embed_dim

        # Depth stream
        self.depth_stem = ModalityStem(in_chans=1, base=64)
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.depth_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        # cross-modal attention: depth queries rgb+heat features
        self.cma = nn.ModuleList([CrossModalAttention(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

    def forward(self, x):
        # x: [B,5,H,W] = RGB(3) + heat(1) + depth(1)
        rgbheat = x[:, :4]
        depth = x[:, 4:5]
        rh_feats = _dinov3_features(self.rgbheat_dinov3, rgbheat, self.interaction_indexes, self.patch_size)
        d_c2, d_c3 = self.depth_stem(depth)
        d_feats = [self.depth_norms[0](self.depth_proj[0](d_c2)),
                   self.depth_norms[1](self.depth_proj[1](d_c3))]
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            d_feats = [torch.zeros_like(f) for f in d_feats]

        fused = [self.cma[0](d_feats[0], rh_feats[0], self.embed_dim),
                 self.cma[1](d_feats[1], rh_feats[1], self.embed_dim),
                 rh_feats[2]]
        detail = self.sta(rgbheat[:, :3])
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 4: Cross-modal PRETRAINED IR encoder (addresses the small-data
# overfitting root cause seen in Step1-3).
#
#   * RGB branch  : DINOv3 (S, 384) FROZEN, init from the AIC RGB detection
#                   checkpoint (best_stg2.pth). Its strong object/semantic
#                   prior is preserved and never touched by the 2000 pairs.
#   * IR  branch  : a SEPARATE DINOv3 (S, 384, in_chans=1) initialised from the
#                   SAME RGB checkpoint via a 3->1 channel mean of patch_embed.
#                   This is the cross-modal pretraining trick: the IR encoder
#                   does NOT start from scratch -- it inherits the RGB network's
#                   natural-image / AIC-object priors and only has to adapt to
#                   the IR modality gap on the 2000 pairs. Exactly the
#                   "little data -> lean on pretrained weights" strategy.
#   * Fusion      : DiffWeightedFusion at c2/c3 (same as Step3); c4 = RGB only.
#   * Trainable   : IR branch + fusion + decoder only. RGB frozen.
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsPreIR(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.15,
        rgb_drop_prob=0.0,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob
        self.rgb_drop_prob = rgb_drop_prob

        # ---- RGB branch (FROZEN, pretrained) ----
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[PreIR] loaded RGB (frozen) weights from {weights_path}")
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        self.embed_dim = self.rgb_dinov3.embed_dim

        # ---- IR branch (pretrained from RGB, trainable) ----
        self.ir_dinov3 = DinoVisionTransformer(name=name, in_chans=1)
        if weights_path and os.path.exists(weights_path):
            sd = _load_dinov3_sd(weights_path)
            w = sd.get("patch_embed.proj.weight")
            if w is not None and w.shape[1] == 3:
                sd["patch_embed.proj.weight"] = w.mean(dim=1, keepdim=True)
                print("[PreIR] IR patch_embed initialised from RGB (3->1 channel mean)")
            self.ir_dinov3.load_state_dict(sd)
            print(f"[PreIR] loaded IR (pretrained) weights from {weights_path}")

        self.fuse = nn.ModuleList([DiffWeightedFusion(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

        # sanity: count trainable params (should be IR branch + fusion + decoder)
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[PreIR] trainable params: {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M")

    def forward(self, x):
        # x: [B,4,H,W] = RGB(3) + raw_ir(1)
        rgb = x[:, :3]
        ir = x[:, 3:4]

        # frozen RGB branch -> no grad
        with torch.no_grad():
            rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)

        # trainable IR branch
        ir_feats = _dinov3_features(self.ir_dinov3, ir, self.interaction_indexes, self.patch_size)

        # Day->Night training (the inverse of ir_drop_prob):
        # sometimes the visible image is completely unavailable at night, so we
        # zero the RGB stream and force the IR branch to carry the detection.
        if self.training and torch.rand(1).item() < self.rgb_drop_prob:
            rgb_feats = [torch.zeros_like(f) for f in rgb_feats]

        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            ir_feats = [torch.zeros_like(f) for f in ir_feats]

        fused = [self.fuse[0](rgb_feats[0], ir_feats[0]),
                 self.fuse[1](rgb_feats[1], ir_feats[1]),
                 rgb_feats[2]]
        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 5: PURE INFRARED detection (single modality, cross-modal pretrained).
#
#   A single DINOv3 (S, 384, in_chans=1) is the ONLY backbone -- no RGB branch,
#   no fusion module. It is initialised from the AIC RGB detection checkpoint
#   via a 3->1 channel mean of patch_embed, so the IR stream inherits the RGB
#   network's natural-image / AIC-object priors (the "little data -> lean on
#   pretrained weights" strategy) and only has to adapt to the IR modality gap
#   on the 2000 pairs. All parameters are trainable.
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsPureIR(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_in_chans=1,  # 1 = heat residual only ; 2 = heat + aligned raw IR
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_in_chans = ir_in_chans

        # Single IR backbone, initialised from the AIC RGB checkpoint (3->N mean)
        self.dinov3 = DinoVisionTransformer(name=name, in_chans=ir_in_chans)
        if weights_path and os.path.exists(weights_path):
            sd = _load_dinov3_sd(weights_path)
            w = sd.get("patch_embed.proj.weight")
            if w is not None and w.shape[1] == 3 and w.shape[1] != ir_in_chans:
                m = w.mean(dim=1, keepdim=True)                      # [D,1,K,K]
                sd["patch_embed.proj.weight"] = m.repeat(1, ir_in_chans, 1, 1)
                print(f"[PureIR] IR patch_embed initialised from RGB (3->{ir_in_chans} mean-repeat)")
            # "stem 改长": if the built backbone is deeper than the loaded stem
            # checkpoint, repeat the available blocks to fill the extra depth
            # (block-stacking init). This is how the pure-IR branch gets a longer
            # stem without a larger DINOv3 family -- it reuses the DINOv3-S
            # weights and stacks them (e.g. 24 = 2x the 12-block S stem).
            import re as _re
            _blk_idx = [int(_re.search(r"^blocks\.(\d+)\.", k).group(1))
                        for k in sd if _re.match(r"^blocks\.\d+\.", k)]
            _sd_depth = (max(_blk_idx) + 1) if _blk_idx else 0
            _built_depth = self.dinov3.n_blocks
            if _built_depth > _sd_depth and _sd_depth > 0:
                print(f"[PureIR] deepening IR stem {_sd_depth} -> {_built_depth} "
                      f"(block-repeat init from stem ckpt)")
                for _b in range(_sd_depth, _built_depth):
                    _src = _b % _sd_depth
                    for _k in list(sd.keys()):
                        if _re.match(rf"^blocks\.{_src}\.", _k):
                            sd[_k.replace(f"blocks.{_src}.", f"blocks.{_b}.", 1)] = sd[_k].clone()
            self.dinov3.load_state_dict(sd)
            print(f"[PureIR] loaded IR (pretrained) weights from {weights_path}")
        self.embed_dim = self.dinov3.embed_dim

        # spatial prior (detail) module on the IR input
        self.sta = SpatialPriorModulev2(inplanes=conv_inplane, in_chans=ir_in_chans)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[PureIR] trainable params: {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M")

    def forward(self, x):
        # x: [B,1,H,W] = pure infrared
        feats = _dinov3_features(self.dinov3, x, self.interaction_indexes, self.patch_size)
        detail = self.sta(x)
        out = []
        for i, f in enumerate(feats):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 6: [5] RDTTrack collaboration backbone (bimodal RGB+IR detection)
# --------------------------------------------------------------------------- #
class EnvReliabilityHead(nn.Module):
    """ENV-inspired lightweight RGB-reliability estimator (EvaNet Sec.III-A
    environment-awareness branch).

    Input: raw RGB (+ optional raw IR). Output: scalar r in [0,1] per image.
    Mirrors EvaNet's ENV branch idea -- estimate how trustworthy the visible
    stream is under the current environment (darkness / occlusion) and let that
    drive the RGB-reliability gate. Unlike the analytic brightness gate, this is
    *learned* end-to-end, so it can also down-weight non-luminance defects
    (overexposure, haze, blur). Trained via the detection loss; near-black -> 0
    so the frozen RGB stream yields to the IR/Depth aux (fixes
    full~bothblk@night005).
    """
    def __init__(self, in_chans=3, use_ir=False, hidden=32):
        super().__init__()
        c = in_chans + (1 if use_ir else 0)
        self.use_ir = use_ir
        self.conv = nn.Sequential(
            nn.Conv2d(c, hidden, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, rgb, ir=None):
        x = rgb
        if self.use_ir and ir is not None:
            if ir.shape[-2:] != rgb.shape[-2:]:
                mode = 'area' if ir.shape[-1] > rgb.shape[-1] else 'bilinear'
                ir = F.interpolate(ir, size=rgb.shape[-2:], mode=mode, align_corners=False)
            x = torch.cat([rgb, ir], dim=1)
        return self.fc(self.conv(x)).view(-1, 1)  # [B,1]


class EVANetReliabilityHead(nn.Module):
    """EVANet-style *per-location* RGB-reliability (gate) estimator.

    Combines the two orthogonal axes of EVANet (Cheng et al., IEEE TPAMI 2026,
    "EvaNet: towards More Efficient and Consistent Infrared and Visible Image
    Fusion Assessment") into the LEVER-4 reliability gate:

      (i) Information-preservation branch (the ENV half): how much usable signal
          the visible stream still carries. Near-black RGB -> ~0.
      (ii) Structural-CONSISTENCY branch (the genuinely novel EVANet half): local
          structural agreement between the visible and IR *edge* maps. Where RGB and
          IR agree (well-lit, aligned) the RGB feature is trustworthy -> ~1; where
          they diverge (RGB corrupted by darkness/noise while IR carries a target
          RGB misses) the RGB feature is unreliable -> ~0, so the gate yields those
          regions to the IR/Depth aux injected via the RDTTrack prompt/diffuse.

    Output is a *per-spatial-location* map r_rgb(y,x) in [0,1] (unlike
    EnvReliabilityHead's single scalar), so the gate can suppress ONLY the
    unreliable regions of the RGB feature instead of the whole image -- the key
    fusion-quality-consistency benefit of EVANet. This is the "combine reliability
    gating + EVANet" deliverable: the gate is now driven by EVANet's consistency
    assessment rather than a global scalar.
    """

    def __init__(self, base=32):
        super().__init__()
        # information branch: raw RGB + IR -> signal-ness over the image
        self.rgb_info = nn.Sequential(
            nn.Conv2d(4, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        self.ir_info = nn.Sequential(
            nn.Conv2d(1, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        # consistency branch: fixed-Sobel edge maps of RGB-luma & IR -> agreement
        self.consist = nn.Sequential(
            nn.Conv2d(2, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        # fusion head: info(rgb,ir) + consistency -> single reliability map
        self.head = nn.Sequential(
            nn.Conv2d(base * 3, base, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1, bias=True), nn.Sigmoid(),
        )
        # start near 0.5 (trust both a little) so RGB is not abruptly killed at step 0
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

        # fixed Sobel kernels (no grad) for edge magnitude
        sobel_x = torch.tensor([[[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]]],
                               dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2), persistent=False)

    def _edges(self, x):
        # x: [B,C,H,W] -> per-channel Sobel magnitude, channel-mean -> [B,1,H,W]
        k_x = self.sobel_x.repeat(x.shape[1], 1, 1, 1)
        k_y = self.sobel_y.repeat(x.shape[1], 1, 1, 1)
        ex = F.conv2d(x, k_x, groups=x.shape[1], padding=1)
        ey = F.conv2d(x, k_y, groups=x.shape[1], padding=1)
        mag = torch.sqrt(ex * ex + ey * ey + 1e-6).mean(dim=1, keepdim=True)
        return mag

    def forward(self, rgb, ir, out_shape=None):
        # rgb [B,3,*], ir [B,1,*] are the (already normalized) model inputs
        rgb_s = F.interpolate(rgb, size=(160, 160), mode="area")
        ir_s = F.interpolate(ir, size=(160, 160), mode="area")
        g_rgb = self._edges(rgb_s)            # [B,1,160,160]
        g_ir = self._edges(ir_s)             # [B,1,160,160]
        fi = self.rgb_info(torch.cat([rgb_s, ir_s], dim=1))   # info branch (rgb+ir)
        ci = self.ir_info(ir_s)
        cc = self.consist(torch.cat([g_rgb, g_ir], dim=1))    # consistency branch
        r = self.head(torch.cat([fi, ci, cc], dim=1))         # [B,1,40,40]
        if out_shape is not None:
            r = F.interpolate(r, size=out_shape, mode="bilinear", align_corners=False)
        return r


@register()
class SE2d(nn.Module):
    """Squeeze-and-Excitation channel gate (Hu et al., CVPR 2018).

    Per-channel recalibration of the fused aux feature (dt_ort output): the
    model learns HOW MUCH of each IR/Depth-fused channel to trust before the
    prompt injects it into the RGB stream. ~dim*2/r params, negligible cost.
    """

    def __init__(self, dim, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // r, 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(dim // r, 4), dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class RGBGuidedDCN(nn.Module):
    """RGB-guided deformable alignment of the fused aux feature (DCNv1, Dai 2017).

    The IR/heat features are spatially offset w.r.t. RGB (mem: "两个通道 ir 都是
    歪的"). Offsets are predicted from [rgb || aux] jointly, so the aux feature is
    warped INTO the RGB frame instead of just being smoothed. Residual form with
    ZERO-init convs => exact identity at init (safe to stack on the tuned l4u3
    recipe without disturbing the starting point).
    """

    def __init__(self, dim, k=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(dim * 2, 2 * k * k, 3, padding=k // 2)
        self.dcn = DeformConv2d(dim, dim, k, padding=k // 2)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.dcn.weight)
        if self.dcn.bias is not None:
            nn.init.zeros_(self.dcn.bias)

    def forward(self, aux, rgb):
        offsets = self.offset_conv(torch.cat([rgb, aux], dim=1))
        return aux + self.dcn(aux, offsets)


@register()
class DINOv3STAsRDTTrack(nn.Module):
    """Tri-modal (RGB+IR+Depth) RDTTrack backbone -- cloned from paper [5].

    Paper [5] (RDTTrack, NeurIPS 2025 D&B) ``Collaborating Vision, Depth, and
    Thermal Signals for Multi-Modal Tracking'' collaborates a pretrained RGB
    (vision) stream with TWO auxiliary modalities, Depth and Thermal (TIR). The
    two auxiliary streams are first fused by an *orthogonal projection
    constraint* (DepthTIR_ort_block) that removes their mutual redundancy, then
    the fused aux feature is injected into the RGB backbone via a Fovea *prompt*
    block (Prompt_block). RGB is the frozen/pretrained main stream; only the aux
    stems / fusion / prompts are learned -- exactly the "little data -> lean on
    the pretrained RGB backbone" strategy that suits our ~2000-pair AIC set.

    Our AIC setting maps the paper's (Vision, Depth, Thermal) onto
    (RGB, Depth, IR): this is therefore a faithful RGB + IR + Depth tri-modal
    collaboration, matching the paper instead of the reduced bi-modal variant.
    """

    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        aux_name=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.15,
        depth_drop_prob=0.15,
        depth_in_chans=1,  # 1 = raw depth ; 3 = geometric (normal) encoding
        ir_in_chans=1,     # 1 = heat residual only ; 2 = heat residual + aligned raw IR (PureIR).
                           # Default 1 keeps 5ch configs (RGB+heat+depth, no raw_ir) correctly
                           # aligned; 6ch configs that feed raw IR must set this to 2 explicitly.
        use_modality_stem=False,  # False = full DINOv3 aux (the 'deepir' upgrade);
                                  # True = tiny 3-conv ModalityStem + 1x1 proj (original ufall recipe).
                                  # Single-variable control for "does a full DINOv3 aux help?".
        unfreeze_rgb_stages=0,
        # LLRD hyper-params, now configurable from the yml. Defaults reproduce the
        # previously hard-coded behaviour so existing configs are unaffected.
        llrd_top_lr=2.0e-5,
        llrd_decay=0.85,
        input_norm='none',
        input_norm_mean=0.45,
        input_norm_std=0.225,
        rgb_gate=False,
        rgb_only=False,   # True = strip the multimodal side ENTIRELY: no IR/Depth
        aux_se=False,     # True = SE2d channel gate on the fused aux (dt_ort output)
                          # before prompt injection (c2/c3). cv-improve review pick.
        aux_dcn=False,    # True = RGB-guided DCNv1 deformable alignment of the fused
                          # aux (zero-init residual, identity at start). Treats the
                          # IR/heat spatial offset at feature level.
                          # streams, no ir/depth_norms, no dt_ort, no prompt -- the
                          # network reduces to RGB-DINOv3 -> STA/convs -> encoder.
                          # Unlike ir/depth_drop_prob=1.0 (features computed then
                          # zeroed), this removes the aux compute & parameters, and
                          # also removes the PromptBlock([rgb||0]) residual path.
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob
        self.depth_drop_prob = depth_drop_prob
        self.depth_in_chans = depth_in_chans
        # optional per-image illumination normalisation of the RGB channels only
        # (see _normalize_rgb) -- pairs with, and does not touch, IR/Depth.
        self.input_norm = input_norm
        self.input_norm_mean = input_norm_mean
        self.input_norm_std = input_norm_std
        # LEVER 4: RGB-reliability gate. False | 'brightness' (analytic, w_rgb from
        # raw RGB luminance) | 'env' (learned EnvReliabilityHead, EvaNet ENV-branch
        # inspired, single scalar) | 'evanet' (EVANetReliabilityHead: the full EVANet
        # fusion-quality assessment -- info-preservation + per-location RGB<->IR
        # structural consistency -- driving a *per-spatial-location* gate map).
        # Near-black / inconsistent regions -> r_rgb -> 0 so the frozen RGB stream
        # yields to the IR/Depth aux injected via the prompt (fixes
        # full~bothblk@night005). 'evanet' is the combine-gating+EVANet variant.
        self.rgb_gate = rgb_gate
        if rgb_gate == 'brightness':
            self.rgb_gate_gain = nn.Parameter(torch.tensor(8.0))
            self.rgb_gate_bias = nn.Parameter(torch.tensor(0.2))
        elif rgb_gate == 'env':
            self.rgb_gate_head = EnvReliabilityHead(in_chans=3, use_ir=True)
        elif rgb_gate == 'evanet':
            self.rgb_gate_head = EVANetReliabilityHead()
        # NOTE: rgb_gate='cssa' (the genuine CSD CSSA module) is instantiated
        # further below, after self.embed_dim (the fusion feature dim) is set.

        # ---- Main stream: RGB (vision), pretrained DINOv3 -- FROZEN ----
        # Paper [5] keeps the RGB (vision) backbone frozen/pretrained and only
        # learns the auxiliary stems / orthogonal fusion / Fovea prompts. With
        # our ~2000-pair AIC set, fine-tuning the whole 22.9M-param RGB DINOv3
        # would overfit; freezing preserves its strong object prior (matches the
        # "little data -> lean on pretrained weights" strategy used in Step4).
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[RDTTrack] loaded DINOv3 weights from {weights_path}")
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        # ---- Optional stage-2 unfreeze of the RGB backbone ----
        # Default 0 = keep RGB frozen (stage 1). Set to N>0 to unfreeze the last N
        # transformer blocks (+ final norm) for a low-LR fine-tuning stage.
        # unfreeze_rgb_stages=6 (vitl800) => only the deepest 6 blocks + norm of the
        # 24-layer ViT-L become trainable, with Layer-wise LR Decay (LLRD) applied
        # per block (see _unfreeze_rgb_with_llrd). The optimizer groups them into
        # their own low-LR sub-groups, so the pretrained prior is preserved.
        if unfreeze_rgb_stages and unfreeze_rgb_stages > 0:
            _unfreeze_rgb_with_llrd(self.rgb_dinov3, unfreeze_rgb_stages,
                                    top_lr=llrd_top_lr, decay=llrd_decay)
        self.embed_dim = self.rgb_dinov3.embed_dim
        self.use_modality_stem = use_modality_stem
        self.rgb_only = rgb_only

        # ---- Genuine CSD CSSA (Channel Switching + Spatial Attention) ----
        # Applied between the RGB stream and the orthogonally-fused aux (IR+Depth)
        # stream at the two finer scales. Created here (not in the rgb_gate branch
        # above) so self.embed_dim -- the fusion feature dim -- is already set.
        if self.rgb_gate == 'cssa':
            self.cssa = nn.ModuleList([CSSA(self.embed_dim) for _ in range(2)])

        # ---- Auxiliary streams: IR (thermal) + Depth -------------------------------
        # With rgb_only=True NONE of this section is built: no ir/depth DINOv3 (or
        # stems), no per-scale aux BatchNorms, no dt_ort, no prompt. The model then
        # has zero multimodal parameters/compute -- a genuine single-modality
        # network rather than a zeroed one (cf. ir/depth_drop_prob=1.0, which still
        # runs both aux streams every forward and keeps the PromptBlock residual).
        if not self.rgb_only:
            # Two interchangeable implementations, selected by `use_modality_stem`, so the
            # "IR/Depth = full DINOv3 (deepir) vs tiny 3-conv stem (original ufall)" choice
            # becomes a single clean variable:
            #   * False (default): FULL DINOv3 encoder per modality (the 'deepir' upgrade).
            #        High capacity; RGB-init via patch_embed channel-repeat (3 -> N), baked
            #        into the resume checkpoint by prepare_x_for_rdttrack.py.
            #   * True : ModalityStem (3-conv) + 1x1 proj to embed_dim (original ufall).
            #        Mirrors the CSD/BiDir/TriModal backbones' IR/Depth path exactly; the
            #        stem output (128ch) is projected up to embed_dim so the rest of the
            #        pipeline (dt_ort / prompt / sta) is byte-for-byte identical.
            self.ir_in_chans = ir_in_chans
            self.depth_in_chans = depth_in_chans
            # `aux_name` lets the IR/Depth auxiliary streams use a *smaller* DINOv3
            # than the main RGB stream. Rationale: the aux stems are NOT pretrained on
            # infrared -- they are initialised by channel-repeating the RGB stem's
            # patch_embed -- so enlarging them buys little representation quality while
            # costing a lot of VRAM/compute (all three streams run every forward). With
            # RGB=ViT-B(768) and aux=ViT-S(384) the RGB prior -- the thing that actually
            # is pretrained -- is upgraded, while the learnable aux capacity stays
            # matched to our ~2000-image training set. Default None == same size as RGB.
            self.aux_name = aux_name or name
            if use_modality_stem:
                self.ir_stem = ModalityStem(in_chans=ir_in_chans, base=64)
                self.ir_proj = nn.ModuleList([nn.Conv2d(128, self.embed_dim, 1, bias=False),
                                              nn.Conv2d(128, self.embed_dim, 1, bias=False)])
                self.depth_stem = ModalityStem(in_chans=depth_in_chans, base=64)
                self.depth_proj = nn.ModuleList([nn.Conv2d(128, self.embed_dim, 1, bias=False),
                                                 nn.Conv2d(128, self.embed_dim, 1, bias=False)])
            else:
                self.ir_dinov3 = DinoVisionTransformer(name=self.aux_name, in_chans=ir_in_chans)
                self.depth_dinov3 = DinoVisionTransformer(name=self.aux_name, in_chans=depth_in_chans)
            # When the aux encoders are narrower than the RGB stream, lift their
            # features to `embed_dim` so the downstream BN / orthogonal fusion / prompt
            # (all built at embed_dim) keep working unchanged. Identity when equal.
            if not use_modality_stem:
                _aux_dim = self.ir_dinov3.embed_dim
                # aux 流层数可能和 RGB 不同 (24 层 ViT-L + 12 层 ViT-S), 索引要按比例
                # 缩放, 否则 get_intermediate_layers 只命中前面几个 block 并 assert。
                self.aux_interaction_indexes = _scale_interaction_indexes(
                    interaction_indexes, len(self.rgb_dinov3.blocks), len(self.ir_dinov3.blocks))
                if self.aux_interaction_indexes != list(interaction_indexes):
                    print(f"[RDTTrack] aux interaction_indexes {list(interaction_indexes)} -> "
                          f"{self.aux_interaction_indexes} "
                          f"(RGB {len(self.rgb_dinov3.blocks)} 层 -> aux "
                          f"{len(self.ir_dinov3.blocks)} 层)")
                if _aux_dim != self.embed_dim:
                    self.ir_aux_proj = nn.ModuleList([nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False),
                                                      nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False)])
                    self.depth_aux_proj = nn.ModuleList([nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False),
                                                         nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False)])
                    print(f"[RDTTrack] aux stream '{self.aux_name}' ({_aux_dim}) -> "
                          f"1x1 proj to RGB embed_dim {self.embed_dim}")
                else:
                    # same width as RGB -> no proj needed (aux_interaction_indexes
                    # already computed above; it equals the main one when depths match)
                    self.ir_aux_proj = None
                    self.depth_aux_proj = None
            else:
                # use_modality_stem: no aux DINOv3 at all, indexes are unused
                self.ir_aux_proj = None
                self.depth_aux_proj = None
                self.aux_interaction_indexes = list(interaction_indexes)
            # shared per-scale BatchNorms for both paths
            self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])
            self.depth_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

            # ---- [5] Depth-TIR orthogonal projection constraint (aux vs aux) ----
            # Applied BETWEEN the two auxiliary modalities (IR & Depth) so only their
            # mutually-orthogonal (complementary) components survive. This is the
            # core "orthogonal projection constraint" of RDTTrack -- it must NOT act
            # between RGB and an aux (that would corrupt the frozen RGB prior).
            self.dt_ort = nn.ModuleList([OrthogonalProjectionFusion(self.embed_dim) for _ in range(2)])

            # ---- [5] Fovea prompt: inject the fused aux (D-TIR) into RGB ----
            # Input is the concatenation [RGB || D-TIR]; the prompt is projected back
            # to `embed_dim` so it can be added onto the RGB feature map.
            self.prompt = nn.ModuleList([PromptBlock(self.embed_dim * 2, hide_channel=8,
                                                     smooth=True, outplanes=self.embed_dim)
                                         for _ in range(2)])

            # ---- cv-improve upgrades (single-variable switches, default off) ----
            # Applied on the fused aux (dt_ort output) BEFORE the prompt concat, at
            # the two finer scales, so the RGB prior is untouched and both modules
            # start at/near identity-neutral and only modulate the aux content.
            self.aux_se = nn.ModuleList([SE2d(self.embed_dim) for _ in range(2)]) if aux_se else None
            self.aux_dcn = nn.ModuleList([RGBGuidedDCN(self.embed_dim) for _ in range(2)]) if aux_dcn else None
            if self.aux_se is not None or self.aux_dcn is not None:
                print(f"[RDTTrack] aux upgrades: SE={bool(aux_se)} DCN={bool(aux_dcn)} "
                      f"(on dt_ort output, c2/c3)")
        else:
            print("[RDTTrack] rgb_only=True -- IR/Depth streams, dt_ort, prompt "
                  "NOT built (true single-modality backbone)")

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

        # sanity: RGB backbone must be frozen; only aux stems / fusion / prompts
        # (+ STA detail / projection) should be trainable.
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[RDTTrack] trainable params: {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M "
              f"(RGB backbone frozen={not any(p.requires_grad for p in self.rgb_dinov3.parameters())})")

    def _normalize_rgb(self, rgb):
        """Per-image illumination normalisation of the RGB stream only.

        Night robustness comes from two independent places: the aux modalities
        (IR/Depth, which the prompt injects) and the *invariance of the RGB
        representation itself* to global brightness. The second one is what made
        the RGB-only ``input_norm: instance`` variant (E5) the best model on the
        night curve, so the same trick is offered here. IR/Depth are already
        illumination-invariant and are deliberately left untouched.
        """
        m = rgb.mean(dim=(1, 2, 3), keepdim=True)
        s = rgb.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        return (rgb - m) / s * self.input_norm_std + self.input_norm_mean

    def forward(self, x):
        # x: [B, 3 + ir_in_chans + depth_in_chans, H, W]
        #    = RGB(3) + IR(ir_in_chans: heat residual + raw IR) + Depth(depth_in_chans)
        raw_rgb = x[:, :3]
        rgb = raw_rgb
        if self.input_norm == 'instance':
            rgb = self._normalize_rgb(rgb)
        if not getattr(self, 'rgb_only', False):
            ir = x[:, 3:3 + self.ir_in_chans]
            depth = x[:, 3 + self.ir_in_chans:3 + self.ir_in_chans + self.depth_in_chans]
        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        # [LEVER 4] RGB-reliability gate: suppress the RGB stream when it is
        # unreliable (near-black). Computed from RAW brightness -- input_norm=
        # 'instance' would otherwise re-normalise a black frame to mean ~0.45 and
        # hide the signal. w_rgb in [0,1]; near-black -> 0, normal -> ~1.
        if self.rgb_gate == 'brightness':
            b = raw_rgb.mean(dim=(1, 2, 3), keepdim=True)        # [B,1,1,1] in [0,1]
            w_rgb = torch.sigmoid(self.rgb_gate_gain * (b - self.rgb_gate_bias))
            rgb_feats = [w_rgb * f for f in rgb_feats]
        elif self.rgb_gate == 'env':
            r_rgb = self.rgb_gate_head(raw_rgb, ir)              # [B,1]
            rgb_feats = [r_rgb.view(-1, 1, 1, 1) * f for f in rgb_feats]
        elif self.rgb_gate == 'evanet':
            # per-location reliability map: suppress ONLY the unreliable regions of
            # the RGB feature (EVANet consistency), not the whole image.
            r_map = self.rgb_gate_head(raw_rgb, ir)   # [B,1,Hm,Wm]
            rgb_feats = [F.interpolate(r_map, size=f.shape[-2:], mode='bilinear',
                                       align_corners=False) * f for f in rgb_feats]
        if getattr(self, 'rgb_only', False):
            # ---- RGB-only fast path (multimodal side was never built) ----------
            # Straight through to the STA detail + conv fusion: c2/c3/c4 are pure
            # RGB features, no aux slicing, no dt_ort, no prompt injection.
            detail = self.sta(rgb)
            out = []
            for i, f in enumerate(rgb_feats):
                d = detail[i] if i < len(detail) else detail[-1]
                out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
            return out[0], out[1], out[2]
        # Auxiliary streams -> multi-scale features (c2, c3) for the aux-vs-aux
        # orthogonal fusion. Path follows `use_modality_stem` (see __init__):
        #   * full DINOv3 : _dinov3_features -> 2 finer scales, each at embed_dim
        #   * conv stem   : ModalityStem -> 1x1 proj -> 2 scales, each at embed_dim
        # Both produce `ir_feats` / `d_feats` as 2-element lists at `embed_dim`, so
        # the downstream dt_ort / prompt / sta logic is identical for either path.
        if self.use_modality_stem:
            ir_c2, ir_c3 = self.ir_stem(ir)
            ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                        self.ir_norms[1](self.ir_proj[1](ir_c3))]
            d_c2, d_c3 = self.depth_stem(depth)
            d_feats = [self.depth_norms[0](self.depth_proj[0](d_c2)),
                       self.depth_norms[1](self.depth_proj[1](d_c3))]
        else:
            ir_all = _dinov3_features(self.ir_dinov3, ir, self.aux_interaction_indexes, self.patch_size)
            _ip = getattr(self, 'ir_aux_proj', None)
            ir_feats = [self.ir_norms[i](_ip[i](ir_all[i]) if _ip is not None else ir_all[i])
                        for i in range(2)]
            d_all = _dinov3_features(self.depth_dinov3, depth, self.aux_interaction_indexes, self.patch_size)
            _dp = getattr(self, 'depth_aux_proj', None)
            d_feats = [self.depth_norms[i](_dp[i](d_all[i]) if _dp is not None else d_all[i])
                       for i in range(2)]

        # Missing-modality augmentation (day/night & sensor-dropout robustness):
        # randomly zero an auxiliary stream at train time so the RGB prompt can
        # fall back gracefully when IR or Depth is unavailable at test time.
        if _zero_stream(self.training, self.ir_drop_prob):
            ir_feats = [torch.zeros_like(f) for f in ir_feats]
        if _zero_stream(self.training, self.depth_drop_prob):
            d_feats = [torch.zeros_like(f) for f in d_feats]

        # [5] collaboration:
        #   (1) orthogonal projection constraint between the two aux modalities
        #   (2) Fovea prompt injection of the fused aux into the RGB stream
        # [LEVER] brightness-adaptive aux (prompt) gate. The aux prompt carries
        # IR+Depth; it is essential in the dark (RGB collapses) but mostly noise at
        # mid/bright illumination (RGB already dominates). We re-scale the prompt by
        # a per-image scalar g derived from the *true* RGB brightness (recovered from
        # the ImageNet-normalised input) so that:
        #     dark   (b small) -> g -> cap   (up-weight aux, fill the dark gap)
        #     mid    (b ~ bias) -> g ~ mid   (down-weight aux vs ufall)
        #     bright (b large) -> g -> floor (aux ~ pure noise, suppress it)
        # g = floor + (cap-floor)*sigmoid(gain*(bias - b)), b in [0,1].
        # Disabled unless AUX_GATE=brightness (env); defaults keep g==1 (original).
        aux_gate = os.environ.get('AUX_GATE', '')
        if aux_gate and not getattr(self, '_env_printed', False):
            print(f"[ENV] AUX_GATE set={aux_gate!r}")
            self._env_printed = True
        g = None
        if aux_gate == 'brightness':
            _im = torch.tensor([0.485, 0.456, 0.406], device=raw_rgb.device).view(1, 3, 1, 1)
            _is = torch.tensor([0.229, 0.224, 0.225], device=raw_rgb.device).view(1, 3, 1, 1)
            b = (raw_rgb * _is + _im).mean(dim=(1, 2, 3), keepdim=True)   # true brightness [0,1]
            _gain = float(os.environ.get('AUX_GATE_GAIN', '10'))
            _bias = float(os.environ.get('AUX_GATE_BIAS', '0.12'))
            _floor = float(os.environ.get('AUX_GATE_FLOOR', '0.0'))
            _cap = float(os.environ.get('AUX_GATE_CAP', '1.4'))
            g = torch.sigmoid(_gain * (_bias - b))
            g = _floor + (_cap - _floor) * g
            g = g.view(-1, 1, 1, 1)
            if not getattr(self, '_ag_printed', False):
                print(f"[AUX_GATE] b_mean={b.mean().item():.4f} g_mean={g.mean().item():.4f} "
                      f"gain={_gain} bias={_bias} floor={_floor} cap={_cap}")
                self._ag_printed = True
        fused = []
        for i in range(2):
            dt = self.dt_ort[i](d_feats[i], ir_feats[i])   # aux vs aux orthogonal fusion
            if getattr(self, 'aux_dcn', None) is not None:
                dt = self.aux_dcn[i](dt, rgb_feats[i])     # RGB-guided deformable alignment
            if getattr(self, 'aux_se', None) is not None:
                dt = self.aux_se[i](dt)                    # per-channel trust gate
            if self.rgb_gate == 'cssa':
                # genuine CSD channel-switching + spatial attention between the
                # RGB stream and the fused aux stream (replaces the made-up env
                # scalar gate / the fixed prompt-add injection).
                fused.append(self.cssa[i](rgb_feats[i], dt))
            else:
                prompt = self.prompt[i](torch.cat([rgb_feats[i], dt], dim=1))
                if g is not None:
                    prompt = prompt * g
                fused.append(rgb_feats[i] + prompt)
        fused.append(rgb_feats[2])  # c4: RGB only

        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# RGB-D (bimodal): RGB + Depth ONLY -- the depth-preprocessing isolation rig
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsRDT(nn.Module):
    """RGB-D bimodal backbone: RGB(3) + Depth(depth_in_chans), **no IR**.

    This is ``DINOv3STAsRDTTrack`` with the IR (thermal) branch removed, keeping
    paper [5]'s injection mechanism intact:

      * RGB   : pretrained DINOv3, frozen by default (``unfreeze_rgb_stages`` N>0
                unfreezes the last N blocks + norm for a low-LR stage 2).
      * Depth : a SECOND full DINOv3, ``depth_in_chans`` input channels, whose
                ``patch_embed`` is 3->N channel-mean initialised from the RGB
                detection checkpoint (same cross-modal trick as PureIR/PreIR).
      * Inject: RDTTrack's Fovea ``PromptBlock`` on ``[RGB || Depth]`` added back
                onto the RGB feature map at the two finer scales.

    ``dt_ort`` (aux-vs-aux orthogonal fusion) is gone on purpose -- with a single
    auxiliary modality there is nothing to de-redundify.

    The point of this backbone is *experimental control*: because the only thing
    that changes between runs is the ``depth_folder`` / depth encoding, any mAP
    delta is attributable to depth PREPROCESSING alone.
    """

    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        depth_drop_prob=0.15,
        depth_in_chans=1,   # 1 = single depth channel ; 3 = multi-channel encoding
        unfreeze_rgb_stages=0,
        input_norm='none',
        input_norm_mean=0.45,
        input_norm_std=0.225,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.depth_drop_prob = depth_drop_prob
        self.depth_in_chans = depth_in_chans
        self.input_norm = input_norm
        self.input_norm_mean = input_norm_mean
        self.input_norm_std = input_norm_std

        # ---- Main stream: RGB (vision), pretrained DINOv3 ----
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[RDT] loaded DINOv3 weights from {weights_path}")
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        if unfreeze_rgb_stages and unfreeze_rgb_stages > 0:
            _unfreeze_rgb_with_llrd(self.rgb_dinov3, unfreeze_rgb_stages)
        self.embed_dim = self.rgb_dinov3.embed_dim

        # ---- Auxiliary stream: Depth (full DINOv3, patch_embed 3->N mean) ----
        self.depth_dinov3 = DinoVisionTransformer(name=name, in_chans=depth_in_chans)
        if weights_path and os.path.exists(weights_path):
            sd = _load_dinov3_sd(weights_path)
            w = sd.get("patch_embed.proj.weight")
            if w is not None and w.shape[1] == 3 and w.shape[1] != depth_in_chans:
                m = w.mean(dim=1, keepdim=True)
                sd["patch_embed.proj.weight"] = m.repeat(1, depth_in_chans, 1, 1)
                print(f"[RDT] depth patch_embed initialised from RGB (3->{depth_in_chans} "
                      f"channel mean-repeat)")
            self.depth_dinov3.load_state_dict(sd, strict=False)
            print(f"[RDT] loaded Depth DINOv3 weights from {weights_path}")
        self.depth_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim),
                                          nn.BatchNorm2d(self.embed_dim)])

        # ---- [5] Fovea prompt: inject depth into the RGB feature map ----
        self.prompt = nn.ModuleList([PromptBlock(self.embed_dim * 2, hide_channel=8,
                                                 smooth=True, outplanes=self.embed_dim)
                                     for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[RDT] trainable params: {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M "
              f"(RGB frozen={not any(p.requires_grad for p in self.rgb_dinov3.parameters())})")

    def _normalize_rgb(self, rgb):
        """Per-image illumination normalisation of the RGB stream only."""
        m = rgb.mean(dim=(1, 2, 3), keepdim=True)
        s = rgb.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        return (rgb - m) / s * self.input_norm_std + self.input_norm_mean

    def forward(self, x):
        # x: [B, 3 + depth_in_chans, H, W] = RGB(3) + Depth(depth_in_chans)
        raw_rgb = x[:, :3]
        rgb = self._normalize_rgb(raw_rgb) if self.input_norm == 'instance' else raw_rgb
        depth = x[:, 3:3 + self.depth_in_chans]

        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        d_all = _dinov3_features(self.depth_dinov3, depth, self.interaction_indexes, self.patch_size)
        d_feats = [self.depth_norms[0](d_all[0]), self.depth_norms[1](d_all[1])]

        # missing-modality augmentation: teach the RGB prompt to fall back when
        # depth is unavailable (holes / out-of-range sensor at test time)
        if _zero_stream(self.training, self.depth_drop_prob):
            d_feats = [torch.zeros_like(f) for f in d_feats]

        fused = []
        for i in range(2):
            prompt = self.prompt[i](torch.cat([rgb_feats[i], d_feats[i]], dim=1))
            fused.append(rgb_feats[i] + prompt)
        fused.append(rgb_feats[2])          # c4: RGB only

        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Hybrid: RDTTrack's aux-vs-aux de-redundancy + DiffFuse's feature-level injection
# --------------------------------------------------------------------------- #
@register()
class DINOv3STAsRDTDiffuse(nn.Module):
    """Tri-modal backbone that keeps RDTTrack's *modality* handling but widens
    the *injection* path.

    Why this exists (measured on AIC val, identical protocol):
      * RDTTrack injects the fused aux as a prompt ADDED onto the frozen RGB
        feature (`rgb + prompt`). The residual path is narrow: zeroing IR costs
        only 0.9% mAP (the paper reports ~2% for TIR).
      * DiffFuse REPLACES the RGB feature with a difference-weighted fusion.
        Zeroing IR costs 12.8% mAP on the same protocol.

    So we keep the part of RDTTrack that is genuinely tri-modal -- the
    OrthogonalProjectionFusion between the two AUX modalities (IR vs Depth),
    which removes their mutual redundancy -- and swap the RGB injection from
    "add a prompt" to "difference-weighted feature-level fusion".

    `inject` lets you A/B all three variants from the config, nothing else moves:
      - "prompt"  : original RDTTrack (baseline / sanity check)
      - "diffuse" : DiffWeightedFusion replaces the RGB feature
      - "both"    : average of the two paths
    """

    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        ir_drop_prob=0.15,
        depth_drop_prob=0.15,
        unfreeze_rgb_stages=0,
        inject="diffuse",
    ):
        super().__init__()
        assert inject in ("prompt", "diffuse", "both"), inject
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob
        self.depth_drop_prob = depth_drop_prob
        self.inject = inject

        # ---- Main stream: RGB, pretrained DINOv3 (frozen by default) ----
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[RDTDiffuse] loaded DINOv3 weights from {weights_path}")
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        if unfreeze_rgb_stages and unfreeze_rgb_stages > 0:
            _unfreeze_rgb_with_llrd(self.rgb_dinov3, unfreeze_rgb_stages)
        self.embed_dim = self.rgb_dinov3.embed_dim

        # ---- Auxiliary streams: IR (thermal) and Depth ----
        self.ir_stem = ModalityStem(in_chans=1, base=64)
        self.ir_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        self.depth_stem = ModalityStem(in_chans=1, base=64)
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
            nn.Conv2d(128, self.embed_dim, 1, bias=False),
        ])
        self.depth_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

        # ---- [5] aux-vs-aux orthogonal projection (kept from RDTTrack) ----
        self.dt_ort = nn.ModuleList([OrthogonalProjectionFusion(self.embed_dim) for _ in range(2)])

        # ---- prompt path (kept, only used when inject in {"prompt", "both"}) ----
        self.prompt = nn.ModuleList([PromptBlock(self.embed_dim * 2, hide_channel=8,
                                                 smooth=True, outplanes=self.embed_dim)
                                     for _ in range(2)])

        # ---- feature-level fusion path (from DiffFuse) ----
        self.fuse = nn.ModuleList([DiffWeightedFusion(self.embed_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[RDTDiffuse] inject={inject} trainable {n_tr/1e6:.2f}M / total {n_all/1e6:.1f}M "
              f"(RGB frozen={not any(p.requires_grad for p in self.rgb_dinov3.parameters())})")

    def forward(self, x):
        # x: [B,5,H,W] = RGB(3) + IR(1) + Depth(1)
        rgb = x[:, :3]
        ir = x[:, 3:4]
        depth = x[:, 4:5]

        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)

        ir_c2, ir_c3 = self.ir_stem(ir)
        ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                    self.ir_norms[1](self.ir_proj[1](ir_c3))]
        d_c2, d_c3 = self.depth_stem(depth)
        d_feats = [self.depth_norms[0](self.depth_proj[0](d_c2)),
                   self.depth_norms[1](self.depth_proj[1](d_c3))]

        if _zero_stream(self.training, self.ir_drop_prob):
            ir_feats = [torch.zeros_like(f) for f in ir_feats]
        if _zero_stream(self.training, self.depth_drop_prob):
            d_feats = [torch.zeros_like(f) for f in d_feats]

        fused = []
        for i in range(2):
            dt = self.dt_ort[i](d_feats[i], ir_feats[i])   # aux vs aux de-redundancy
            if self.inject == "prompt":
                fused.append(rgb_feats[i] + self.prompt[i](torch.cat([rgb_feats[i], dt], dim=1)))
            elif self.inject == "both":
                p = rgb_feats[i] + self.prompt[i](torch.cat([rgb_feats[i], dt], dim=1))
                f = self.fuse[i](rgb_feats[i], dt)
                fused.append(0.5 * (p + f))
            else:  # "diffuse": feature-level replacement (wide path)
                fused.append(self.fuse[i](rgb_feats[i], dt))
        fused.append(rgb_feats[2])  # c4: RGB only

        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 2 -- [4] faithful re-implementation: Bi-Directional Progressive Fusion
# (Tang et al., Information Fusion 2026). Paper explicitly uses NO cross-attention;
# fusion is progressive bi-directional (top-down + bottom-up) with Global (conv /
# Transformer self-attention) + Local (hierarchical feature alignment) branches.
# --------------------------------------------------------------------------- #
class BiDirProgFusion(nn.Module):
    """Bi-Directional Progressive Fusion neck (paper [4], faithful).

    Inputs: rgb_feats, ir_feats = list[3] feature maps
            [shallow(~1/8), mid(~1/16), deep(~1/32)].
    Output: fused list[3] at the same 3 scales (paper's P3,P4,P5).

    Stage1 HighLevelPreFusion (deep scale): concat(RGB,IR) + conv + global context.
    Stage2 TopDown: Global G-Fuse-C (RepVGG-style conv) guide + Local THFA
                    (upsample high-level + hook RGB/IR at each scale).
    Stage3 BottomUp: Global G-Fuse-T (self-attention over tokens, C2T->blocks->T2C)
                    guide + Local BHFA.
    No cross-attention between modalities (paper avoids it).
    """

    def __init__(self, dim, hidden=None, nhead=6):
        super().__init__()
        d = dim
        # Stage1 high-level pre-fusion (deep scale): concat(RGB,IR) -> conv
        self.pre = nn.Sequential(
            nn.Conv2d(d * 2, d, 1, bias=False), nn.BatchNorm2d(d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU(),
        )
        # Stage2 global conv (RepVGG-style single-branch 3x3, train-time equivalent
        # to the paper's multi-branch RepVGG). Aggregates all 6 branch features
        # aligned to the mid scale.
        self.gconv = nn.Sequential(
            nn.Conv2d(d * 6, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.GELU(),
        )
        # Local THFA / BHFA: concat(high_level_upsampled, rgb_scale, ir_scale) -> 1x1
        self.lconv = nn.Sequential(nn.Conv2d(d * 3, d, 1, bias=False),
                                   nn.BatchNorm2d(d), nn.GELU())
        # Stage3 global transformer (C2T -> encoder layer -> T2C)
        self.c2t = nn.Conv2d(d * 3, d, 1, bias=False)  # G-Fuse-T: project 3-scale concat (3*d) -> d before tokenizing
        self.t2c = nn.Conv2d(d, d, 1, bias=False)
        self.tblock = nn.TransformerEncoderLayer(d, nhead=nhead, dim_feedforward=d * 2,
                                                 batch_first=True, norm_first=True)
        self.fuse = nn.Conv2d(d * 2, d, 1, bias=False)
        self.norm_out = nn.BatchNorm2d(d)

    def _al(self, f, size):
        return f if f.shape[-2:] == size else F.interpolate(f, size=size, mode='bilinear', align_corners=False)

    def forward(self, rgb_feats, ir_feats):
        rc, rm, rd = rgb_feats          # shallow, mid, deep
        ic, im, id = ir_feats
        # ---- Stage1: high-level pre-fusion (deep) ----
        Fd = self.pre(torch.cat([rd, id], dim=1))
        gp = self._al(F.adaptive_avg_pool2d(Fd, 1), Fd.shape[-2:])
        Fd = Fd + gp                     # global context (multi-scale pool)
        # ---- Stage2: top-down (Global G-Fuse-C + Local THFA) ----
        # Global guide at mid scale: align all 6 branch features to mid and concat
        g_in = torch.cat([rm,
                          self._al(rc, rm.shape[-2:]), self._al(rd, rm.shape[-2:]),
                          self._al(ic, rm.shape[-2:]), self._al(im, rm.shape[-2:]),
                          self._al(id, rm.shape[-2:])], dim=1)   # 6*d
        Gm = self.gconv(g_in)            # global guide, mid scale
        # THFA: Fd upsampled to mid + (rm, im)
        Fd_m = self._al(Fd, rm.shape[-2:])
        Fm_l = self.lconv(torch.cat([Fd_m, rm, im], dim=1))
        Fm = self.norm_out(self.fuse(torch.cat([Gm, Fm_l], dim=1)))
        # THFA shallow: Fm upsampled to shallow + (rc, ic)
        Fm_c = self._al(Fm, rc.shape[-2:])
        Gc = self._al(Gm, rc.shape[-2:])
        Fc_l = self.lconv(torch.cat([Fm_c, rc, ic], dim=1))
        Fc = self.norm_out(self.fuse(torch.cat([Gc, Fc_l], dim=1)))
        # ---- Stage3: bottom-up (Global G-Fuse-T + Local BHFA) ----
        # Global guide at deep scale: align Fd,Fm,Fc; concat; tokens; self-attention
        tok = self.c2t(torch.cat([Fd, self._al(Fm, rd.shape[-2:]),
                                  self._al(Fc, rd.shape[-2:])], dim=1))
        B, d, H, W = tok.shape
        tok = tok.flatten(2).transpose(1, 2)
        tok = self.tblock(tok)
        tok = tok.transpose(1, 2).reshape(B, d, H, W)
        gt = self.t2c(tok)               # global guide, deep scale
        Pm_g = self._al(gt, Fm.shape[-2:])
        Pc_g = self._al(gt, Fc.shape[-2:])
        # BHFA: Fm + (rc,ic) at shallow ; Fd + (rm,im) at mid
        Pc_l = self.lconv(torch.cat([Fm_c, rc, ic], dim=1))
        Pm_l = self.lconv(torch.cat([Fd_m, rm, im], dim=1))
        Pc = self.norm_out(self.fuse(torch.cat([Pc_g, Pc_l], dim=1)))
        Pm = self.norm_out(self.fuse(torch.cat([Pm_g, Pm_l], dim=1)))
        Pd = Fd                        # deepest scale = high-level pre-fusion (paper P3=F3)
        return [Pc, Pm, Pd]


@register()
class DINOv3STAsBiDirPaper(nn.Module):
    """Step 2 faithful re-impl of paper [4] Tang et al. (Bi-Directional Fusion +
    Deformation-Sensitive Loss). Two branches -- RGB (frozen pretrained DINOv3) +
    IR (trainable DINOv3 1ch, initialised from RGB 3->1 mean) -- fused by
    BiDirProgFusion (paper [4] neck, NO cross-attention, NO depth: paper [4] is
    RGB-T only and our depth stream is dead weight at night per ablation).
    Returns (c2,c3,c4) like DINOv3STAs so HybridEncoder/decoder are unchanged.
    """

    def __init__(self, name="dinov3_vits16", weights_path=None,
                 interaction_indexes=[5, 8, 11], embed_dim=384, num_heads=6,
                 patch_size=16, conv_inplane=32, hidden_dim=None, ir_drop_prob=0.15):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob
        # ---- RGB branch (frozen, pretrained) ----
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)
        self.embed_dim = self.rgb_dinov3.embed_dim
        # ---- IR branch (DINOv3 1ch, pretrained from RGB, trainable) ----
        self.ir_dinov3 = DinoVisionTransformer(name=name, in_chans=1)
        if weights_path and os.path.exists(weights_path):
            sd = _load_dinov3_sd(weights_path)
            w = sd.get("patch_embed.proj.weight")
            if w is not None and w.shape[1] == 3:
                sd["patch_embed.proj.weight"] = w.mean(dim=1, keepdim=True)
            self.ir_dinov3.load_state_dict(sd)
            print("[BiDirPaper] IR branch init from RGB (3->1 mean)")
        # ---- BiDir progressive fusion neck ----
        self.fuse = BiDirProgFusion(self.embed_dim, hidden_dim or self.embed_dim, num_heads)
        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[BiDirPaper] trainable {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M")

    def forward(self, x):
        # x: [B,4,H,W] = RGB(3) + raw_IR(1)
        rgb = x[:, :3]
        ir = x[:, 3:4]
        with torch.no_grad():
            rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)
        ir_feats = _dinov3_features(self.ir_dinov3, ir, self.interaction_indexes, self.patch_size)
        if self.training and torch.rand(1).item() < self.ir_drop_prob:
            ir_feats = [torch.zeros_like(f) for f in ir_feats]
        fused = self.fuse(rgb_feats, ir_feats)
        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Step 8: Multimodal Representation Learning & Disentangled Fusion (RepFusion)
# --------------------------------------------------------------------------- #
class SemanticDeformableAlignment(nn.Module):
    """RGB-guided spatial semantic alignment for auxiliary modalities.

    Corrects the 11~43px semantic displacement between the IR thermal core
    (body temperature center) or Depth discontinuity and the RGB optical outline.
    Predicts a dense displacement field (dx, dy) guided by concatenated RGB & Aux features,
    and resamples the Aux feature via bilinear grid sampling.
    Initialized with zero weights/biases so it starts as a perfect identity mapping.
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(dim * 2, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.conv_offset = nn.Conv2d(hidden_dim, 2, 3, padding=1, bias=True)
        # Zero-init so at step 0 offset is identically 0 (pure identity mapping)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, aux, rgb):
        # aux: [B, C, H, W], rgb: [B, C, H, W]
        B, C, H, W = aux.shape
        x = torch.cat([rgb, aux], dim=1)
        feat = self.act(self.bn1(self.conv1(x)))
        offset = self.conv_offset(feat)  # [B, 2, H, W] in pixel displacement
        # Scale pixel displacement to normalized grid coordinates [-1, 1]
        dx = 2.0 * offset[:, 0:1, :, :] / max(W, 1)
        dy = 2.0 * offset[:, 1:2, :, :] / max(H, 1)
        grid_offset = torch.cat([dx, dy], dim=1).permute(0, 2, 3, 1)  # [B, H, W, 2]

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=aux.device, dtype=aux.dtype),
            torch.linspace(-1.0, 1.0, W, device=aux.device, dtype=aux.dtype),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]
        vgrid = base_grid + grid_offset
        return F.grid_sample(aux, vgrid, mode='bilinear', padding_mode='border', align_corners=False)


class DisentangledSubspaceBlock(nn.Module):
    """Shared-Private Subspace Disentanglement Block.
    Projects an input modality feature into:
      - Shared subspace Z_s (inter-modal consistent semantic/boundary features)
      - Private subspace Z_p (modality-specific physical properties: IR thermal, Depth range)
    """
    def __init__(self, in_dim, sub_dim=96):
        super().__init__()
        self.proj_shared = nn.Sequential(
            nn.Conv2d(in_dim, sub_dim, 1, bias=False),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )
        self.proj_private = nn.Sequential(
            nn.Conv2d(in_dim, sub_dim, 1, bias=False),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj_shared(x), self.proj_private(x)


class BoundedConvexGating(nn.Module):
    """Uncertainty & Illumination Aware Bounded Convex Gating.
    Replaces the unbounded additive prompt (rgb + prompt) with:
      1. Bounded Convex Combination of Shared Features:
         f_shared = alpha * Z_s^rgb + beta * Z_s^ir + gamma * Z_s^depth
         where alpha + beta + gamma = 1, alpha, beta, gamma >= 0.
         Driven by raw RGB illumination and feature statistics.
      2. Bounded Modality-Specific Feature Injection:
         f_priv = s * Tanh(Conv([Z_p^ir, Z_p^depth]))
         where s is a learnable scalar initialized to 0.05.
      3. Final fused output:
         f_fused = f_rgb + out_proj(f_shared) + f_priv
    """
    def __init__(self, embed_dim, sub_dim=96, smooth_tau=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.sub_dim = sub_dim
        self.smooth_tau = smooth_tau

        # Gating network: takes pooled shared features + illumination prior
        self.gate_mlp = nn.Sequential(
            nn.Linear(sub_dim * 3 + 1, sub_dim),
            nn.GELU(),
            nn.Linear(sub_dim, 3),
        )
        # Default initialization: slight bias towards RGB
        nn.init.zeros_(self.gate_mlp[-1].weight)
        self.gate_mlp[-1].bias.data = torch.tensor([2.0, 0.0, 0.0])

        self.out_proj = nn.Sequential(
            nn.Conv2d(sub_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

        # Private feature integration
        self.priv_conv = nn.Conv2d(sub_dim * 2, embed_dim, 1, bias=False)
        self.priv_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, f_rgb, s_rgb, s_ir, s_depth, p_ir, p_depth, raw_rgb=None):
        B, _, H, W = s_rgb.shape
        # Environmental illumination prior
        if raw_rgb is not None:
            # Estimate mean luminance in [0, 1]
            lum = raw_rgb.mean(dim=(1, 2, 3), keepdim=True).view(B, 1)  # [B, 1]
        else:
            lum = torch.full((B, 1), 0.5, device=s_rgb.device, dtype=s_rgb.dtype)

        # Pool shared features
        gap_rgb = F.adaptive_avg_pool2d(s_rgb, 1).view(B, -1)
        gap_ir = F.adaptive_avg_pool2d(s_ir, 1).view(B, -1)
        gap_d = F.adaptive_avg_pool2d(s_depth, 1).view(B, -1)

        stat = torch.cat([gap_rgb, gap_ir, gap_d, lum], dim=-1)
        logits = self.gate_mlp(stat) / self.smooth_tau
        weights = F.softmax(logits, dim=-1)  # [B, 3]

        w_rgb = weights[:, 0].view(B, 1, 1, 1)
        w_ir = weights[:, 1].view(B, 1, 1, 1)
        w_d = weights[:, 2].view(B, 1, 1, 1)

        # 1. Bounded Convex Combination of Shared Features
        f_shared = w_rgb * s_rgb + w_ir * s_ir + w_d * s_depth
        f_shared_proj = self.out_proj(f_shared)

        # 2. Bounded Private Feature Injection
        priv_cat = torch.cat([p_ir, p_depth], dim=1)
        f_priv = torch.tanh(self.priv_conv(priv_cat)) * self.priv_scale

        # Combined output
        f_fused = f_rgb + f_shared_proj + f_priv
        return f_fused, weights


@register()
class DINOv3STAsRepFusion(nn.Module):
    """Tri-modal (RGB+IR+Depth) Representation Learning & Disentangled Fusion Backbone.

    Addresses the core problems of previous fusion schemes:
      1. Replaces mechanical pixel-wise inner products with RGB-guided semantic deformable
         alignment (resolves the 11~43px thermal centroid vs optical bbox mismatch).
      2. Disentangles features into shared and private subspaces to prevent interference
         between thermal radiation physics and RGB visual textures.
      3. Replaces unbounded additive prompt injection with uncertainty & illumination
         aware bounded convex gating (w_rgb*S_rgb + w_ir*S_ir + w_d*S_d), ensuring numerical
         stability and preventing noise injection in bright/background scenes.
    """
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        aux_name=None,
        interaction_indexes=[5, 8, 11],
        embed_dim=384,
        num_heads=6,
        patch_size=16,
        conv_inplane=32,
        hidden_dim=None,
        sub_dim=96,
        ir_drop_prob=0.15,
        depth_drop_prob=0.15,
        depth_in_chans=1,
        ir_in_chans=2,
        use_modality_stem=False,
        unfreeze_rgb_stages=0,
        llrd_top_lr=2.0e-5,
        llrd_decay=0.85,
        input_norm='none',
        input_norm_mean=0.45,
        input_norm_std=0.225,
        enable_deformable=True,
        rgb_only=False,
    ):
        super().__init__()
        self.name = name
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        self.ir_drop_prob = ir_drop_prob
        self.depth_drop_prob = depth_drop_prob
        self.depth_in_chans = depth_in_chans
        self.ir_in_chans = ir_in_chans
        self.input_norm = input_norm
        self.input_norm_mean = input_norm_mean
        self.input_norm_std = input_norm_std
        self.use_modality_stem = use_modality_stem
        self.rgb_only = rgb_only
        self.enable_deformable = enable_deformable

        # ---- Main stream: RGB (vision), pretrained DINOv3 (Frozen) ----
        self.rgb_dinov3 = DinoVisionTransformer(name=name, in_chans=3)
        if weights_path and os.path.exists(weights_path):
            self.rgb_dinov3.load_state_dict(_load_dinov3_sd(weights_path))
            print(f"[RepFusion] loaded DINOv3 weights from {weights_path}")
        self.rgb_dinov3.eval()
        self.rgb_dinov3.requires_grad_(False)

        if unfreeze_rgb_stages and unfreeze_rgb_stages > 0:
            _unfreeze_rgb_with_llrd(self.rgb_dinov3, unfreeze_rgb_stages,
                                    top_lr=llrd_top_lr, decay=llrd_decay)
        self.embed_dim = self.rgb_dinov3.embed_dim

        # ---- Auxiliary streams: IR + Depth ----
        if not self.rgb_only:
            self.aux_name = aux_name or name
            if use_modality_stem:
                self.ir_stem = ModalityStem(in_chans=ir_in_chans, base=64)
                self.ir_proj = nn.ModuleList([nn.Conv2d(128, self.embed_dim, 1, bias=False),
                                              nn.Conv2d(128, self.embed_dim, 1, bias=False)])
                self.depth_stem = ModalityStem(in_chans=depth_in_chans, base=64)
                self.depth_proj = nn.ModuleList([nn.Conv2d(128, self.embed_dim, 1, bias=False),
                                                 nn.Conv2d(128, self.embed_dim, 1, bias=False)])
                self.aux_interaction_indexes = list(interaction_indexes)
                self.ir_aux_proj = None
                self.depth_aux_proj = None
            else:
                self.ir_dinov3 = DinoVisionTransformer(name=self.aux_name, in_chans=ir_in_chans)
                self.depth_dinov3 = DinoVisionTransformer(name=self.aux_name, in_chans=depth_in_chans)
                _aux_dim = self.ir_dinov3.embed_dim
                self.aux_interaction_indexes = _scale_interaction_indexes(
                    interaction_indexes, len(self.rgb_dinov3.blocks), len(self.ir_dinov3.blocks))
                if _aux_dim != self.embed_dim:
                    self.ir_aux_proj = nn.ModuleList([nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False),
                                                      nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False)])
                    self.depth_aux_proj = nn.ModuleList([nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False),
                                                         nn.Conv2d(_aux_dim, self.embed_dim, 1, bias=False)])
                else:
                    self.ir_aux_proj = None
                    self.depth_aux_proj = None

            self.ir_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])
            self.depth_norms = nn.ModuleList([nn.BatchNorm2d(self.embed_dim), nn.BatchNorm2d(self.embed_dim)])

            # 1. Semantic Deformable Alignment
            if enable_deformable:
                self.align_ir = nn.ModuleList([SemanticDeformableAlignment(self.embed_dim) for _ in range(2)])
                self.align_depth = nn.ModuleList([SemanticDeformableAlignment(self.embed_dim) for _ in range(2)])
            else:
                self.align_ir = None
                self.align_depth = None

            # 2. Shared-Private Disentangled Subspace Blocks
            self.disentangle_rgb = nn.ModuleList([DisentangledSubspaceBlock(self.embed_dim, sub_dim) for _ in range(2)])
            self.disentangle_ir = nn.ModuleList([DisentangledSubspaceBlock(self.embed_dim, sub_dim) for _ in range(2)])
            self.disentangle_depth = nn.ModuleList([DisentangledSubspaceBlock(self.embed_dim, sub_dim) for _ in range(2)])

            # 3. Bounded Convex Gating Blocks
            self.fusion_gate = nn.ModuleList([BoundedConvexGating(self.embed_dim, sub_dim) for _ in range(2)])

        self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim + conv_inplane * 2, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
            nn.Conv2d(self.embed_dim + conv_inplane * 4, hidden_dim or self.embed_dim, 1, bias=False),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim or self.embed_dim) for _ in range(3)])

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[RepFusion] trainable params: {n_tr/1e6:.1f}M / total {n_all/1e6:.1f}M "
              f"(RGB frozen={not any(p.requires_grad for p in self.rgb_dinov3.parameters())})")

    def _normalize_rgb(self, rgb):
        m = rgb.mean(dim=(1, 2, 3), keepdim=True)
        s = rgb.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        return (rgb - m) / s * self.input_norm_std + self.input_norm_mean

    def forward(self, x):
        raw_rgb = x[:, :3]
        rgb = raw_rgb
        if self.input_norm == 'instance':
            rgb = self._normalize_rgb(rgb)

        rgb_feats = _dinov3_features(self.rgb_dinov3, rgb, self.interaction_indexes, self.patch_size)

        if getattr(self, 'rgb_only', False):
            detail = self.sta(rgb)
            out = []
            for i, f in enumerate(rgb_feats):
                d = detail[i] if i < len(detail) else detail[-1]
                out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
            return out[0], out[1], out[2]

        ir = x[:, 3:3 + self.ir_in_chans]
        depth = x[:, 3 + self.ir_in_chans:3 + self.ir_in_chans + self.depth_in_chans]

        if self.use_modality_stem:
            ir_c2, ir_c3 = self.ir_stem(ir)
            ir_feats = [self.ir_norms[0](self.ir_proj[0](ir_c2)),
                        self.ir_norms[1](self.ir_proj[1](ir_c3))]
            d_c2, d_c3 = self.depth_stem(depth)
            d_feats = [self.depth_norms[0](self.depth_proj[0](d_c2)),
                        self.depth_norms[1](self.depth_proj[1](d_c3))]
        else:
            ir_all = _dinov3_features(self.ir_dinov3, ir, self.aux_interaction_indexes, self.patch_size)
            _ip = getattr(self, 'ir_aux_proj', None)
            ir_feats = [self.ir_norms[i](_ip[i](ir_all[i]) if _ip is not None else ir_all[i])
                        for i in range(2)]
            d_all = _dinov3_features(self.depth_dinov3, depth, self.aux_interaction_indexes, self.patch_size)
            _dp = getattr(self, 'depth_aux_proj', None)
            d_feats = [self.depth_norms[i](_dp[i](d_all[i]) if _dp is not None else d_all[i])
                        for i in range(2)]

        if _zero_stream(self.training, self.ir_drop_prob):
            ir_feats = [torch.zeros_like(f) for f in ir_feats]
        if _zero_stream(self.training, self.depth_drop_prob):
            d_feats = [torch.zeros_like(f) for f in d_feats]

        fused = []
        for i in range(2):
            f_rgb_i = rgb_feats[i]
            f_ir_i = ir_feats[i]
            f_d_i = d_feats[i]

            # 1. Semantic Deformable Alignment
            if self.enable_deformable and self.align_ir is not None:
                f_ir_i = self.align_ir[i](f_ir_i, f_rgb_i)
                f_d_i = self.align_depth[i](f_d_i, f_rgb_i)

            # 2. Shared-Private Disentanglement
            s_rgb, p_rgb = self.disentangle_rgb[i](f_rgb_i)
            s_ir, p_ir = self.disentangle_ir[i](f_ir_i)
            s_d, p_d = self.disentangle_depth[i](f_d_i)

            # 3. Bounded Convex Gating
            f_fused_i, _ = self.fusion_gate[i](
                f_rgb=f_rgb_i,
                s_rgb=s_rgb,
                s_ir=s_ir,
                s_depth=s_d,
                p_ir=p_ir,
                p_depth=p_d,
                raw_rgb=raw_rgb
            )
            fused.append(f_fused_i)

        fused.append(rgb_feats[2])  # c4: RGB only

        detail = self.sta(rgb)
        out = []
        for i, f in enumerate(fused):
            d = detail[i] if i < len(detail) else detail[-1]
            out.append(self.norms[i](self.convs[i](torch.cat([f, d], dim=1))))
        return out[0], out[1], out[2]

