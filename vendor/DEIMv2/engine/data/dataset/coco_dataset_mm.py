"""
Multimodal COCO-style dataset for DEIMv2 (AIC).

Pairs the visible (RGB) image with the precomputed IR "heat" residual and the
depth map by *filename* (visible / infrared / depth dirs share the same names).
Returns a multi-channel tensor:

    Step1/Step2 : [RGB(3), heat(1)]              -> 4 channels
    Step3       : [RGB(3), heat(1), depth(1)]    -> 5 channels

Design notes:
  * The IR heat image (deimv2_heat) is pseudo-color (3ch); we collapse it to a
    single luminance channel. The depth image (deimv2_depth) is already 1ch.
  * We reuse CocoDetection's `ConvertCocoPolysToMask` for target parsing, feeding
    it the RGB PIL so box coordinates / orig_size are computed against the correct
    image size. The multimodal tensor is built separately and returned instead of
    the 3ch RGB tensor.
  * The transform pipeline for this dataset must NOT contain `ConvertPILImage`
    (the image is already a float tensor) and must use an N-channel Normalize
    (see the step configs). Geometric transforms (Resize / HFlip) operate on the
    tensor and therefore apply identically to all channels (alignment preserved).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from PIL import Image
from .coco_dataset import CocoDetection, ConvertCocoPolysToMask, convert_to_tv_tensor, mscoco_category2label
from ...core import register


@register()
class CocoDetectionMultiModal(CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        ir_folder=None,
        raw_ir_folder=None,
        deleak_ir_folder=None,  # 3rd IR channel (optional): de-leaked IR
        depth_folder=None,
        return_masks=False,
        remap_mscoco_category=False,
        n_channels=4,  # 4 = rgb+heat ; 5 = rgb+heat+depth ; 7 = rgb+heat+depth_geo(3)
        modality='rgb_ir',  # 'rgb_ir' = default ; 'ir' = pure infrared (1ch) ;
                            # 'ir_raw' = raw IR only ; 'rgb_depth' = RGB + Depth
                            # bimodal (no IR) -- the depth-preprocessing rig ;
                            # 'rgb' = RGB only (3ch) -- the single-modality control
        depth_mode='raw',  # 'raw' = 1ch depth ; 'geo' = 3ch surface-normal encoding
        depth_read='raw16',  # 'raw16' = correct 16-bit mm read (DEFAULT, fixed) ;
                             # 'clip8' = legacy BROKEN PIL convert("L") read, kept
                             # ONLY to re-evaluate checkpoints trained on clipped maps
        destripe_ir=False,   # 1D Column-wise Guided Destriping for Infrared NUC
        strict_inputs=False,
    ):
        # torchvision CocoDetection only reads annotations; image IO is custom.
        try:
            import torchvision.datasets as TVD
            TVD.CocoDetection.__init__(self, img_folder, ann_file)
        except Exception:
            super(CocoDetection, self).__init__(img_folder, ann_file)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ir_folder = ir_folder
        self.raw_ir_folder = raw_ir_folder
        self.deleak_ir_folder = deleak_ir_folder
        self.depth_folder = depth_folder
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.n_channels = n_channels
        self.modality = modality
        self.depth_mode = depth_mode
        self.depth_read = depth_read
        self.destripe_ir = destripe_ir
        self.strict_inputs = bool(strict_inputs)
        if self.strict_inputs:
            if modality == 'rgb':
                if n_channels != 3 or any(x is not None for x in (ir_folder, raw_ir_folder, deleak_ir_folder, depth_folder)):
                    raise ValueError('Strict RGB-only contract requires 3 channels and no IR/depth folders')
            elif n_channels != 6 or any(x is None for x in (ir_folder, raw_ir_folder, depth_folder)):
                raise ValueError('Strict AIC input contract requires RGB + aligned IR + LBG IR + depth (6 channels)')
            if depth_read != 'raw16' or depth_mode != 'raw':
                raise ValueError('Strict AIC input contract requires raw16/raw depth')
            if destripe_ir:
                from destripe_1d import stripe_non_uniformity_correction_1d

    def _read_channel(self, folder, name, n_ch, destripe=False):
        p_png = os.path.join(folder, os.path.splitext(name)[0] + ".png")
        p_jpg = os.path.join(folder, name)
        path = p_png if os.path.exists(p_png) else p_jpg
        im = Image.open(path)
        if n_ch == 1:
            im = im.convert("L")
            if destripe:
                try:
                    from destripe_1d import stripe_non_uniformity_correction_1d
                    arr = np.array(im)
                    arr = stripe_non_uniformity_correction_1d(arr)
                    t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
                except Exception:
                    if self.strict_inputs:
                        raise
                    t = T.functional.pil_to_tensor(im).float() / 255.0
            else:
                t = T.functional.pil_to_tensor(im).float() / 255.0  # [1,H,W]
        else:
            im = im.convert("RGB")
            t = T.functional.pil_to_tensor(im).float() / 255.0
            t = t.mean(dim=0, keepdim=True)  # collapse pseudo-color -> 1ch
        return t

    def _read_depth_raw16(self, folder, name):
        """Read depth as a [0,1] float map (0 = invalid). Handles 16-bit .png
        (mm) and 8-bit .jpg fallbacks, matching _read_channel's naming.

        This is the ONLY correct way to read AIC depth. Going through
        ``Image.convert("L")`` (i.e. ``_read_channel``) silently clips 16-bit
        millimetre depth to 0-255: measured on dataset/train/depth, ~69% of
        pixels saturate at 255, ~31% are 0 and <0.001% keep an intermediate
        value -- the map degenerates into a *binary* valid/invalid mask that
        carries zero depth information.
        """
        base = os.path.splitext(name)[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(folder, base + ext)
            if os.path.exists(p):
                im = Image.open(p)
                arr = np.array(im).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(axis=2)        # pseudo-color depth -> luminance
                # Decide the scale from Pillow's mode rather than the value
                # range: 'I;16'/'I' sources store millimetres (0..~20000), the
                # 8-bit fallbacks are already normalised to 0..255.
                scale = 19999.0 if getattr(im, 'mode', '') in ('I;16', 'I') else 255.0
                arr = arr / scale
                return torch.from_numpy(arr.clip(0.0, 1.0))[None]  # [1,H,W]
        raise FileNotFoundError(os.path.join(folder, base + '.(png|jpg)'))

    def _depth_to_geo(self, dep_mm):
        """Lever 2: raw 1ch depth -> 3ch surface-normal encoding (no intrinsics).

        Surface normal n = normalize(-dz/dx, -dz/dy, 1) computed with Sobel
        gradients on the (median-inpainted) depth. Returns 3ch in [0, 1].
        Invalid pixels are zeroed. Geometric edges are what a CNN can actually
        learn, unlike the raw (near-flat) depth map.
        """
        d = dep_mm.clone()
        valid = dep_mm > 0
        if valid.sum() > 0:
            d[~valid] = dep_mm[valid].median()
        d = d.clamp(0.0, 1.0)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        d4 = d.unsqueeze(0)                                   # [1,1,H,W]
        dx = F.conv2d(d4, kx, padding=1)[0, 0]                # [H,W]
        dy = F.conv2d(d4, ky, padding=1)[0, 0]
        nz = torch.ones_like(dx)
        n = torch.stack([-dx, -dy, nz], dim=0)
        n = n / n.norm(dim=0, keepdim=True).clamp_min(1e-6)   # [3,H,W] in [-1,1]
        n = n * 0.5 + 0.5                                     # [0,1]
        n[:, ~valid[0]] = 0.0
        return n

    def load_item(self, idx):
        if self.strict_inputs and os.environ.get('ABLATE', ''):
            raise ValueError('ABLATE environment override is forbidden for the clean input contract')
        img_name = self.coco.loadImgs(self.ids[idx])[0]["file_name"]
        rgb_path = os.path.join(self.root, img_name)
        rgb_pil = Image.open(rgb_path).convert("RGB")

        # aux channels (resized to RGB size so they always align)
        if self.ir_folder is not None:
            heat = self._read_channel(self.ir_folder, img_name, 1, destripe=self.destripe_ir)
            heat = T.functional.resize(heat, rgb_pil.size[::-1])
        else:
            heat = torch.zeros((1, rgb_pil.size[1], rgb_pil.size[0]))
        if self.raw_ir_folder is not None:
            # 2nd IR channel: ALIGNED RAW IR (not the residual). Keeps the
            # cold-target / edge structure that max(IR-RGB,0) throws away, so a
            # stem can see both the heat source and the full IR structure.
            raw_ir = self._read_channel(self.raw_ir_folder, img_name, 1, destripe=self.destripe_ir)
            raw_ir = T.functional.resize(raw_ir, rgb_pil.size[::-1])
        else:
            raw_ir = None
        if self.deleak_ir_folder is not None:
            # 3rd IR channel: DE-LEAKED IR (the visible-light ghost removed).
            # Layout becomes [RGB(3), heat, raw_ir, deleak, depth?].
            deleak_ir = self._read_channel(self.deleak_ir_folder, img_name, 1, destripe=self.destripe_ir)
            deleak_ir = T.functional.resize(deleak_ir, rgb_pil.size[::-1])
        else:
            deleak_ir = None
        _use_depth = (self.depth_folder is not None and
                      (self.n_channels >= 5 or self.modality == 'rgb_depth'))
        if _use_depth:
            if self.depth_mode == 'geo':
                dep = self._read_depth_raw16(self.depth_folder, img_name)  # [1,H,W] mm
                dep = T.functional.resize(dep, rgb_pil.size[::-1])
                dep = self._depth_to_geo(dep)                              # [3,H,W]
            elif self.depth_read == 'clip8':
                # LEGACY / BROKEN on purpose: PIL convert("L") clips 16-bit mm
                # depth to 0-255 and collapses the map to a binary mask. Only
                # for re-evaluating checkpoints that were trained on those
                # clipped maps; never use for a new run.
                dep = self._read_channel(self.depth_folder, img_name, 1)
                dep = T.functional.resize(dep, rgb_pil.size[::-1])
            else:
                dep = self._read_depth_raw16(self.depth_folder, img_name)  # [1,H,W]
                dep = T.functional.resize(dep, rgb_pil.size[::-1])
        else:
            _dc = 3 if self.depth_mode == 'geo' else 1
            dep = torch.zeros((_dc, rgb_pil.size[1], rgb_pil.size[0]))

        rgb_t = T.functional.pil_to_tensor(rgb_pil).float() / 255.0  # [3,H,W]
        ir_ch = heat
        if raw_ir is not None:
            ir_ch = torch.cat([ir_ch, raw_ir], dim=0)
        if deleak_ir is not None:
            ir_ch = torch.cat([ir_ch, deleak_ir], dim=0)   # [1|2|3,H,W]
        if self.modality == 'rgb':
            # RGB ONLY: the exact tensor the fused layouts carry in channels 0..2.
            x = rgb_t                                  # [3,H,W]
        elif self.modality == 'ir':
            # PURE INFRARED: detect from the IR channel(s) only (heat residual,
            # + aligned raw IR as the 2nd channel when raw_ir_folder is set).
            # RGB is still read above purely to derive box coordinates /
            # original image size for the target parsing.
            x = ir_ch                                  # [1|2,H,W]
        elif self.modality == 'ir_raw':
            # PURE RAW INFRARED: detect from the ALIGNED RAW IR channel ONLY
            # (drops the heat residual). Isolates the standalone IR detection
            # signal of the full IR structure. raw_ir_folder must be provided.
            x = raw_ir if raw_ir is not None else ir_ch   # [1,H,W]
        elif self.modality == 'rgb_depth':
            # RGB-D BIMODAL: RGB(3) + Depth only -- the depth-preprocessing rig.
            # No IR / heat channel at all, so any mAP delta is attributable to
            # the depth ENCODING alone (strict single-variable control).
            x = torch.cat([rgb_t, dep], dim=0)         # [4|6,H,W]
        elif _use_depth:
            x = torch.cat([rgb_t, ir_ch, dep], dim=0)  # [5|6,H,W]
        else:
            x = torch.cat([rgb_t, ir_ch], dim=0)       # [4|5,H,W]

        # ---- channel ablation (inference-time occlusion) ----
        # set ABLATE=heat|rawir|depth (comma-sep) to fill that channel with its
        # Normalize mean (0.5) so the normalized channel becomes exactly 0 (neutral "off").
        _ab = os.environ.get("ABLATE", "")
        if _ab:
            _idx = {"heat": 3, "rawir": 4, "depth": 5}
            for _tok in _ab.split(","):
                _tok = _tok.strip().lower()
                if _tok == "rgb":
                    # RGB occupies channels 0..2 of the fused 6ch layout; blank
                    # all three together so the RGB branch sees a constant image
                    # (-> no usable information). Only meaningful when the fused
                    # layout is active; silently ignored for pure-IR modalities.
                    if x.shape[0] >= 6:
                        x[0:3] = 0.5
                    continue
                _i = _idx.get(_tok)
                if _i is not None:
                    x[_i] = 0.5

        # target parsing (uses RGB pil for size / boxes)
        image_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(self.ids[idx]))
        target = {'image_id': image_id, 'annotations': anno}
        if self.remap_mscoco_category:
            image, target = self.prepare(rgb_pil, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(rgb_pil, target)

        target['idx'] = torch.tensor([idx])
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes',
                                                   spatial_size=rgb_pil.size[::-1])
        return x, target

    def __getitem__(self, idx):
        x, target = self.load_item(idx)
        if self._transforms is not None:
            x, target, _ = self._transforms(x, target, self)

        # ---- DepthPrior: per-GT-box median depth, threaded to the criterion ----
        # The depth channel is the 5th (index 4) of the transformed tensor. It has
        # already been Normalized as (raw - 0.5) / 0.5, so we map it back to the
        # raw [0, 1] depth used for the depth-prior weights.
        # NOTE (after the 16-bit read fix): this is now ABSOLUTE depth in mm/19999
        # -- near = LOW value, far = HIGH value -- NOT inverse-depth. So
        # depth_invert must be False for the far-object up-weighting
        # (depth_weights=[1,2,4]) to fire on far boxes. We just store the raw median.
        # Depth channel index: RGB(3) + IR(ir_in_chans: 1 heat, or 2 heat+raw)
        _n_ir = 1 + int(self.raw_ir_folder is not None) + int(self.deleak_ir_folder is not None)
        _depth_idx = 3 if self.modality == 'rgb_depth' else 3 + _n_ir
        if self.depth_mode == 'raw' and 'boxes' in target and x.shape[0] > _depth_idx and target['boxes'].shape[0] > 0:
            try:
                dep = (x[_depth_idx] * 0.5 + 0.5).clamp(0.0, 1.0)   # [H, W] raw depth
                H, W = dep.shape[-2:]
                boxes = target['boxes']                    # cxcywh, normalized [0, 1]
                cx, cy, w, h = boxes.unbind(-1)
                x0 = ((cx - w / 2) * W).clamp(0, W - 1)
                y0 = ((cy - h / 2) * H).clamp(0, H - 1)
                x1 = ((cx + w / 2) * W).clamp(0, W - 1)
                y1 = ((cy + h / 2) * H).clamp(0, H - 1)
                dv = []
                for i in range(boxes.shape[0]):
                    yy0, yy1 = int(y0[i].item()), max(int(y1[i].item()) + 1, int(y0[i].item()) + 1)
                    xx0, xx1 = int(x0[i].item()), max(int(x1[i].item()) + 1, int(x0[i].item()) + 1)
                    crop = dep[yy0:yy1, xx0:xx1]
                    dv.append(crop.median() if crop.numel() > 0 else dep.median())
                target['depth'] = torch.stack(dv)
            except Exception:
                if self.strict_inputs:
                    raise
                # never let the depth hook break the dataloader
                pass

        return x, target


@register()
class CocoDetectionMultiModalCached(CocoDetectionMultiModal):
    """Same as CocoDetectionMultiModal but the multi-channel image tensor is read
    from a prebuilt offline cache (.pt) instead of decoding 4 PNGs + resizing +
    stacking per sample.  The cache is produced by prep_deepir_cache.py and holds
    the [6,H,W] float tensor (in [0,1]) at the ORIGINAL image size, so the exact
    same online augmentation pipeline (RandomNight5ch / RandomIoUCrop / Normalize
    ...) applies.  Target parsing reuses the parent's `prepare` against a dummy
    PIL sized from the cached tensor (prepare only reads image.size)."""

    def __init__(self, *args, cache_img_folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_img_folder = cache_img_folder
        assert self.cache_img_folder is not None, "cache_img_folder required for CocoDetectionMultiModalCached"

    def load_item(self, idx):
        img_name = self.coco.loadImgs(self.ids[idx])[0]["file_name"]
        stem = os.path.splitext(img_name)[0]
        cache_path = os.path.join(self.cache_img_folder, stem + ".pt")
        # weights_only: the cache is a single tensor, safe + fast
        x = torch.load(cache_path, map_location="cpu", weights_only=True)  # [C,H,W] uint8
        if not x.is_floating_point():
            x = x.float() / 255.0          # back to float [0,1]
        H, W = x.shape[1], x.shape[2]

        # ---- channel ablation (inference-time occlusion); see parent load_item ----
        _ab = os.environ.get("ABLATE", "")
        if _ab:
            _idx = {"heat": 3, "rawir": 4, "depth": 5}
            for _tok in _ab.split(","):
                _tok = _tok.strip().lower()
                if _tok == "rgb":
                    # RGB occupies channels 0..2 of the fused 6ch layout; blank
                    # all three together so the RGB branch sees a constant image
                    # (-> no usable information). Only meaningful when the fused
                    # layout is active; silently ignored for pure-IR modalities.
                    if x.shape[0] >= 6:
                        x[0:3] = 0.5
                    continue
                _i = _idx.get(_tok)
                if _i is not None:
                    x[_i] = 0.5

        # target parsing uses image.size only -> dummy PIL at cached size
        rgb_dummy = Image.new("RGB", (W, H))
        image_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(self.ids[idx]))
        target = {"image_id": image_id, "annotations": anno}
        if self.remap_mscoco_category:
            image, target = self.prepare(rgb_dummy, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(rgb_dummy, target)

        target["idx"] = torch.tensor([idx])
        if "boxes" in target:
            # NOTE: torchvision BoundingBoxes expects spatial_size as (height, width),
            # i.e. (H, W), matching the non-cached CocoDetectionMultiModal.load_item
            # (rgb_pil.size[::-1]). Using (W, H) here transposes the size and makes
            # torchvision geometry ops (RandomZoomOut / RandomIoUCrop / Mosaic, which
            # activate at later epochs via the stop_epoch policy) raise
            # "Found multiple HxW dimensions" when they compare the image (H, W)
            # against the boxes (W, H).
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes",
                                                   spatial_size=(H, W))
        return x, target

