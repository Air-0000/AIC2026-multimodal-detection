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
import torch
import torchvision.transforms.v2 as T

from PIL import Image
from .coco_dataset import CocoDetection, ConvertCocoPolysToMask, convert_to_tv_tensor


class CocoDetectionMultiModal(CocoDetection):
    __inject__ = ['transforms']

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        ir_folder=None,
        depth_folder=None,
        return_masks=False,
        n_channels=4,  # 4 = rgb+heat ; 5 = rgb+heat+depth
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
        self.depth_folder = depth_folder
        self.n_channels = n_channels

    def _read_channel(self, folder, name, n_ch):
        p_png = os.path.join(folder, os.path.splitext(name)[0] + ".png")
        p_jpg = os.path.join(folder, name)
        path = p_png if os.path.exists(p_png) else p_jpg
        im = Image.open(path)
        if n_ch == 1:
            im = im.convert("L")
            t = T.functional.pil_to_tensor(im).float() / 255.0  # [1,H,W]
        else:
            im = im.convert("RGB")
            t = T.functional.pil_to_tensor(im).float() / 255.0
            t = t.mean(dim=0, keepdim=True)  # collapse pseudo-color -> 1ch
        return t

    def load_item(self, idx):
        img_name = self.coco.loadImgs(self.ids[idx])[0]["file_name"]
        rgb_path = os.path.join(self.root, img_name)
        rgb_pil = Image.open(rgb_path).convert("RGB")

        # aux channels (resized to RGB size so they always align)
        if self.ir_folder is not None:
            heat = self._read_channel(self.ir_folder, img_name, 1)
            heat = T.functional.resize(heat, rgb_pil.size[::-1])
        else:
            heat = torch.zeros((1, rgb_pil.size[1], rgb_pil.size[0]))
        if self.depth_folder is not None and self.n_channels >= 5:
            dep = self._read_channel(self.depth_folder, img_name, 1)
            dep = T.functional.resize(dep, rgb_pil.size[::-1])
        else:
            dep = torch.zeros((1, rgb_pil.size[1], rgb_pil.size[0]))

        rgb_t = T.functional.pil_to_tensor(rgb_pil).float() / 255.0  # [3,H,W]
        x = torch.cat([rgb_t, heat, dep], dim=0)  # [N,H,W]

        # target parsing (uses RGB pil for size / boxes)
        image_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(self.ids[idx]))
        target = {'image_id': image_id, 'annotations': anno}
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
        return x, target
