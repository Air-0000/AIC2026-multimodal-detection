"""Imported through custom_imports by the AIC Co-DINO configs: TF32 for matmul and cuDNN (Ampere and newer).

Training stays FP32 otherwise (no autocast): mmcv's deformable-attention kernel is not validated under AMP here.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
