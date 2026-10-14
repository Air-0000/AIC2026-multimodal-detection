"""Pre-extract and cache Depth-Anything-V2 ViT geometric tokens for offline fast training.

Usage:
  python scripts/prep/prep_dav2_vit_tokens.py \
    --input_dir dataset/train/visible \
    --output_dir work_dir/dav2_token_cache/train \
    --encoder vits \
    --ckpt work_dir/da2_weights/depth_anything_v2_vits.pth \
    --input_size 518
"""

import os
import sys
import glob
import argparse
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor/Depth-Anything-V2"))

from depth_anything_v2.dinov2 import DINOv2

def main():
    parser = argparse.ArgumentParser(description="Pre-extract DAv2 ViT Tokens")
    parser.add_argument("--input_dir", type=str, default=os.path.join(ROOT, "dataset/train/visible"),
                        help="Input images directory (visible, infrared, or depth)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT, "work_dir/dav2_token_cache/train"),
                        help="Directory to save precomputed .pt tokens")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--ckpt", type=str, default=os.path.join(ROOT, "work_dir/da2_weights/depth_anything_v2_vits.pth"))
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading DAv2 {args.encoder} model...")
    vit = DINOv2(args.encoder)
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        vit_sd = {k[len("pretrained."):] if k.startswith("pretrained.") else k: v
                  for k, v in sd.items() if not k.startswith("depth_head.")}
        vit.load_state_dict(vit_sd, strict=False)
        print(f"Successfully loaded weights from {args.ckpt}")
    else:
        print(f"Warning: Checkpoint {args.ckpt} not found! Using random weights.")

    vit = vit.to(device).eval()

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")) +
                         glob.glob(os.path.join(args.input_dir, "*.jpg")))
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    print(f"Found {len(image_paths)} images to process. Saving to: {args.output_dir}")

    # Standard ImageNet normalization for DAv2
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patch_dim = args.input_size // 14

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting DAv2 tokens"):
            stem = os.path.splitext(os.path.basename(path))[0]
            out_file = os.path.join(args.output_dir, f"{stem}.pt")
            if os.path.exists(out_file):
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            # Convert BGR to RGB, resize to multiple of 14
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)

            img_resized = F.interpolate(img_t, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)
            img_norm = (img_resized - mean) / std

            # Extract intermediate layers [5, 8, 11]
            layers = vit.get_intermediate_layers(img_norm, [5, 8, 11], return_class_token=False)

            # Reshape tokens into 2D maps [1, 384, 37, 37]
            tokens_saved = {}
            for i, layer_idx in enumerate([5, 8, 11]):
                tok = layers[i].permute(0, 2, 1).contiguous().view(1, -1, patch_dim, patch_dim)
                tokens_saved[f"layer_{layer_idx}"] = tok.cpu().half()  # fp16 to save disk space

            torch.save(tokens_saved, out_file)

    print(f"Done! All extracted tokens cached in {args.output_dir}")

if __name__ == "__main__":
    main()
