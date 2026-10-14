#!/usr/bin/env python3
"""
Download Meta Official DINOv3 ViT-Large pretrained model
Repo: facebook/dinov3-vitl16-pretrain-lvd1689m (~1.21 GB)
"""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    repo = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    target_dir = Path("/home/constant/CodeBuddy/Dev/AIC/weights/dinov3_large")
    target_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Starting DINOv3-ViT-Large snapshot download from {repo} to {target_dir}...")
    sys.stdout.flush()

    try:
        path = snapshot_download(
            repo_id=repo,
            token=token,
            local_dir=str(target_dir),
            max_workers=4
        )
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] SUCCESS! DINOv3-ViT-Large model downloaded to: {path}")
        # Verify files
        for f in target_dir.iterdir():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.2f} MB")
    except Exception as e:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] ERROR downloading DINOv3-ViT-Large: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
