#!/usr/bin/env python3
"""
Automated Submission Importer for AIC 2026 Master Unified Workstation
=====================================================================
Automatically registers newly generated competition submission packages (.zip)
into the workstation model registry and rebuilds the web workstation (index.html),
ensuring any newly generated model predictions are immediately visible on http://localhost:8088/.

Usage:
  # From Python:
  from scripts.viz.import_submission_to_workstation import auto_import_submission
  auto_import_submission("path/to/submission.zip", model_id="v29", name="Sovereign V29", score="59.20")

  # From Shell:
  python3 scripts/viz/import_submission_to_workstation.py --zip path/to/submission.zip --id v29 --name "Sovereign V29"
  python3 scripts/viz/import_submission_to_workstation.py --scan  # Scans and imports all available zips
"""

import os
import sys
import json
import zipfile
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = '/home/constant/CodeBuddy/Dev/AIC'
REGISTRY_FILE = os.path.join(ROOT_DIR, 'work_dir/submissions/workstation_models_registry.json')
BUILD_SCRIPT = os.path.join(ROOT_DIR, 'scripts/viz/build_master_unified_workstation.py')
SERVER_SCRIPT = os.path.join(ROOT_DIR, 'scripts/viz/workstation_server.py')


def inspect_zip(zip_path):
    """Inspects a submission zip file to extract box counts and stats."""
    total_boxes = 0
    scores = []
    cids = set()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for fname in zf.namelist():
            if fname.endswith('.txt'):
                content = zf.read(fname).decode('utf-8', errors='ignore').strip()
                if content:
                    for line in content.split('\n'):
                        p = line.strip().split()
                        if len(p) >= 6:
                            total_boxes += 1
                            cids.add(int(p[0]))
                            scores.append(float(p[5]))
    mean_sc = sum(scores) / len(scores) if scores else 0.0
    return total_boxes, mean_sc, len(cids)


def ensure_server_alive():
    """Checks if the local workstation HTTP server is listening on port 8088, restarts if needed."""
    try:
        res = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8088/"],
                             capture_output=True, text=True, timeout=2)
        if res.stdout.strip() in ("200", "304", "302"):
            return True
    except Exception:
        pass

    print("[Workstation] Port 8088 not responding, launching workstation_server.py in background...")
    log_file = os.path.join(ROOT_DIR, 'logs/workstation_server.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f_out:
        subprocess.Popen(
            [sys.executable, SERVER_SCRIPT],
            cwd=ROOT_DIR,
            stdout=f_out,
            stderr=f_out,
            close_fds=True
        )
    return True


def auto_import_submission(
    zip_path: str,
    model_id: str = None,
    name: str = None,
    score: str = None,
    badge: str = None,
    color: str = None,
    desc: str = None,
    rebuild: bool = True
):
    """
    Registers a submission package into the workstation registry and triggers rebuild.
    """
    zip_path = os.path.abspath(zip_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Submission zip file not found: {zip_path}")

    stem = Path(zip_path).stem
    total_boxes, mean_sc, num_classes = inspect_zip(zip_path)

    if not model_id:
        model_id = stem.replace('submission_', '').replace('sovereign_', '').lower()
    if not name:
        name = stem.replace('_', ' ').title()
    if not badge:
        badge = f"{total_boxes:,} 框 / 均值置信度 {mean_sc:.2f}"
    if not score:
        score = f"{mean_sc:.3f} 均置信"
    if not color:
        # Default nice neon color palette
        color_choices = ['#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#14b8a6']
        idx = hash(model_id) % len(color_choices)
        color = color_choices[idx]
    if not desc:
        desc = f"自动导入提交包: {stem} (含 {total_boxes:,} 个检测框，覆盖 {num_classes} 个类别)"

    model_entry = {
        'id': model_id,
        'name': name,
        'score': score,
        'badge': badge,
        'path': zip_path,
        'color': color,
        'desc': desc
    }

    # Load existing registry
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    registry = []
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        except Exception:
            registry = []

    # Update or append
    updated = False
    for i, item in enumerate(registry):
        if item.get('id') == model_id:
            registry[i] = model_entry
            updated = True
            break
    if not updated:
        registry.append(model_entry)

    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"\n[Workstation Importer] Registered '{name}' ({model_id}) -> {zip_path}")
    print(f"                       Stats: {total_boxes:,} boxes, mean conf {mean_sc:.3f}")

    if rebuild:
        print("[Workstation Importer] Triggering master unified workstation rebuild...")
        ret = subprocess.run([sys.executable, BUILD_SCRIPT], cwd=ROOT_DIR)
        if ret.returncode == 0:
            print("[Workstation Importer] Webstation HTML rebuild complete! Available at http://localhost:8088/\n")
            ensure_server_alive()
        else:
            print("[Workstation Importer] Error rebuilding workstation HTML.")

    return model_entry


def scan_and_import():
    """Scans submissions directories and imports any found packages."""
    search_dirs = [
        os.path.join(ROOT_DIR, 'work_dir/submissions'),
        os.path.join(ROOT_DIR, 'outputs/submissions'),
    ]
    found = []
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                if 'archive' in root.lower() or 'unfiltered' in root.lower():
                    continue
                for f in files:
                    if f.endswith('.zip') and not f.startswith('.'):
                        found.append(os.path.join(root, f))

    print(f"[Workstation Scanner] Found {len(found)} submission packages.")
    for p in sorted(found):
        stem = Path(p).stem
        auto_import_submission(p, rebuild=False)

    # Final single rebuild
    subprocess.run([sys.executable, BUILD_SCRIPT], cwd=ROOT_DIR)
    ensure_server_alive()


def main():
    parser = argparse.ArgumentParser(description="Auto-Import Submission Package to Master Workstation")
    parser.add_argument("--zip", help="Path to submission zip file")
    parser.add_argument("--id", help="Model identifier")
    parser.add_argument("--name", help="Display name")
    parser.add_argument("--score", help="Score or metric badge")
    parser.add_argument("--badge", help="Badge text")
    parser.add_argument("--desc", help="Description")
    parser.add_argument("--color", help="Hex color code")
    parser.add_argument("--scan", action="store_true", help="Scan and import all zips in submissions directories")
    args = parser.parse_args()

    if args.scan:
        scan_and_import()
    elif args.zip:
        auto_import_submission(
            zip_path=args.zip,
            model_id=args.id,
            name=args.name,
            score=args.score,
            badge=args.badge,
            desc=args.desc,
            color=args.color,
            rebuild=True
        )
    else:
        # Default behavior: rebuild workstation with existing configs and ensure server is up
        print("[Workstation] No zip specified, rebuilding master workstation with current registry...")
        subprocess.run([sys.executable, BUILD_SCRIPT], cwd=ROOT_DIR)
        ensure_server_alive()


if __name__ == "__main__":
    main()
