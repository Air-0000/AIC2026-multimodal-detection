"""Shared output directories for analysis scripts.

All generated figures (*.png) go to FIG_DIR, all generated summary/text logs go
to LOG_TEXT_DIR.  Evaluation *.log files produced by the run_*_eval.sh scripts
stay under the repo-root logs/ directory (real logs) and are NOT touched here.
"""
import os

ROOT = "/home/constant/CodeBuddy/Dev/AIC"
FIG_DIR = os.path.join(ROOT, "work_dir", "visualizations/figures")
LOG_TEXT_DIR = os.path.join(ROOT, "work_dir", "audits/logs_text")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_TEXT_DIR, exist_ok=True)
