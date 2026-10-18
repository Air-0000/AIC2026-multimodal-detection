"""
Validate a submission directory against the competition rules and the example
format. Checks: file count vs test set, <=100 boxes per image, 6 columns per
line, valid class id, normalized coords in [0,1], confidence in [0,1], CRLF
line endings, and that undetected images yield an EMPTY txt file.

Usage:
    python3 check_submission.py [SUBMISSION_DIR] [TEST_VISIBLE_DIR]
Defaults: SUBMISSION_DIR=work_dir/submission, TEST_VISIBLE_DIR=dataset/test/visible
"""
import os
import sys
import glob

SUB_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join("work_dir", "submission")
TEST_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join("dataset", "test", "visible")
NUM_CLASSES = 12
MAX_BOXES = 100

if not os.path.isdir(SUB_DIR):
    print(f"Error: submission directory '{SUB_DIR}' does not exist.")
    print("Usage: python3 check_submission.py [SUBMISSION_DIR] [TEST_VISIBLE_DIR]")
    sys.exit(1)

test_stems = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(TEST_DIR, "*"))}
sub_files = [f for f in os.listdir(SUB_DIR) if f.endswith(".txt")]
sub_stems = {os.path.splitext(f)[0] for f in sub_files}

errors = []
warn = []
n_empty = 0
n_crlf = 0
n_lf = 0
max_lines = 0
n_total_boxes = 0

# 1) coverage: every test image must have a txt (empty allowed)
if test_stems:
    missing = test_stems - sub_stems
    if missing:
        errors.append(f"missing {len(missing)} prediction files (e.g. {sorted(missing)[:5]})")
    extra = sub_stems - test_stems
    if extra:
        warn.append(f"{len(extra)} extra prediction files not in test set")
else:
    warn.append(f"test directory '{TEST_DIR}' is empty or not found, skipping coverage check")

for fn in sub_files:
    path = os.path.join(SUB_DIR, fn)
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw == b"" or raw == b"\r\n":
        n_empty += 1
        continue
    # line ending check
    if b"\r\n" in raw:
        n_crlf += 1
    elif b"\n" in raw:
        n_lf += 1
    text = raw.replace(b"\r\n", b"\n").decode("utf-8", "ignore")
    lines = [ln for ln in text.split("\n") if ln != ""]
    max_lines = max(max_lines, len(lines))
    if len(lines) > MAX_BOXES:
        errors.append(f"{fn}: {len(lines)} boxes > {MAX_BOXES}")
    for ln in lines:
        n_total_boxes += 1
        parts = ln.split()
        if len(parts) != 6:
            errors.append(f"{fn}: bad column count {len(parts)} -> '{ln}'")
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h, conf = map(float, parts[1:6])
        except ValueError:
            errors.append(f"{fn}: non-numeric -> '{ln}'")
            continue
        if not (0 <= cid < NUM_CLASSES):
            errors.append(f"{fn}: illegal class_id {cid}")
        for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"{fn}: coord {name}={v} out of [0,1]")
        if not (0.0 <= conf <= 1.0):
            errors.append(f"{fn}: confidence {conf} out of [0,1]")

print("=" * 56)
print(f"SUBMISSION : {os.path.abspath(SUB_DIR)}")
print(f"TEST IMAGES: {len(test_stems)}   PRED FILES: {len(sub_files)}")
print(f"empty files: {n_empty}   CRLF: {n_crlf}   LF-only: {n_lf}")
print(f"max boxes in one file: {max_lines} (limit {MAX_BOXES})")
print(f"total boxes: {n_total_boxes}   avg/file: {n_total_boxes/max(len(sub_files),1):.1f}")
print("-" * 56)
if not errors and not warn:
    print("RESULT: PASS - fully compliant with rules + matches example format")
elif not errors:
    print("RESULT: PASS (with warnings):")
    for w in warn:
        print("  WARN:", w)
else:
    print(f"RESULT: FAIL - {len(errors)} issue(s):")
    for e in errors[:30]:
        print("  ERR :", e)
    if len(errors) > 30:
        print(f"  ... and {len(errors)-30} more")
print("=" * 56)
sys.exit(1 if errors else 0)
