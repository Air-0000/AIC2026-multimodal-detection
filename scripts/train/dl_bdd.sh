#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# BDD100K val image downloader for a given timeofday split -- plain curl against
# hf-mirror (native hf_hub_download gets killed by this environment's bulk-delete
# guard; see dl_bdd_night.sh). Writes <name>.part then renames on success, so it
# never deletes anything.
#
#   usage: bash $AIC/scripts/train/dl_bdd.sh <split>     # day | night | dawn/dusk
# =============================================================================
set -u
SPLIT="${1:-night}"
RAW=/home/constant/CodeBuddy/Dev/AIC/data_aux/BDD100K_raw
LIST=$RAW/uuid_${SPLIT//\//_}.txt
BASE=https://hf-mirror.com/datasets/dgural/bdd100k/resolve/main/data
mkdir -p "$RAW/data"

if [ ! -s "$LIST" ]; then
  /home/constant/miniconda3/envs/deimv2/bin/python - "$RAW/samples.json" "$LIST" "$SPLIT" <<'PY'
import json, sys
raw, out, split = sys.argv[1], sys.argv[2], sys.argv[3]
s = json.load(open(raw))["samples"]
ids = [x["filepath"].split("/")[-1][:-4] for x in s
       if (x.get("timeofday") or {}).get("label") == split]
open(out, "w").write("\n".join(ids) + "\n")
print(f"{split}: {len(ids)} uuids -> {out}")
PY
fi

echo "split=$SPLIT list=$(wc -l < "$LIST") have=$(ls "$RAW/data" | wc -l)"
export RAW BASE
cat "$LIST" | xargs -P 32 -I{} bash -c '
  f="$RAW/data/{}.jpg"
  [ -s "$f" ] && exit 0
  curl -fsSL --retry 3 --connect-timeout 15 -o "$f.part" "$BASE/{}.jpg" && mv -f "$f.part" "$f"
'
echo "done: $(ls "$RAW/data" | wc -l) images total"
