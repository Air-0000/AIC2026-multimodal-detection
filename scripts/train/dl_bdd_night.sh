#!/usr/bin/env bash
export PYTHONPATH="/home/constant/CodeBuddy/Dev/AIC/scripts/lib:${PYTHONPATH:-}"
# =============================================================================
# BDD100K night-image downloader -- plain curl against hf-mirror.
#
# hf_hub_download() churns hundreds of temp files under .cache/huggingface,
# which this environment's bulk-delete guard blocks
# (SAFE_DELETE_BULK_CONFIRM_REQUIRED) and the downloader gets killed. We fetch
# the resolve URLs directly instead, writing to <name>.part and renaming only on
# success -- no deletions at all.
# =============================================================================
set -u
RAW=/home/constant/CodeBuddy/Dev/AIC/data_aux/BDD100K_raw
LIST=$RAW/night_uuids.txt
BASE=https://hf-mirror.com/datasets/dgural/bdd100k/resolve/main/data
mkdir -p "$RAW/data"

if [ ! -s "$LIST" ]; then
  /home/constant/miniconda3/envs/deimv2/bin/python - "$RAW/samples.json" "$LIST" <<'PY'
import json, sys
raw, out = sys.argv[1], sys.argv[2]
s = json.load(open(raw))["samples"]
ids = [x["filepath"].split("/")[-1][:-4] for x in s
       if (x.get("timeofday") or {}).get("label") == "night"]
open(out, "w").write("\n".join(ids) + "\n")
print(len(ids), "night uuids ->", out)
PY
fi

echo "list: $(wc -l < "$LIST") uuids, have $(ls "$RAW/data" | wc -l) images"

export RAW BASE
cat "$LIST" | xargs -P 32 -I{} bash -c '
  f="$RAW/data/{}.jpg"
  [ -s "$f" ] && exit 0
  curl -fsSL --retry 3 --connect-timeout 15 -o "$f.part" "$BASE/{}.jpg" && mv -f "$f.part" "$f"
' 
echo "done: $(ls "$RAW/data" | wc -l) images, $(ls "$RAW/data"/*.part 2>/dev/null | wc -l) partials left"
