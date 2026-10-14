import json, re, sys

path = sys.argv[1]
pat = re.compile(r'"test_coco_eval_bbox": \[([0-9.eE+, -]+)\]')
vals = []
for line in open(path, errors='ignore'):
    m = pat.search(line)
    if m:
        try:
            vals.append(float(m.group(1).split(',')[0]))
        except ValueError:
            pass
for i, v in enumerate(vals):
    print(i, round(v, 4))
print('---tail---')
for i, v in list(enumerate(vals))[-8:]:
    print(i, round(v, 4))
