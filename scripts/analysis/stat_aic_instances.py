import json
from collections import Counter

base = '/home/constant/CodeBuddy/Dev/AIC/work_dir/deimv2_dataset'
for split in ['train', 'valid']:
    name = 'train' if split == 'train' else 'val'
    p = f'{base}/{split}/annotations/instances_{name}.json'
    d = json.load(open(p))
    cats = {c['id']: c['name'] for c in d['categories']}
    n_img = len(d['images'])
    n_ann = len(d['annotations'])
    cnt = Counter(a['category_id'] for a in d['annotations'])
    print(f'=== {split} === images={n_img} annotations={n_ann} classes={len(cats)} avg_inst/img={n_ann/max(n_img,1):.2f}')
    for cid, cname in sorted(cats.items(), key=lambda x: cnt.get(x[0], 0)):
        inst = cnt.get(cid, 0)
        imgs_with = len({a['image_id'] for a in d['annotations'] if a['category_id'] == cid})
        print(f'  {cname:12s} inst={inst:5d}  imgs_with_class={imgs_with:5d}')
