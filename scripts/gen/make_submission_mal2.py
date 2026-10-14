#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_submission_mal2.py — mal2 (best ckpt) 测试集提交包生成 + val 黄金校验
宿主机运行:
    cd /home/constant/CodeBuddy/Dev/AIC/vendor/DEIMv2
    python ../../scripts/gen/make_submission_mal2.py
产出: work_dir/submissions/submission_mal2/ 下两版 zip
    submit_mal2_A_full.zip    (score > 0.01 全量, 不加 TTA)
    submit_mal2_B_thr050.zip  (score >= 0.5 过滤)
行格式(与官方范例逐字节一致): "class_id cx cy w h score" 归一化 cxcywh, CRLF
熔断: 先用同一条推理管线在 val200 复算 mAP, 与训练 best 偏差 > tol
      则 abort, 不产出任何提交包。
"""
import argparse, copy, glob, hashlib, json, os, sys, zipfile


def log(*a):
    print('[submit]', *a, flush=True)


def fatal(msg):
    print('[submit] FATAL:', msg, flush=True)
    sys.exit(1)


# ------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))          # scripts/gen
REPO = os.path.dirname(os.path.dirname(HERE))              # 仓库根
V2 = os.path.join(REPO, 'vendor', 'DEIMv2')
OUTD = os.path.join(REPO, 'work_dir', 'submissions/submission_mal2')

os.chdir(V2)                                               # DEIMv2 相对路径基准
sys.path.insert(0, V2)                                     # engine / configs
sys.path.insert(0, os.path.join(REPO, 'scripts', 'lib'))   # nightify_aic

import torch
import yaml as pyyaml

ap = argparse.ArgumentParser()
ap.add_argument('--expect-ap', type=float, default=0.5621)
ap.add_argument('--tol', type=float, default=0.003)
ap.add_argument('--ckpt', default=None)
ap.add_argument('--thr-b', type=float, default=0.5)
ap.add_argument('--test-dir', default=os.path.join(REPO, 'dataset/test/AIC2026_PHASE_2_1000/visible'))
args = ap.parse_args()

# ------------------------------------------------- 1. config & ckpt
from engine.core import YAMLConfig

cfg_cands = sorted(glob.glob('configs/deimv2/*mal2*val_orig*.yml'))
if not cfg_cands:
    fatal('找不到 mal2 val 配置')
CONFIG = cfg_cands[0]
log('config =', CONFIG)

EXP = 'outputs/' + os.path.basename(CONFIG).replace('_val_orig.yml', '')
log_path = os.path.join(EXP, 'log.txt')
per_ep = {}
if os.path.isfile(log_path):
    for line in open(log_path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if 'test_coco_eval_bbox' in d and 'epoch' in d:
            per_ep[int(d['epoch'])] = float(d['test_coco_eval_bbox'][0])
best_ep = None
if per_ep:
    best_ep = max(per_ep, key=per_ep.get)
    log('训练 log: %d epochs, best ep%d AP=%.4f' % (len(per_ep), best_ep, per_ep[best_ep]))
else:
    log('WARN: log.txt 无逐 epoch AP, 按文件名 fallback 选 ckpt')

loaded, info = {}, {}
for p in sorted(glob.glob(os.path.join(EXP, '*.pth'))):
    try:
        d = torch.load(p, map_location='cpu', weights_only=False)
        ep = int(d.get('last_epoch', -1))
        loaded[p], info[p] = d, ep
        log('  ckpt %-20s last_epoch=%s' % (os.path.basename(p), ep))
    except Exception as e:
        log('WARN 读 ckpt 失败', os.path.basename(p), repr(e))

CKPT = args.ckpt
if not CKPT and best_ep is not None:
    for p, ep in info.items():
        if ep == best_ep:
            CKPT = p
            break
if not CKPT:
    for name in ('best_stg2.pth', 'best_stg1.pth', 'last.pth'):
        p = os.path.join(EXP, name)
        if os.path.isfile(p):
            CKPT = p
            break
if not CKPT:
    fatal('找不到任何 ckpt')
log('SELECTED ckpt =', CKPT)

# ------------------------------------------------- 2. build model (EMA 优先)
cfg = YAMLConfig(CONFIG)
model = cfg.model
sd = loaded[CKPT]
state, src = None, '?'
ema = sd.get('ema')
if isinstance(ema, dict):
    m = ema.get('module', ema.get('model'))
    if isinstance(m, dict) and any(torch.is_tensor(v) for v in list(m.values())[:5]):
        state, src = m, 'ema.module'
if state is None:
    m = sd.get('model')
    if isinstance(m, dict) and any(torch.is_tensor(v) for v in list(m.values())[:5]):
        state, src = m, 'model'
if state is None:
    fatal('ckpt 无可用权重 dict; keys=%s' % list(sd.keys())[:10])
missing, unexpected = model.load_state_dict(state, strict=False)
log('权重来源=%s missing=%d unexpected=%d' % (src, len(missing), len(unexpected)))
if len(missing) > 20:
    fatal('missing keys 过多(%d): %s' % (len(missing), list(missing)[:8]))
model.eval().cuda()


# ------------------------------------------------- 3. 推理适配器
@torch.no_grad()
def infer(model, inputs, orig=None):
    try:
        out = model(inputs)
    except TypeError:
        out = model(inputs, orig)
    if isinstance(out, dict) and 'pred_logits' in out:
        sc, lb = out['pred_logits'].sigmoid().max(-1)
        return lb, out['pred_boxes'], sc                # 归一化 cxcywh
    if isinstance(out, (list, tuple)):
        if len(out) == 3:
            return out
        if len(out) == 2:
            boxes, sc = out
            if sc.dim() == 3:
                sc, lb = sc.max(-1)
            else:
                lb = torch.zeros(sc.shape[0], sc.shape[1],
                                 dtype=torch.long, device=sc.device)
            return lb, boxes, sc
    raise RuntimeError('未知模型输出: %r' % (type(out),))


def run_loader(loader):
    res, n = {}, 0
    for inputs, targets in loader:
        if torch.is_tensor(inputs):
            inputs = inputs.cuda(non_blocking=True)
        sizes = [list(map(int, t['orig_size'])) for t in targets]
        orig = torch.tensor(sizes).cuda()
        lb, bx, sc = infer(model, inputs, orig)
        if n == 0:
            log('probe: boxes.max=%.4f scores.max=%.4f' % (float(bx.max()), float(sc.max())))
        if float(bx.max()) > 2.0:
            fatal('框坐标 >2 (%.3f) — 疑似绝对坐标口径, 熔断' % float(bx.max()))
        for i, t in enumerate(targets):
            v = t.get('image_id', t.get('id'))
            iid = int(v) if not torch.is_tensor(v) else int(v)
            keep = sc[i] > 0.01
            b = bx[i][keep].float().cpu().numpy()
            s = sc[i][keep].float().cpu().numpy()
            c = lb[i][keep].long().cpu().numpy()
            res[iid] = [(int(cc), float(x), float(y), float(w), float(h), float(ss))
                        for cc, (x, y, w, h), ss in zip(c, b, s)]
        n += len(targets)
    log('推理完成: %d 张' % n)
    return res


# ------------------------------------------------- 4. STEP 1/3 val 黄金校验
log('=== STEP 1/3: val200 黄金校验 ===')
vdl = cfg.val_dataloader
ds = vdl.dataset
coco_gt = getattr(ds, 'coco', None)
if coco_gt is None:
    from pycocotools.coco import COCO
    coco_gt = COCO(str(getattr(ds, 'ann_file')))
id2img = {im['id']: im for im in coco_gt.dataset['images']}
cats = coco_gt.dataset['categories']
log('val ann: %d images / %d classes' % (len(id2img), len(cats)))
log('categories:', {c['id']: c['name'] for c in cats})

val_res = run_loader(vdl)

from pycocotools.cocoeval import COCOeval
dets = []
for iid, items in val_res.items():
    if iid not in id2img:
        continue
    W, H = id2img[iid]['width'], id2img[iid]['height']
    for c, cx, cy, w, h, s in items:
        dets.append({'image_id': iid, 'category_id': int(c),
                     'bbox': [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                     'score': s})
log('val det 条数 =', len(dets))
if not dets:
    fatal('val 复算 0 检出 — 管线异常, 熔断')
coco_dt = coco_gt.loadRes(dets)
E = COCOeval(coco_gt, coco_dt, 'bbox')
E.evaluate()
E.accumulate()
E.summarize()
mAP = float(E.stats[0])
log('GOLD-CHECK: val 复算 mAP=%.4f vs 训练 best=%.4f |diff|=%.4f (tol=%.4f)'
    % (mAP, args.expect_ap, abs(mAP - args.expect_ap), args.tol))
if abs(mAP - args.expect_ap) > args.tol:
    fatal('黄金校验未过 — 推理口径与评测不一致, 已中止不产包')

# ------------------------------------------------- 5. STEP 2/3 测试集推理
log('=== STEP 2/3: 测试集推理 ===')
test_dir = args.test_dir if (args.test_dir and os.path.isdir(args.test_dir)) else os.path.join(REPO, 'dataset', 'test', 'visible')
if not os.path.isdir(test_dir):
    cands = [p for p in glob.glob(os.path.join(REPO, 'dataset', 'test', '**'), recursive=True)
             if os.path.isdir(p)
             and len([f for f in os.listdir(p)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) > 900]
    if not cands:
        fatal('找不到测试图片目录')
    test_dir = sorted(cands)[0]
test_files = sorted(f for f in os.listdir(test_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')))
log('test dir = %s | 图片数 = %d' % (test_dir, len(test_files)))
if len(test_files) != 1000:
    fatal('测试图不是 1000 张 (%d)' % len(test_files))

vann = {'images': [{'id': i + 1, 'file_name': f, 'width': 1920, 'height': 1080}
                   for i, f in enumerate(test_files)],
        'annotations': [], 'categories': cats}
tmp_ann = '/tmp/test_virtual_ann.json'
with open(tmp_ann, 'w') as f:
    json.dump(vann, f)

tmp_yml = 'configs/deimv2/_tmp_test_infer_mal2.yml'
with open(tmp_yml, 'w') as f:
    f.write("__include__: ['%s']\n" % os.path.basename(CONFIG))
    f.write("val_dataloader:\n  dataset:\n")
    f.write("    img_folder: %s\n" % test_dir)
    f.write("    ann_file: %s\n" % tmp_ann)
log('tmp yml ->', tmp_yml)
cfg2 = YAMLConfig(tmp_yml)
tdl = cfg2.val_dataloader
log('test dataset: %s, img_folder=%s'
    % (type(tdl.dataset).__name__, getattr(tdl.dataset, 'img_folder', '?')))
test_res = run_loader(tdl)

# ------------------------------------------------- 6. STEP 3/3 写包
log('=== STEP 3/3: 写包 ===')
stem2id = {f.rsplit('.', 1)[0]: i + 1 for i, f in enumerate(test_files)}


def write_pkg(res, out_dir, smin, zip_path):
    os.makedirs(out_dir, exist_ok=True)
    nbox, per_cls, hist = 0, {}, []
    for stem, iid in stem2id.items():
        items = sorted([it for it in res.get(iid, []) if it[5] >= smin],
                       key=lambda it: -it[5])
        with open(os.path.join(out_dir, stem + '.txt'), 'w', newline='') as f:
            for c, cx, cy, w, h, s in items:
                f.write('%d %.6f %.6f %.6f %.6f %.4f\r\n' % (c, cx, cy, w, h, s))
        nbox += len(items)
        hist.append(len(items))
        for c, *_ in items:
            per_cls[c] = per_cls.get(c, 0) + 1
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out_dir)):
            z.write(os.path.join(out_dir, fn), fn)
    md5 = hashlib.md5(open(zip_path, 'rb').read()).hexdigest()
    nh = sorted(hist)
    log('%s: %d txt / %d 框 / 类分布=%s / 框数 p10=%d p50=%d p90=%d / md5=%s'
        % (os.path.basename(zip_path), len(os.listdir(out_dir)), nbox,
           dict(sorted(per_cls.items())),
           nh[len(nh) // 10], nh[len(nh) // 2], nh[len(nh) * 9 // 10], md5))
    for stem in sorted(os.listdir(out_dir))[:2]:
        with open(os.path.join(out_dir, stem), 'rb') as f:
            log('  sample %s: %r' % (stem, f.read(110)))


os.makedirs(OUTD, exist_ok=True)
write_pkg(test_res, os.path.join(OUTD, 'submit_A_full'), 0.01,
          os.path.join(OUTD, 'submit_mal2_A_full.zip'))
write_pkg(test_res, os.path.join(OUTD, 'submit_B_thr050'), args.thr_b,
          os.path.join(OUTD, 'submit_mal2_B_thr050.zip'))
write_pkg(test_res, os.path.join(OUTD, 'submit_C_thr045'), 0.45,
          os.path.join(OUTD, 'submit_mal2_thr45.zip'))
log('ALL DONE →', OUTD)

try:
    from scripts.viz.import_submission_to_workstation import auto_import_submission
    auto_import_submission(
        os.path.join(OUTD, 'submit_mal2_B_thr050.zip'),
        model_id='mal2',
        name='自研 MAL2 (纯 RGB 55.5 / 56.14%)',
        score='56.14% mAP',
        badge='自研纯可见光',
        color='#38bdf8',
        rebuild=False
    )
    auto_import_submission(
        os.path.join(OUTD, 'submit_mal2_thr45.zip'),
        model_id='mal2_thr45',
        name='自研 MAL2 (τ=0.45 门控)',
        score='56.14% mAP',
        badge='自研纯 RGB',
        color='#38bdf8',
        rebuild=True
    )
    log('Auto-imported mal2 submissions to workstation.')
except Exception as e:
    log('Workstation auto-import notice:', e)
