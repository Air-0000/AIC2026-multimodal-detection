import os, sys, glob
ROOT = "/home/constant/CodeBuddy/Dev/AIC"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "DEIMv2"))

# 统一后的路径 (与 dcnmm.yml 一致)
CACHE = f"{ROOT}/work_dir/cache/train"
IMG = f"{ROOT}/work_dir/deimv2_dataset/train/images"
IR = f"{ROOT}/work_dir/deleak/images"
RAWIR = f"{ROOT}/work_dir/deleak/images"   # raw_ir_folder 重指到 deleak (ir_in_chans=1 不读)
DEPTH = f"{ROOT}/dataset/train/depth"
ANN = f"{ROOT}/work_dir/deimv2_dataset/train/annotations/instances_train.json"
VAL_IMG = f"{ROOT}/work_dir/deimv2_dataset/valid/images"
VAL_IR = f"{ROOT}/work_dir/deleak/valid"
VAL_ANN = f"{ROOT}/work_dir/deimv2_dataset/valid/annotations/instances_val.json"

print("=== 路径存在性 ===")
for k, p in [("cache", CACHE), ("img", IMG), ("ir(deleak)", IR), ("depth", DEPTH), ("ann", ANN),
            ("val_img", VAL_IMG), ("val_ir", VAL_IR), ("val_ann", VAL_ANN)]:
    ok = os.path.exists(p)
    print(f"  {'OK ' if ok else '!! '} {k:12s} {p}")
n_pt = len(glob.glob(CACHE + "/*.pt"))
print(f"  cache .pt 数量: {n_pt}")

from engine.data.dataset.coco_dataset_mm import CocoDetectionMultiModalCached, CocoDetectionMultiModal

print("\n=== train (Cached) 读取一个 6ch 样本 ===")
ds = CocoDetectionMultiModalCached(img_folder=IMG, ann_file=ANN, cache_img_folder=CACHE,
                                    ir_folder=IR, raw_ir_folder=RAWIR, depth_folder=DEPTH,
                                    n_channels=6)
x, t = ds[0]
print("  train sample x.shape:", tuple(x.shape), " dtype:", x.dtype)

print("=== val (MultiModal 非缓存) 读取一个 6ch 样本 ===")
vd = CocoDetectionMultiModal(img_folder=VAL_IMG, ann_file=VAL_ANN, ir_folder=VAL_IR,
                             raw_ir_folder=VAL_IR, depth_folder=DEPTH, n_channels=6)
vx, vt = vd[0]
print("  val sample x.shape:", tuple(vx.shape), " dtype:", vx.dtype)
print("\nALL OK -> 统一路径后数据管线可加载")
