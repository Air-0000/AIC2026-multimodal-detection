# 04 — Mr.DETR 训练环境搭建与验证（远端 seeta RTX PRO 6000 Blackwell）

- **日期**：2026-11-01（Asia/Shanghai），env-builder
- **节点**：`ssh -p <port> root@<seetacloud-host>`（AutoDL 容器 `<autodl-container-id>`）
- **GPU**：1× NVIDIA RTX PRO 6000 Blackwell Server Edition，97887 MiB，capability (12,0) = sm_120；208 核 CPU / 1.0 TiB 内存
- **磁盘**：`/` overlay 30 GB（编译环境放这里）；`/autodl-fs`（md0 网络盘）570 GB 可用但 **97% 使用率、I/O 拥塞**；`/root/autodl-tmp` 仅 7.1 GB 可用
- **结论**：**方案 A（base env，Python 3.12 + torch 2.12.1+cu130 + nvcc 13.0）全部成功，未启用方案 B**。`detectron2._C`、`detrex._C`（含 MSDA deformable attention CUDA op）均编译并实测通过；Swin-L 5-scale 900q 模型前向、AIC 12 类微调配置 instantiate + 权重加载 + forward/backward、batch 1/2/4 测速全部 PASS。
- **当前状态**：Lead 已于 17:35:24 用该环境启动正式微调（PID 10603，日志 `/root/aic_run/train_173524.log`），**环境已进入生产使用**；本报告所有数字均为实测。

---

## 1. 环境总览（最终状态）

| 项 | 值 |
|---|---|
| Python 可执行文件 | **`/root/miniconda3/bin/python`**（conda base，Python 3.12.3） |
| torch / torchvision | 2.12.1+cu130 / 0.27.1+cu130 |
| nvcc / toolkit | `/usr/local/cuda-13.0`，V13.0.88（与 torch 的 cu130 一致；`CUDA_HOME` 由 torch 自动解析为 `/usr/local/cuda` → `cuda-13.0`） |
| host gcc / g++ | 11.4.0 |
| 编译目标架构 | sm_120（`TORCH_CUDA_ARCH_LIST=12.0`） |
| detectron2 | **0.6**（源码 pin `9604f5995cc628619f0e4fd913453b4d7d61db3f`），**pip editable 安装**，CUDA 扩展 `_C` 已编译 |
| detrex | **0.3.0**（Mr.DETR 仓库 `main@287c693` 内的 `detrex/`），**pip editable 安装**，CUDA 扩展 `_C` 已编译（MSDA + DCNv3） |
| 源码规范位置 | `/autodl-fs/data/mrdetr/detectron2_src`、`/autodl-fs/data/mrdetr/mrdetr_src` |
| 编译工作区（本地 overlay，速度快） | `/root/build_mrdetr/detectron2_src`、`/root/build_mrdetr/mrdetr_src` |
| 编译产物（最终，两处一致） | `/autodl-fs/data/mrdetr/detectron2_src/detectron2/_C.cpython-312-x86_64-linux-gnu.so`（1.45 MB）<br>`/autodl-fs/data/mrdetr/mrdetr_src/detrex/_C.cpython-312-x86_64-linux-gnu.so`（2.93 MB） |
| 权重 | `/autodl-fs/data/mrdetr/mrdetr_weights/MrDETR_align_swinl_12ep_900q_objects365_processed.pth`（903,827,174 B）、`MrDETR_align_r50_12ep_900q.pth`（221,723,325 B） |
| 激活脚本 | **`/autodl-fs/data/mrdetr/activate_mrdetr.sh`**（可选；不 source 也能 import，见 §7） |
| 验证脚本 | `/autodl-fs/data/mrdetr/env_scripts/{smoke_import,smoke_swinl_forward,smoke_aic_finetune,smoke_r50_ckpt,bench_swinl}.py`、`build_all.sh` |

python 内实际 import 到的路径（最终状态，`source activate_mrdetr.sh` 后）：

```
detectron2  0.6  /autodl-fs/data/mrdetr/detectron2_src/detectron2/__init__.py
detrex           /autodl-fs/data/mrdetr/mrdetr_src/detrex/__init__.py
detectron2._C    /autodl-fs/data/mrdetr/detectron2_src/detectron2/_C.cpython-312-x86_64-linux-gnu.so
detrex._C        /autodl-fs/data/mrdetr/mrdetr_src/detrex/_C.cpython-312-x86_64-linux-gnu.so
from detrex.layers import MultiScaleDeformableAttention  → OK
```

> 说明：`site-packages` 里的 editable 链接（`detectron2.egg-link` / `detrex.egg-link` / `easy-install.pth`）已指向 `/autodl-fs/data/mrdetr/...`；`/root/build_mrdetr/...` 保留为编译缓存兼 fallback（当前正在跑的正式训练进程是 17:35 用后者启动的，两者代码/产物完全一致，互不影响）。

初始已装（未动）：numpy 2.4.6、opencv-python-headless 5.0.0.93、timm 1.0.30、scipy 1.18.1、psutil 7.2.2、tqdm 4.66.2、Pillow 12.2.0、matplotlib 3.11.0、tensorboard 2.20.0、packaging 23.2、setuptools 69.5.1、pip 24.0。

---

## 2. 依赖安装命令（逐条可复现）

阿里云 pip 镜像已配好（`/etc/pip.conf` → `mirrors.aliyun.com`），全程只从该镜像装包：

```bash
# 2.1 detectron2 / detrex 运行时缺失的纯 Python / 小依赖
/root/miniconda3/bin/pip install -q pycocotools yacs tabulate cloudpickle termcolor \
    fvcore iopath "omegaconf>=2.2" hydra-core pybind11 einops

# 2.2 detrex 顶层 import 链额外需要的包（首次 import 时依次暴露）
#     wandb  ← detrex.utils.events 模块级 import wandb
#     fairscale ← detrex.modeling.backbone.eva
#     submitit ← detrex requirements.txt 列出（训练脚本用）
/root/miniconda3/bin/pip install -q wandb submitit fairscale
```

装完的关键版本（`pip list` 原始值）：

```
cloudpickle 3.1.2   einops 0.8.2        fairscale 0.4.13    fvcore 0.1.5.post20221221
hydra-core 1.3.8    iopath 0.1.10       numpy 2.4.6         omegaconf 2.3.1
opencv-python-headless 5.0.0.93         pillow 12.2.0       psutil 7.2.2
pybind11 3.1.0      pycocotools 2.0.11  scipy 1.18.1        submitit 1.5.4
tabulate 0.10.0     termcolor 3.3.0     timm 1.0.30         wandb 0.31.0
yacs 0.1.8          torch 2.12.1+cu130  torchvision 0.27.1+cu130
```

**注意（有意为之，非错误）**：detrex 的 `requirements.txt` 写死了 `scipy==1.7.3`、`omegaconf==2.1.0`（2022 年的 pin，py3.12 无 wheel 且与 hydra 1.3 冲突），因此 detrex 用 **`--no-deps`** 安装并手工装现代版本。`pip check` 会提示 `detrex 0.3.0 requires omegaconf==2.1.0 / scipy==1.7.3`，属预期，运行时用 2.3.1 / 1.18.1 已实测通过全部冒烟与训练。

---

## 3. 编译命令与关键日志

### 3.1 最终采用的编译方式（本地 overlay 盘）

两个扩展都用同一个模式：`pip install -e . --no-deps` + 显式 CUDA 环境变量，`MAX_JOBS=64`，只编 sm_120。

```bash
# detectron2（在本地盘编译工作区）
cd /root/build_mrdetr/detectron2_src
FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=64 TORCH_CUDA_ARCH_LIST=12.0 \
    /root/miniconda3/bin/pip install -e . --no-deps

# detrex（同上；编译 MSDA + DCNv3 自定义 CUDA op）
cd /root/build_mrdetr/mrdetr_src
FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=64 TORCH_CUDA_ARCH_LIST=12.0 \
    /root/miniconda3/bin/pip install -e . --no-deps
```

成功日志（原始）：

```
# /root/build_mrdetr/d2_build.log
  Running setup.py develop for detectron2
Successfully installed detectron2-0.6

# /root/build_mrdetr/detrex_build2.log
  Running setup.py develop for detrex
Successfully installed detrex-0.3.0
```

编译耗时：本地盘上 detectron2（11 个 .o 源文件）约 2.5 分钟，detrex 约 1.5 分钟（208 核 / MAX_JOBS=64）。

### 3.2 首次编译失败与修复（detrex / DCNv3，torch 2.12 API 不兼容）

第一次 `pip install -e .` 编译 detrex 时 **`detrex._C` 编译失败**，`setup.py develop did not run successfully`，原始报错：

```
/root/build_mrdetr/mrdetr_src/detrex/layers/csrc/DCNv3/dcnv3_cuda.cu(69): error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
          [&] { const auto& the_type = input.type(); constexpr const char* at_dispatch_name = "ms_deform_attn_forward_cuda"; torch::headeronly::ScalarType _st = ::detail::scalar_type(the_type); ;
                                                                                                                       ^
/root/build_mrdetr/mrdetr_src/detrex/layers/csrc/DCNv3/dcnv3_cuda.cu(147): error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
        2 errors detected in the compilation of "/root/build_mrdetr/mrdetr_src/detrex/layers/csrc/DCNv3/dcnv3_cuda.cu".
        error: command '/usr/local/cuda-13.0/bin/nvcc' failed with exit code 2
error: subprocess-exited-with-error
    × python setup.py develop did not run successfully.
```

原因：detrex 0.3.0 的 csrc 是 2022 年代码，`AT_DISPATCH_FLOATING_TYPES(x.type(), ...)` 依赖 `at::DeprecatedTypeProperties` 到 `c10::ScalarType` 的隐式转换；**torch ≥ 2.5 移除了该隐式转换**（应传 `x.scalar_type()`）。

修复（共 2 个文件、4 处调用，均为一行替换；已同步回 `/autodl-fs` 规范源码树）：

```diff
# detrex/layers/csrc/DCNv3/dcnv3_cuda.cu   (line 70, 148)
-            input.type(), "ms_deform_attn_forward_cuda", ([&] {
+            input.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
-            input.type(), "ms_deform_attn_backward_cuda", ([&] {
+            input.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {

# detrex/layers/csrc/MsDeformAttn/ms_deform_attn_cuda.cu   (line 65, 135)
-        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda", ([&] {
+        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
-        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda", ([&] {
+        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
```

`AT_ASSERTM(x.type().is_cuda(), ...)` 等**不改**（只是 deprecated 警告，仍能编译）。patch 后重编成功（日志见 §3.1）。**detectron2 的 csrc 在 torch 2.12 / nvcc 13.0 下无需任何改动，一次编译通过。**

---

## 4. 冒烟验证（全部实测 PASS，命令 + 原始输出）

统一使用 `/root/miniconda3/bin/python`，在 `/root` 下执行（不依赖异常 cwd）：

### 4.1 `import detectron2` / `import detrex` / `detectron2._C` / `detrex._C` / MSDA CUDA op

```bash
/root/miniconda3/bin/python /autodl-fs/data/mrdetr/env_scripts/smoke_import.py
```

```
python : /root/miniconda3/bin/python 3.12.3
torch  : 2.12.1+cu130 | cuda: 13.0 | avail: True
gpu    : NVIDIA RTX PRO 6000 Blackwell Server Edition | cap: (12, 0)
detectron2: 0.6 | /root/build_mrdetr/detectron2_src/detectron2/__init__.py
detectron2._C: /root/build_mrdetr/detectron2_src/detectron2/_C.cpython-312-x86_64-linux-gnu.so
detrex: /root/build_mrdetr/mrdetr_src/detrex/__init__.py
detrex._C: /root/build_mrdetr/mrdetr_src/detrex/_C.cpython-312-x86_64-linux-gnu.so
detrex.layers.MultiScaleDeformableAttention import: OK
MSDA CUDA op fwd/bwd: OK | out (2, 100, 32) | loss -27.85967254638672 | grad_norm 269.8994140625
SMOKE_IMPORT_RESULT: PASS
```

> 该次运行早于最终路径切换，故显示 `/root/build_mrdetr`；切换后复核输出（`source /autodl-fs/data/mrdetr/activate_mrdetr.sh`）为 `/autodl-fs/data/mrdetr/...`，MSDA import 同样 OK。MSDA 测试是真实 `MultiScaleDeformableAttnFunction.apply` 前向 + 反向（不是纯 Python fallback），证明自定义 CUDA op 可计算。

### 4.2 官方 Swin-L 5-scale 900q 配置：instantiate + 1 次 CUDA 前向

配置：`projects/mr_detr_align/configs/deformable_detr_swinl_two_stage_12ep_plusplus_5scale_fintuning.py`（**只 instantiate model，不 resume/load**，因为 `train.init_checkpoint` 指向本地不存在的 Objects365 路径）

```bash
/root/miniconda3/bin/python /autodl-fs/data/mrdetr/env_scripts/smoke_swinl_forward.py
```

```
[1] LazyConfig.load OK  (1.1s)
[2] instantiate OK | params total 224.9M (backbone 195.2M) | num_classes=80 num_queries=900 levels=5 two_stage=True mixed_selection=True
[3] eval forward OK | 1.42s at 1x(3,1152,2048) | out boxes=(300, 4) scores=(300,)
[4] peak mem (eval fwd, no grad) 4.14 GiB
SMOKE_SWINL_RESULT: PASS
```

（`[1]` 处用了 `LazyConfig.load` + `from detectron2.config import instantiate`；本 pinned detectron2 0.6 里 `instantiate` 是模块级函数，`LazyConfig.instantiate` 不存在——任务书里的写法要按此修正。）

### 4.3 AIC 微调配置（12 类）：load + instantiate + Swin-L 权重加载 + 1 次 forward/backward

配置：`projects/mr_detr_align/configs/aic_swinl_finetune.py`（继承官方 5-scale 配置，`num_classes=12`，batch 2，12 epoch，注册 `aic_train`/`aic_val`）

```bash
/root/miniconda3/bin/python /autodl-fs/data/mrdetr/env_scripts/smoke_aic_finetune.py
```

```
[1] LazyConfig.load OK (incl. aic_train/aic_val 注册)
[2] instantiate OK | num_classes=12 num_queries=900 levels=5 two_stage=True select_box_nums=300 params=224.7M
    class_embed heads=7, out_features=[12, 12, 12, 12, 12, 12, 12]; label_enc.weight=(12, 256)
[3] checkpoint found: .../MrDETR_align_swinl_12ep_900q_objects365_processed.pth 862.0 MB
    ckpt keys=784 | shape-match=753 | skipped=31 | missing-in-ckpt=0
    SKIP transformer.decoder.class_embed.0.weight      ckpt=(80, 256) model=(12, 256)
    ...（7 个 decoder class_embed 头 × weight/bias = 14 个）
    SKIP transformer.decoder.class_embed_o2m_encoder.weight ckpt=(80, 256) model=(12, 256)
    SKIP transformer.decoder.class_embed_o2m_encoder.bias   ckpt=(80,)    model=(12,)
    SKIP class_embed.0.weight ... class_embed.6.bias      （7 个顶层头 × weight/bias = 14 个）
    SKIP label_enc.weight                              ckpt=(80, 256) model=(12, 256)
[4] DetectionCheckpointer(model).load OK
[5] fwd+bwd OK | 1.59s | loss=91.9248 | nan=False | loss terms=(130 项 loss_class*/loss_bbox*/loss_giou* 及 dn)
[6] peak mem 12.66 GiB
SMOKE_AIC_RESULT: PASS
```

**80→12 类头的 skip 统计（任务要求记录）**：
- checkpoint 共 **784** 个 key，其中 **753 个 shape 完全匹配并加载**，**31 个因形状不符被跳过**（`incorrect_shapes`），**0 个 missing-in-ckpt**（模型 784 个 key 全在 ckpt 里）。
- 31 个被跳过 key = **30 个分类头 key + 1 个 `label_enc.weight`**：
  - `class_embed.{0..6}.{weight,bias}`（顶层，7 头 × 2 = 14）
  - `transformer.decoder.class_embed.{0..6}.{weight,bias}`（7 头 × 2 = 14）
  - `transformer.decoder.class_embed_o2m_encoder.{weight,bias}`（2，Mr.DETR++ 第二路由分类头）
  - `label_enc.weight`（1，去噪 query 的类别 embedding，80→12）
- 结果：backbone / neck / transformer / bbox_embed / 位置编码等 **753 个权重全部加载成功**，只有 12 类分类头（含 o2m 路由头、DN label embedding）随机初始化。这正是微调想要的语义；`DetectionCheckpointer.load` 内部按 shape 自动跳过（fvcore `_IncompatibleKeys.incorrect_shapes`），不抛异常。

### 4.4 官方 R50 权重（COCO 80 类）加载链路验证（Swin-L 权重到位前的独立验证）

配置与权重按 Mr.DETR README 对应关系取：`MrDETR_align_r50_12ep_900q.pth` ↔ `projects/mr_detr_align/configs/deformable_detr_r50_two_stage_12ep_plusplus.py`

```bash
/root/miniconda3/bin/python /autodl-fs/data/mrdetr/env_scripts/smoke_r50_ckpt.py
```

```
[1] LazyConfig.load OK
[2] instantiate OK | num_classes=80 num_queries=900 levels=4 two_stage=True params=54.2M
[3] ckpt top-level keys: ['model']
[4] ckpt keys=688 | shape-match=688 | skipped=0 | missing-in-ckpt=0
[5] DetectionCheckpointer(model).load OK
[6] eval forward OK | 0.76s | out boxes=(300, 4) scores=(300,) max_score=0.0611
[7] peak mem 1.17 GiB
SMOKE_R50_RESULT: PASS
```

---

## 5. 测速（forward+backward，10 iter 计时 + 2 iter warmup）

脚本：`/autodl-fs/data/mrdetr/env_scripts/bench_swinl.py`，配置 = **AIC 微调配置**（Swin-L / 5-scale / 900q / 12 类 / backbone gradient checkpointing 开），与 `custom_train_net_swinl_gradaccumulate_fintuning.py` 一致：**AMP fp16 autocast + GradScaler + AdamW**，输入固定 `(3,1152,2048)` 随机图（数据加载不计入），batch 1/2/4。

```bash
/root/miniconda3/bin/python /autodl-fs/data/mrdetr/env_scripts/bench_swinl.py --batches 1,2,4 --iters 10 --warmup 2
```

```
model instantiate OK | classes=12 queries=900 levels=5 params=224.7M
--- batch=1 warmup=2 iters=10 ---
batch=1  mean=0.889s/it  (min 0.857, max 0.944)  peak_mem=15.24 GiB  loss 92.480->245.849  nan=False
--- batch=2 warmup=2 iters=10 ---
batch=2  mean=1.668s/it  (min 1.615, max 1.736)  peak_mem=26.49 GiB  loss 234.685->235.814  nan=False
--- batch=4 warmup=2 iters=10 ---
batch=4  mean=3.277s/it  (min 3.227, max 3.402)  peak_mem=50.19 GiB  loss 228.503->82.853  nan=False
BENCH_RESULT: PASS
```

单 iter 原始计时（batch=2）：`1.7002 1.7359 1.7215 1.7129 1.6377 1.6782 1.6286 1.6169 1.6310 1.6148` s；batch=4：`3.2741 3.4016 3.2762 3.2412 3.2402 3.2534 3.2676 3.3456 3.2401 3.2269` s。结果 JSON 在 `/root/build_mrdetr/bench_swinl.json`。

### 5.1 固定 (1152×2048) 下的 12 epoch 预估

AIC 微调集 1800 张图（`ITERS_PER_EPOCH = 1800 // batch`，12 epoch；注意实际训练用 batch=2 → 900 iter/epoch → 10800 iter）：

| batch | iters/epoch | 12 epoch iters | s/it（固定 1152×2048） | 峰值显存 | 12 epoch 纯训练预估* |
|---|---|---|---|---|---|
| 1 | 1800 | 21600 | **0.889** | **15.24 GiB** | ≈ 5.33 h |
| 2 | 900 | 10800 | **1.668** | **26.49 GiB** | ≈ 5.00 h |
| 4 | 450 | 5400 | **3.277** | **50.19 GiB** | ≈ 4.92 h |

\* 12 epoch 预估 = iters × s/it，**按最坏情况（固定 1152×2048）估算**；不计 eval/checkpoint/数据加载。

结论：**吞吐在 batch=2~4 基本平台化**（batch1 1.12 img/s、batch2 1.20 img/s、batch4 1.22 img/s），batch=4 只快 ~2% 却多吃 24 GiB 显存，故 **batch=2 是性价比最优**；三档显存均能放下（96 GiB），但 batch=4 的 50 GiB 会挤压 eval/碎片余量。

### 5.2 现网训练交叉验证（只读观察，未干预）

Lead 启动的正式训练日志 `/root/aic_run/train_173524.log`（batch=2，训练分辨率 480–1280 短边 / max 2048，比固定 1152×2048 小很多）：

```
[11/01 17:37:33 d2.utils.events]: eta: 2:51:11  iter: 119  ... time: 0.9804  data_time: 0.0040  max_mem: 29929M
[11/01 17:38:38 d2.utils.events]: eta: 2:57:36  iter: 179  ... time: 0.9955  data_time: 0.0042  max_mem: 37291M
```

即真实训练在随机分辨率下约 **0.98–1.0 s/iter**（data_time ≈ 0.004 s，数据加载完全被掩盖），**10800 iter ≈ 2.9–3.0 h**，与日志 ETA（17:35 启动，预计 ~20:30 完成）一致。峰值显存 29–37 GiB。这说明 §5.1 的固定分辨率数字是**上限/保守值**，Lead 的实跑时长可再打 ~0.6 折。

---

## 6. 踩坑记录（按发生顺序）

1. **`/autodl-fs` (md0) I/O 拥塞导致首次 `pip install -e` 停滞**：第一次直接在 `/autodl-fs/data/mrdetr/detectron2_src` 编译时，11 个 `.o` 全部编译成功，但最后 link 阶段（`ld`）在 D-state（`wait_on_page_bit`）卡住 6+ 分钟不推进；当时 md0 **97% 使用率**（570 GB 可用）、实测写 60 MB/s / 读 83 MB/s（对比本地 overlay `dd` 2.4 GB/s）。同时段另一个会话正在向 md0 写 900 MB 权重，加剧拥塞。ssh 通道在 16.5 分钟后断开（exit 255），`ld` 成孤儿进程。
2. **orphan `ld` 写出的是半成品 `.so`**：救援时若直接 `cp build/lib.../_C.so` 到包目录会出现 **`invalid ELF header`**（Lead 已确认该文件不可用）。因此**不要从被 kill 的 link 进程产物里取 `.so`**；清理半成品后从零重编。
3. **解决方式（最终方案）**：编译工作区搬到本地 overlay 盘 `/root/build_mrdetr/`（`/` 有 30 GB，本地盘 `dd` 2.4 GB/s），编译完成后把 `.so`、patch 后的 csrc、`.egg-info`、config 软链回填到 `/autodl-fs` 规范源码树，并把 site-packages 的 editable 链接指向 `/autodl-fs`。最终 **import 路径 = `/autodl-fs`，编译缓存 = `/root/build_mrdetr`**，两者一致互为 fallback。
4. **detrex csrc 与 torch 2.12 不兼容**（`AT_DISPATCH_FLOATING_TYPES(x.type())` 隐式转换被移除）——报错与 4 处一行修复见 §3.2。
5. **detrex 缺依赖是"逐个 import 暴露"的**：`import detrex` 依次需要 `wandb`（`detrex/utils/events.py` 模块级 import）→ `fairscale`（`detrex/modeling/backbone/eva.py`）→ `submitit`。装 `--no-deps` 时必须手动补齐，否则报 `ModuleNotFoundError`。
6. **detrex `requirements.txt` 的 2022 年 pin**（`scipy==1.7.3`、`omegaconf==2.1.0`、`black==22.3.0` 等）在 py3.12 无法满足：安装用 `--no-deps`，手动装现代版本；`pip check` 的 2 条 incompatibility 警告是预期的，实测训练正常。
7. **detectron2 0.6 的 API 细节**：该 pin 没有 `LazyConfig.instantiate`，应 `from detectron2.config import LazyConfig, instantiate` 后 `instantiate(cfg.model)`（任务书里的 `LazyConfig.instantiate(...)` 写法会 AttributeError）。
8. **无害警告**（已确认不影响正确性，训练日志里也有）：`torch.cuda.amp.autocast/custom_fwd/custom_bwd` deprecation、`torch.utils.checkpoint` 需显式 `use_reentrant`、`torch.meshgrid` 需 `indexing=`、`non-tuple sequence for multidimensional indexing`（criterion 里 `pos_weights[pos_ind]`）。
9. **坏文件清理**：`MrDETR_align_swinL_...`（15 B，大写 L 的坏文件）已删除；`mrdetr_weights/` 现只保留 R50（212 MB）与小写 swinl 真权重（903,827,174 B）两个文件。

---

## 7. 使用方式（复现 / 给训练脚本）

**最简（无需任何 activate 脚本）**——editable 安装已写进 site-packages，任何 cwd、任何 shell 直接：

```bash
/root/miniconda3/bin/python -c "import detectron2, detrex, detectron2._C, detrex._C; from detrex.layers import MultiScaleDeformableAttention; print('OK')"
```

**需要 `projects.*` 可 import / 改编译参数时（推荐训练时 source）**：

```bash
source /autodl-fs/data/mrdetr/activate_mrdetr.sh
# 等价于：
#   export CUDA_HOME=/usr/local/cuda-13.0
#   export PATH=/root/miniconda3/bin:$CUDA_HOME/bin:$PATH
#   export PYTHONPATH=/autodl-fs/data/mrdetr/detectron2_src:/autodl-fs/data/mrdetr/mrdetr_src:/autodl-fs/data/mrdetr/mrdetr_src/projects:$PYTHONPATH
#   export TORCH_CUDA_ARCH_LIST=12.0
```

**训练命令（Lead 已在跑，仅存档）**：

```bash
cd /autodl-fs/data/mrdetr/mrdetr_src
/root/miniconda3/bin/python projects/mr_detr_align/custom_train_net_swinl_gradaccumulate_fintuning.py \
    --config-file projects/mr_detr_align/configs/aic_swinl_finetune.py \
    --num-gpus 1 --grad-accum-steps 8
```

**重编译扩展（若换节点/换 torch）**：直接重跑 §3.1 两条 `pip install -e . --no-deps` 命令（源码已含 patch）。建议把源码 rsync 到本地盘再编译，避免 md0 拥塞。

---

## 8. 验证边界与未做的事

- **未启动任何正式训练**（本报告只做 10 iter 级测速；正式训练由 Lead 于 17:35 启动）。
- **未做 dataloader 端到端验证**（`aic_train`/`aic_val` 只验证了注册成功 + config load；真实数据加载/增强未跑，但 Lead 的正式训练日志显示 data_time ≈ 0.004 s/iter、loss 正常下降，事实上已覆盖）。
- **未测 eval 耗时**（每 epoch 200 张 val，短边 1280/max 2048 + COCO eval；预计每轮数分钟，未纳入 §5 预估）。
- 验证用的 `logits/loss` 是随机数据 + 随机 GT，**只能证明形状/流程/显存/速度正确，不能说明精度**。
- `/autodl-fs` 仍在 97% 使用率，长时间训练/多次 checkpoint 请继续写本地 overlay（如 `/root/aic_run`），符合 Lead 的路径安排。

## 附：本次验证脚本与原始输出落点

| 文件 | 说明 |
|---|---|
| `/autodl-fs/data/mrdetr/env_scripts/smoke_import.py` | §4.1 |
| `/autodl-fs/data/mrdetr/env_scripts/smoke_swinl_forward.py` | §4.2 |
| `/autodl-fs/data/mrdetr/env_scripts/smoke_aic_finetune.py` | §4.3 |
| `/autodl-fs/data/mrdetr/env_scripts/smoke_r50_ckpt.py` | §4.4 |
| `/autodl-fs/data/mrdetr/env_scripts/bench_swinl.py` | §5 |
| `/autodl-fs/data/mrdetr/env_scripts/build_all.sh` | 两扩展编译 + import 自检的批处理脚本 |
| `/root/build_mrdetr/bench_swinl.json` | 测速原始 JSON |
| `/root/build_mrdetr/{d2_build.log,detrex_build.log,detrex_build2.log}` | 编译日志（失败 + 成功） |

本地留档（gitignored）：`archive/scratch/mrdetr_sync/`（脚本源码 + `raw/*.out` 原始输出）。
