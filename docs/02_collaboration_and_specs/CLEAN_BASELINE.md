# AIC 纯净基准 v1：通用正确性修复与未确定改进项

截至2026-10-21。用户最新顺序为：先修好通用缺陷并固定纯净基准，再统一列改进项；此前“优先stem单项实验”等排序已被本次修复任务取代。

**状态：已于 2026-10-22 在 ailab40 GPU 1 正式启动 50 轮完整训练 (PID 18924)；执行纯净单调 Clean Lifecycle，日志落点远端 `logs/aic_clean_baseline_v1_train.log`。** 不能把旧V34–V36权重或成绩改名成这份已修正基准的训练结果。

**2026-10-23 更新：v1 在第 6 轮第 165 步因 `loss_co_cls` 非有限值被严格守卫停止（第 0–5 轮验证 mAP 至 0.321）。诊断结论为 ailab40 GPU1 显存硬件错误（ECC 关闭、6 页双比特错误退役记录；同一份数据反复读取校验和 84/660 次不一致，健康 GPU 对照 0 次），不是模型数值缺陷。** 详见 MODEL_RECORDS §四十。在硬件修复或换卡前不应在该卡上继续训练。

**2026-10-23/24 更新：v1 迁到 seeta RTX PRO 6000 Blackwell 从头重训（BF16 autocast + TF32，见下方“精度”），截至 10-24 完成 46/50 轮，验证集最佳为 epoch 37 的 mAP 55.94。** 详见 MODEL_RECORDS §四十一–四十三；该数字的口径与限制见 §四十四。

**比赛约束：逐图独立检测是必须遵守的要求。没有时间记忆不列为结构缺陷；所有改进和诊断保持单图推理，不引入跨图状态。**

## 唯一基准入口

`vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml` 为完全展开配置，无传递include。工程仍使用现有DEIMv2，保持V36家族HGNetv2 B5、RGB/双IR/Depth六通道、CSSA/DCDA/DOG/3D-Lift、P2-v10、DINOv3教师与Co-ATSS。

这里“纯净”指修复实现与实验流程正确性、明确来源和关闭未验证新增分支，不表示这些既有部件各自都被因果消融证明有效，也不等于官方纯RGB模型。

固定：12类；1440×2560；seed42；普通aux BN更新（freeze_bn=false，RGB自身freeze_norm保持原骨架）；有效batch为2（accum_steps只拆micro，不乘成4或8）；50轮flatcosine（flat30/no_aug4），最后4轮停横翻。epoch0–49单调推进，final/last与best分别保存。

精度（2026-10-23 commit `c8657a6` 起，随GPU架构变化）：三臂配置均为`use_amp: true`、`accum_steps: 1`、训练`num_workers: 8`+`persistent_workers`。SM≥8.0（Ampere/Ada/Hopper/Blackwell）上`train.py`打开TF32，训练前向走BF16 autocast（criterion在FP32下计算，GradScaler仍启用），验证不走autocast；SM<8.0（V100）上TF32与cuDNN关闭，`train.py`与`clean_lifecycle.py`拒绝`use_amp: true`，须加`-u use_amp=False`按纯FP32运行。因此同一配置在V100与Blackwell上的数值设置不同，ailab40 中断的那次（FP32，`accum_steps: 2`，`num_workers: 0`）与 seeta 重训（BF16+TF32，`accum_steps: 1`，`num_workers: 8`）不是同一设置。同一组对照必须在同一架构上跑。修改前的FP32定义：`git show c8657a6^:vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml`。

查询选择明确保留历史P3配额35%+全局补齐；Fallback关闭，Routing/P4配额改动关闭，Aux分类减权关闭，SDQS/distinct开关False，V37平移/抗混叠/空间偏置/offset限制/层间融合/SECL全部不纳入。类内NMS0.60、标准COCO评测；不启用测试TTA/SAHI或内容融合。

## 已修复并验证的通用问题

| 项 | 当前行为 | 证据 |
|---|---|---|
| YAML加载共享默认dict | 每次load_config独立，避免跨模型串配置 | 独立加载对抗测试 |
| CLI seed0覆盖YAML | 未传--seed时尊重YAML；显式CLI仍最高优先级 | 原生入口mock测试确认seed42 |
| IR stem复制过早 | COCO加载RGB后、构建优化器/EMA前复制；拒绝覆盖已有多模态stem | 真实COCO、64滤波器逐元素契约；远端整模型相同 |
| 80类与12类混用 | 根配置/decoder/postprocessor均12；COCO类别语义不匹配的分类头显式保留新初始化，不冒充Obj365→COCO映射 | shape/mapping测试及1582键严格重载 |
| 教师状态与来源 | 父model.train不改变教师eval；所需精确路径缺失或SHA不符立即拒绝 | eval冻结测试、错误哈希拒绝测试；官方教师真实strict加载 |
| Co-ATSS监督冲突 | 同一anchor分类和回归使用相同最高IoU GT owner；损失仅由criterion统一乘权 | 双GT异类别争同anchor对抗fixture |
| NaN/Inf隐式修复/跳步 | 基准严格模式对输入、嵌套输出、总loss、梯度、参数/buffer及验证输出检查，异常即抛错，不清零后继续 | 非有限值/嵌套输出/预期anchor+inf对抗测试 |
| micro分批权重与LR | 不等长micro按样本数加权；每次更新前设置绝对iteration LR；不构建第二个warmup/scheduler | 不更新参数的标量梯度fixture；独立clean lifecycle |
| optimizer状态没恢复 | 不再对optimizer/scheduler错误传strict参数；模型/EMA仍严格加载 | RNG与optimizer状态恢复测试 |
| 恢复加载吞错 | 基准strict_resume缺model/EMA/optimizer/RNG等必需状态或shape不符则失败；记录有效配置和seed | 不完整/错shape恢复拒绝测试 |
| fresh随机流 | 模型/权重/EMA/loader构建后重置运行RNG；resume恢复Python/NumPy/Torch/CUDA RNG | RNG精确恢复fixture |
| 末轮保存/中途回滚 | clean生命周期不从best回滚，不重置epoch或optimizer；验证后原子保存last、配置终点final、独立best，并绑定当轮metrics | 4轮mock生命周期：epoch连续，best与final不同仍正确 |
| 评测二次tuning | _setup已加载tuning，eval不再重复加载模型、使EMA与model脱节 | 修正实际入口路径；独立推理重载验收 |
| 输入处理依赖静默退化 | strict_inputs缺通道/错误raw16约定/NUC依赖异常/ABLATE覆盖立即拒绝，读取depth不convert(L) | 完整8000文件内容审计及两种格式实际loader前向 |

此前已修的P3捕获顺序、SDQS未实现开关fail-closed、P2输入层数守卫和DCDA缓存清理继续保留。训练异常严格模式是基准显式选项；旧配置行为及远端历史运行副本没有被覆盖。

开跑必须使用固定单卡环境；输入和源码若改变必须重做接受检查，不应把一次功能验收当成跨环境绝对可复现承诺。

## 验收证据与边界

- CPU功能测试14/14通过，无真实数据训练、无实际参数更新；包含真实COCO stem加载。日志 `logs/clean_baseline_contract_20261021.log`。
- 远端ailab40 GPU1，独立目录 `/home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021`。原AIC_V29和AIC_V36_INJ历史源码/数据/权重均未改。
- 1800 train+200 val，四路共8000文件，内容SHA清单齐全。最终检查重算全部SHA，复用第一次完整解码的格式元数据，不复用未校验内容。
- 深度1851张I;16、149张RGB JPEG，原输入保留。确定的是读取分母/输入来源，JPEG是否有统一公制标定仍未证实；本次不擅自替换深度或生成新标签。
- COCO起点SHA `39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b`；DINOv3教师SHA `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`，均固定在配置中并实际校验。
- 两张验证图（RGB JPEG深度image1、16位深度image17），[1,6,1440,2560]输入，1582个推理键从磁盘strict重载；原模型和独立重载模型的pred_logits/pred_boxes最大绝对差均0。此为功能回归，不是完整训练/泛化评测。
- `initial_inference_state.pth` 是COCO初始化后的未训练推理fixture，不能用于恢复完整训练或称为已训练baseline。
- 本地证据根：`work_dir/audits/aic_clean_baseline_v1_20261021/`，最终源包 `source_final.tar.gz`、逐文件清单 `source_final_manifest.json`、结果 `acceptance_final/report.json`、完整输入 `acceptance_final/inputs.json`。
- 早期接受attempt1保留远端，最终以acceptance_final为准；修复前本地源码快照 `archive/backups/clean_baseline_before_20261021/`。没有提交git或覆盖其他既有WIP。

## 使用与重放

本地只做无参数更新的功能测试：

```bash
CUDA_VISIBLE_DEVICES='' /home/constant/miniconda3/envs/deimv2/bin/python scripts/diagnostics/test_clean_baseline_contract.py
```

远端完整只读验收（输出目录必须不存在，先确认GPU空闲）：

```bash
cd /home/ailab40/hejianzhong/AIC_CLEAN_BASELINE_20261021
CUDA_VISIBLE_DEVICES=1 /home/ailab/ssd/bigdata/hejianzhong/_infra/envs/aic2026/bin/python scripts/diagnostics/audit_clean_baseline.py --framework vendor/DEIMv2 --config vendor/DEIMv2/configs/deimv2/aic_clean_baseline_v1.yml --out work_dir/acceptance_replay --gpu 1
```

将来明确启动训练时，cd vendor/DEIMv2，使用该完整配置运行train.py（config自身已绑定COCO tuning、seed和正确性开关）；新实验必须用全新output_dir。恢复完整基准checkpoint应显式-r并用-u tuning=null，保持配置、seed、总预算与输入一致。当前没有启动该命令。

## 未确定收益的改进项（统一清单，当前均不进入基准）

| 项目 | 已知证据/尚欠什么 |
|---|---|
| DOG平滑深度回退 | 零样本与2轮pilot有指标取舍，欠修正基准上的完整同预算对照 |
| P3/P4查询配额、纯全局TopK | 历史Control已经保P3 35%；Routing pilot mAP与AP75有取舍，欠完整训练 |
| TADL辅助分类减权 | 推理前等价、梯度方向有变化；pilot有AP_small/AP75代价，欠完整训练 |
| backbone注入位置[1,2] vs [2,3] | 既有两seed短跑未过small-tight预设；不能用平均AP75替代，也不直接扩大训练 |
| 深度语义/校准/预处理统一 | JPEG公制语义、缺深度时DCDA和3D-Lift含义、depth_p0及天空/质量mask需各自审计和独立对照 |
| P2引导、教师、Co-ATSS的单项贡献 | 作为现有骨架保留，尚不能从V34/V35/V36跨版本成绩断言各自收益；需要同起点移除/保留对照 |
| light/细长目标分类稳定性 | 两帧已定位到encoder差异及后段decoder放大，尚未证明query排斥因果；不全局降阈值掩盖问题 |
| V37微平移、真正采样前抗混叠、空间偏置、offset约束、score-IoU/配对一致性 | 当前组合有实现缺口：零ReLU偏置无梯度、SECL并非平移配对、模糊在下采样后；实验代码先单独修正和验收，再按单项对照，不能直接塞回基准 |
| 模态dropout、质量门控、背景硬负样本等候选 | 当前无统一完整受控收益；需要先分类FP/定位/重复/错分类来源 |

stem复制时机、类别配置、seed覆盖等已确认正确性缺陷已经纳入本轮基准修复，不再包装为“待猜测的新机制”；修复后的成绩仍必须由将来的独立训练取得。

A02 Control42历史训练日志与重载分差仍未解释。这是来源/复现欠账，不是已经验证的新改进，不会被本轮新基准掩盖。


## 单图稳定性筛查（2026-10-21）

已用历史V36权重完成12臂×200验证图诊断，未训练v1。发现1px平移可使几何框保持但GT类置信度崩落；固定选点仅部分解释，直接提高P3下限/减弱交互/减弱融合均未提供全局改进证据。优先定位采样和平移导致的分类logit放大，再决定是否试采样前抗混叠或训练期一致性；两项尚无收益结论。单一实验真相源见`docs/MODEL_RECORDS.md`同日“12臂结构敏感性筛查”，完整证据与重放见`outputs/reports/single_image_stability_20261021/README.md`。


### 微小平移逐层定位补充

同日已完成19个平移丢检内部目标及225个同图稳定目标的三臂逐层追踪，57/57前向与前轮输出逐值一致。观测定位优先指向Decoder2/3的cross-attention/gateway/FFN放大路径，并非所有个例都在同一层；LQE并非主要分差项。下一步先做该路径局部隔离，再决定抗混叠或训练期一致性，不直接改基准或启动训练。详见MODEL_RECORDS同日逐层定位条目与`outputs/reports/layer_translation_trace_20261021/README.md`。


### 局部干预与FFN90候选验收

已执行18臂激活替换及6臂×200图独立单图推理验证。原图FFN×0.90仅有微小AP变化，平移丢检19个、有效框分数跌穿阈值17个均未改善，因此D2/D3衰减候选不纳入基准。不继续扫推理系数；下一项候选收敛为训练期平移配对分类一致性，需要纯净基准同起点Control对照，目前未启动训练。详见MODEL_RECORDS同日FFN90验收及`outputs/reports/translation_ffn_damping_20261021/README.md`。


### 查询谱系与分类评分新线索

19症状目标最终最佳有效候选仍为正确argmax类别，17个所有有效候选类别分数均低于0.48；同token上encoder变化较小而decoder最终分数大跌。全200图softmax评分反证可救回14/19但误检激增/lightAP降低，未采用。冻结TADL源码显示正例target=1、未匹配query为负；NDSL关闭，不能归因于IoU目标上限或额外NDSL抑制。下一线索为原生matcher正例身份/分类代价与去重的耦合，尚无根因或训练收益结论。详见MODEL_RECORDS与`outputs/reports/translation_query_lineage_20261021/README.md`。


### 多项原因测试与解释修正（最新优先）

已完成原生matcher/TADL重放17550条、采样核11臂×19图、IoU边界扫描和FFN/LayerNorm审计/仿射分解。19例中4例有0.45–0.50高分近失配，13例即IoU0.3仍无高分；先前light GT9“纯分类崩落”解释修正为IoU0.488477的定位近失配。token换位稳定组也常见，不能定根因；采样核各分支替换不能普遍恢复。排除eps近零/溢出/残差整体抵消，在Decoder2/3观察到gamma通道重加权放大差异，尚非归一化bug或改法有效证明。最新边界、完整计数与原始证据见MODEL_RECORDS同日条目及`outputs/reports/translation_cause_validation_20261021/README.md`。无新训练或基准改动。

### 多尺度、平移相位与负面筛选结果

冻结V36继续验证：完整200图RGB/RGB+IR stem低通降低原图mAP及共同原图TP稳定性，D2 gamma16减弱10%未救回原19丢检；均不采用。19图相位扫描发现方向相关且非单调，尚无固定步长周期证据。恢复编码器P3/P4/P5分别救回强13中的7/3/5，P3+P4为10；插值对齐ALL为9，支持多尺度共同参与，但反事实激活与插值有局限，不是可部署改法或唯一根因。详细结果、来源与否定项见MODEL_RECORDS同日条目及outputs/reports/translation_cause_followup_20261021/README.md。纯净v1仍未训练、124文件hash一致；完整200图方向测试已完成：右1和下1各丢19个内部原TP，仅7个重合；不可泛化为仅水平故障。负方向与填充对照进一步支持特定采样位置失稳，但未证明固定周期或唯一故障层。

局部恢复诊断补记：目标框+64像素窗恢复仅约2.56%空间位置，强13找回9（相对锁token新增8），225稳定对照无损失；外部补集为3，五组等面积随机为2/0/1/1/3。只支持该症状队列中目标附近多尺度特征的重要性；GT/原图激活不可部署，未证明采样混叠或某层bug，详见MODEL_RECORDS。

### 跨query几何影响已验证，仍不改基准（2026-10-22）

**最新结论（10月22日）：已拿到跨query几何→自注意力→第二层表征分歧→分类低分的受控干预证据；仍未发现可直接纳入基准的有效修复。** 原query参考框不动，只改其他51个query的初始高度，垃圾桶该query从0.68%恢复到91.63%；另一船例在原图不动、只改一个query初始CY时，该query从94.66%降到1.57%，但另一query能补位。必须区分query掉分与整体漏检。第2层门控/归一化/FFN链条的放大发生在该案例，尚非所有案例或某一层代码bug的证明。

原19丢检仍分为13强低分、4定位边界、2混合。完整200图方向、低通、gamma减弱、多尺度、局部/框内空间、五组等面积随机、query/memory/逐层/参考几何/单query双向与注意力传播测试均已完成；简单低通与gamma减弱均未通过，不写入基准。纯净v1仍未训练、124源文件保持冻结，无新训练。

当前未验证改进方向仅保留两类：训练期微平移一致性（可包含初始参考几何、query表征和分数稳健性）；经过训练适配的采样抗混叠。两者均未证实有效，现有固定预滤波的负结果不能改写成训练适配版有效或无效。必须先按统一协议取得纯净v1实测基线，才能评价新训练改进；本轮不启动。

完整因果对照、失败现场、query身份、特征/注意力trace与来源SHA见MODEL_RECORDS末尾及outputs/reports/translation_cause_followup_20261021/README.md。
