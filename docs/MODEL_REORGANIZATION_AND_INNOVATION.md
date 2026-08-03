# GeoSem-STDA 模型整理与创新方案

本文档用于给后续 AI、同学或消融实验协作者快速理解当前 GeoSem-STDA 模型，并基于已有代码提出下一阶段可落地的创新模型方向。

## 1. 当前研究状态

当前项目已经在 SGDA 框架上实现了一个新的 DEAP 跨被试模型：

```text
GeoSem-STDA sparse reliability
```

当前任务：

```text
DEAP valence 二分类
5 个固定目标被试 + 31 个候选源被试 + 完整数据 + 200 epoch 完整训练协议
```

当前关键代码：

- `models/geosem_stda.py`
- `experiments/deap/crossSubject_geosem_stda_deap.py`
- `experiments/deap/run_geosem_stda_deap_5target_full.ps1`

当前实验结果：

```text
Accuracy = 62.00 +/- 6.75
Macro-F1 = 61.29 +/- 7.59
Micro-F1 = 62.00 +/- 6.75
```

同 5 个目标被试上，原始 SGDA 为：

```text
Accuracy = 67.50 +/- 9.64
Macro-F1 = 65.34 +/- 10.18
```

因此当前模型结论是：

```text
模型可行，但尚未优于原始 SGDA。
它已经证明可以降低多源分支计算开销，但仍存在后期性能退化、源选择区分度不足和可能的负迁移。
```

## 2. 当前模型结构

当前 GeoSem-STDA 可以整理为四条主线。

### 2.1 几何空间主线

输入 DEAP 特征：

```text
X: [B, L, C, F]
L = 9
C = 32
F = 5
```

几何处理流程：

```text
DE feature
-> shrinkage covariance
-> SPD matrix
-> matrix logarithm
-> Log-Euclidean reference
-> tangent deviation R
-> geometric adjacency A_geo
```

对应代码：

- `shrinkage_covariance`
- `_matrix_log_spd`
- `log_euclidean_reference`
- `tangent_deviation`
- `geometric_adjacency`

作用：

```text
让模型显式利用 EEG 通道间协方差结构，而不是只用普通时间序列特征。
```

### 2.2 动态图时空编码主线

模型将几何图和可学习图结合：

```text
A = alpha * A_geo + (1 - alpha) * A_learn
```

其中：

- `A_geo` 来自 SPD / Log-Euclidean 几何结构。
- `A_learn` 来自节点特征的 attention adjacency。
- `alpha` 由几何 gate 自适应控制。

对应模块：

- `DynamicGraphConv`
- `GeometryGate`
- `AttentionAdjacency`
- `TemporalEncoder`
- `CrossAttentionBlock`
- `GeoSemEncoder`

作用：

```text
同时建模 EEG 的通道图结构、时间窗口结构和频带特征。
```

### 2.3 语义空间主线

当前语义空间仍然存在。

流程：

```text
EEG spatio-temporal feature h
-> source adapter
-> prototype head
-> normalized semantic embedding z
-> compare with CLIP text prototypes
```

对应模块：

- `BottleneckAdapter`
- `PrototypeClassifier`
- `prototype_contrastive_loss`

作用：

```text
将 EEG 特征映射到情绪语义原型空间，增强类别可解释性。
```

### 2.4 多源迁移主线

当前仍然保留多源分支，但不再让最终训练完整使用 31 个源。

当前流程：

```text
31 candidate source subjects
-> warm-up model
-> target-aware sparse reliability source selection
-> select 6 sources
-> final training with 6 source adapters
```

源域可靠性评分包括：

```text
score = - d_marg - d_cond + 0.2 * source_acc_proxy
```

其中：

- `d_marg`: 源域和目标域语义嵌入的边缘分布距离。
- `d_cond`: 源域类别中心和目标伪类别中心的条件分布距离。
- `source_acc_proxy`: 源域样本在文本原型上的分类代理准确率。

当前问题：

```text
source_acc_proxy 在实验中基本饱和为 1.0。
选中的 6 个源域权重接近 0.16-0.17，区分度不足。
```

## 3. 当前主要问题

### 3.1 后期性能退化明显

当前 5 个目标被试的 best epoch 和 epoch 200 差距如下：

| Target | Best Epoch | Best Acc | Epoch 200 Acc | Gap |
|---:|---:|---:|---:|---:|
| S3 | 24 | 66.67 | 57.50 | -9.17 |
| S14 | 16 | 71.11 | 65.56 | -5.56 |
| S20 | 23 | 57.50 | 47.50 | -10.00 |
| S23 | 134 | 54.72 | 50.00 | -4.72 |
| S32 | 107 | 60.00 | 55.00 | -5.00 |

说明：

```text
模型在前期已经能学到有效迁移特征，但后期训练持续进行时，目标域表现下降。
```

可能原因：

1. 模型容量偏大，源域拟合过快。
2. MMD 后期持续增强，可能把目标域拉向错误源分布。
3. 源域筛选只 warm-up 5 epoch，可靠性估计可能过早。
4. 语义原型 CE 很快饱和，后期主要由 MMD 和源域过拟合驱动。

### 3.2 可靠性筛源有效降计算，但可靠性不够尖锐

当前每个目标都从 31 个候选源筛到 6 个源。

这是当前模型最明确的优势：

```text
最终多源分支计算从 31 源降低到 6 源，约减少 80.6% 的源分支开销。
```

但问题是：

```text
权重接近均匀，说明 sparse reliability 更像是 top-6 截断，而不是强可靠性加权。
```

因此后续创新不能只写“可靠性筛源”，还要增强可靠性在训练过程中的作用。

### 3.3 当前模型比原始 SGDA 更复杂，但 Acc 没有提升

当前模型新增了：

- SPD 几何。
- 动态图。
- 时空编码。
- 多源 adapter。
- sparse reliability。

但 Acc 低于原始 SGDA。

这说明下一步不适合继续盲目堆模块，而应当：

```text
先增强稳定性和正则化，再引入轻量创新模块。
```

## 4. 当前不建议立刻修改的部分

### 4.1 不改 target-best 汇总协议

虽然严格无监督域适应中不应使用 target label 选择 best epoch，但当前实验要和学长原脚本保持一致。

因此当前主流程仍保留：

```text
summary result = best target epoch result
epoch_log.csv = 记录每个 epoch 的 acc / macro_f1 / micro_f1
```

更严格的 source-validation early stopping 可以作为后续独立消融，但不要替换当前主协议。

在当前项目中，这一点作为硬约束：

```text
不修改学长协议。
不把 target-best 改为 final epoch。
不把 source-validation early stopping 作为主流程。
```

因此，后续所有创新模块只能改：

```text
模型结构
训练损失
源域可靠性权重
数据增强/正则化
MMD 调度方式
```

不能改：

```text
结果汇总协议
目标被试标签使用方式
5-target + 31 candidate source 的完整实验设定
```

### 4.2 不建议第一步加入 EOG/EMG 污染

当前输入是 DE 特征，不是原始 EEG。

因此不建议直接在 DE 特征上加入所谓 EOG/EMG 噪声，因为生理意义不清晰。

更合适的第一步是：

```text
time masking
channel dropout
frequency masking
structured EEG CutMix
```

EOG/EMG 更适合作为第二阶段鲁棒性实验，前提是能从原始 EEG 重新提取 DE 和 SPD。

## 5. 推荐创新模型：RSG-CutMix

建议下一阶段创新模块命名为：

```text
RSG-CutMix
Reliability-aware Semantic-Geometric EEG CutMix
```

中文名称：

```text
可靠性感知的语义-几何结构化 EEG CutMix
```

它不是替代 GeoSem-STDA 主模型，而是作为 GeoSem-STDA 的正则化与鲁棒性增强模块。

注意：不能把“EEG-CutMix 本身”写成本文创新。已有工作已经研究过 EEG-Mixup、EEG-CutMix 和情绪子空间增强。当前可写的创新点是：

```text
在几何引导语义时空域适应框架中，
将目标感知源域可靠性、Log-Euclidean 几何邻近性、图连通脑区 mask 和语义原型一致性
共同用于约束 EEG 结构化混合增强。
```

也就是说，创新不是 CutMix，而是：

```text
Reliability-aware + Semantic-consistent + Geometry-constrained + Graph-structured
```

这四个约束与当前模型共同构成新模块。

## 6. RSG-CutMix 的核心思想

普通 EEG CutMix 的问题：

```text
随机混合 EEG 片段可能破坏情绪语义、脑区结构和跨被试迁移关系。
```

RSG-CutMix 的改进：

```text
只在高可靠源域、同类别样本、几何距离较近、语义一致的 EEG 样本之间进行结构化混合。
```

配对概率：

```text
P(i, j) proportional to
w_si * w_sj * exp(-d_geo(i, j) / tau_g) * 1[y_i = y_j]
```

其中：

- `w_si`, `w_sj`: 两个样本所属源域的可靠性权重。
- `d_geo`: 两个样本 SPD / Log-Euclidean 几何距离。
- `1[y_i = y_j]`: 只允许同类别混合。

## 7. RSG-CutMix 的结构化 mask

输入：

```text
X in R^{L x C x F}
```

不使用随机独立 mask，而使用 EEG 结构化 mask：

```text
连续时间窗口
x
图连通电极子集
x
相邻频带
```

混合形式：

```text
X_mix = M * X_i + (1 - M) * X_j
```

第一版建议：

- 时间维：随机连续 2-4 个窗口。
- 通道维：根据 `A_geo` 或固定脑区选择连通子图。
- 频带维：选择 1 个频带或 2 个相邻频带。
- 标签：只做同类别混合，所以 `y_mix = y_i = y_j`。

## 8. RSG-CutMix 与当前模型如何结合

当前训练流程：

```text
source batch
target batch
-> GeoSem-STDA
-> L_proto + lambda_mmd * L_mmd
```

加入 RSG-CutMix 后：

```text
selected reliable source batch
-> build structured EEG CutMix samples
-> recompute covariance and tangent deviation
-> feed X_mix, R_mix into GeoSem-STDA
-> add prototype loss for mixed samples
```

新增损失第一版只加：

```text
L_aug_proto
```

总损失：

```text
L = L_proto + lambda_mmd * L_mmd + lambda_aug * L_aug_proto
```

暂时不要第一版就加入过多一致性损失，避免继续增加不稳定因素。

## 9. 为什么这个创新适合当前模型

RSG-CutMix 和当前 GeoSem-STDA 的结合点很自然：

| 当前模块 | RSG-CutMix 如何利用 |
|---|---|
| sparse reliability | 决定哪些源域更适合参与增强 |
| SPD / Log-Euclidean geometry | 限制几何距离，避免乱混合 |
| dynamic graph | 选择图连通电极子集 |
| semantic prototypes | 保证混合样本仍靠近同类情绪原型 |
| multi-source adapters | 只对入选可靠源域做增强，避免增加 31 源计算 |

因此它不是孤立的数据增强，而是：

```text
源可靠性 + 几何结构 + 情绪语义一致性
```

共同约束的增强模块。

## 9.1 与已有工作的原创性边界

为了避免“换名抄袭”，论文和代码说明中需要明确区分已有工作和本文候选创新。

| 已有方向 | 已有工作关注点 | 当前模型不能声称 | 当前模型可以强调 |
|---|---|---|---|
| EEG-Mixup | 通过 EEG 样本混合增加训练样本 | 不能声称首次提出 EEG Mixup | 同类别、可靠源域、几何邻近的受限混合 |
| EEG-CutMix | 构造 EEG 混合域或通道级 CutMix | 不能声称首次提出 EEG CutMix | 结合 Log-Euclidean SPD 几何和动态图连通脑区 |
| 情绪子空间增强 | 约束生成样本处在情绪子空间 | 不能声称首次提出情绪子空间约束 | 使用文本语义原型进行轻量一致性约束 |
| 多源域适应 | 多源对齐和源域选择 | 不能声称首次做多源 EEG DA | 将源域可靠性用于源筛选、loss 加权和增强配对 |

当前更稳妥的创新表述：

```text
不同于已有随机或相似度驱动的 EEG 混合增强，本文在 GeoSem-STDA 框架内提出可靠性感知的语义-几何结构化混合正则化。该模块利用目标感知源域可靠性筛选混合候选，利用 Log-Euclidean 几何距离限制跨样本混合范围，利用动态图邻接构造连通脑区 mask，并通过文本语义原型保持情绪类别一致性。
```

## 10. 第一版实现建议

优先实现轻量版本，避免大改主模型：

```text
Version RSG-CutMix v1
```

功能：

1. 只对 selected source loaders 启用。
2. 只混合同类别 source samples。
3. 只在每个 source batch 内或可靠源之间混合。
4. 使用简单结构化 mask：
   - 连续时间块。
   - 随机 channel block 或基于 A_geo 的 top-k 邻域。
   - 相邻频带。
5. 对混合样本重新计算 `R_mix`。
6. 只加入 `L_aug_proto`。

新增参数建议：

```text
--use_rsg_cutmix
--rsg_prob 0.3
--rsg_lambda_aug 0.1
--rsg_time_min 2
--rsg_time_max 4
--rsg_band_width 1
--rsg_channel_ratio 0.25
--rsg_same_source_only
```

第一轮默认：

```text
use_rsg_cutmix = false
```

这样不会破坏当前 baseline；只有显式打开时才运行创新模块。

## 11. 消融实验顺序

建议按以下顺序做，不要一次性把所有创新都加进去。

### A0 当前模型

```text
GeoSem-STDA sparse reliability
```

目的：

```text
保留当前可行性基线。
```

### A1 源选择敏感性

比较：

```text
source_selection = none
source_selection = fixed_top_m
source_selection = sparse_reliability
```

目的：

```text
判断当前性能瓶颈是否来自源选择。
```

### A2 可靠性权重尖锐度

比较：

```text
source_weight_tau = 0.1, 0.2, 0.5
sparse_k_max = 4, 6, 8
sparse_rho = 0.70, 0.80, 0.85
```

目的：

```text
让可靠性权重不再接近均匀。
```

### A3 MMD 调度

比较：

```text
lambda_max = 0.0, 0.1, 0.2, 0.3
```

后续可加入：

```text
warmup-increase
warmup-hold
warmup-decay
```

目的：

```text
判断后期退化是否由持续 MMD 对齐导致。
```

该消融不改变学长协议，只改变训练过程中的 `lambda_mmd` 取值或曲线。结果仍按当前脚本 best-target 汇总。

### A4 简单结构扰动

比较：

```text
time masking
channel dropout
frequency masking
```

目的：

```text
判断低成本正则化是否能缓解后期退化。
```

### A5 RSG-CutMix v1

加入：

```text
same-class structured source CutMix
L_aug_proto
```

目的：

```text
验证可靠性-几何-语义约束是否比普通扰动更有效。
```

## 11.1 评价目标

后续实验不能只追求“模型更复杂”，而要同时满足三个目标：

```text
1. EEG 情绪识别有效：Acc 和 Macro-F1 至少接近或超过原始 SGDA。
2. 计算开销可控：最终源分支数量保持远小于 31，优先保持 4-8 个源。
3. 创新不抄袭：明确引用已有 EEG-Mixup / EEG-CutMix / 情绪子空间增强工作，只强调本文组合约束和与 GeoSem-STDA 的耦合设计。
```

## 12. 当前最优先代码方向

优先级排序：

```text
P0: 保留当前 baseline，不改 target-best 协议
P1: 调 source_weight_tau / sparse_k_max / sparse_rho，增强可靠性区分
P2: 加入 MMD 调度选项，诊断后期退化
P3: 加入轻量结构扰动
P4: 实现 RSG-CutMix v1
P5: 后续再做 EOG/EMG 鲁棒性实验
```

如果只能做一个创新模块，建议先做：

```text
RSG-CutMix v1
```

原因：

1. 和当前可靠性源选择天然相关。
2. 和 SPD 几何图天然相关。
3. 和语义原型天然相关。
4. 不需要取消当前多源结构。
5. 只在最终 6 个源上增强，不会大幅增加 31 源计算开销。

## 13. 给后续 AI 的关键约束

后续 AI 修改代码时必须注意：

1. 当前项目主要输入维度为 `[B, 9, 32, 5]`。
2. `st_dim=128` 必须能被 `heads=4` 整除。
3. 当前 summary 仍使用 best target epoch，不能擅自改成 final epoch。
4. `target_subject_ids` 只限制目标被试，不能裁剪源域池。
5. 对 5-target 完整实验，源候选数量应为 31。
6. 新增增强模块默认应关闭，避免破坏当前 baseline。
7. 如果混合 `X`，必须重新计算对应的 SPD / tangent deviation `R`。
8. 不要直接在 DE 特征上声称加入真实 EOG/EMG 污染。

## 14. 推荐论文表述

可以暂时这样描述创新点：

```text
To reduce the computational burden and mitigate unreliable source transfer in multi-source EEG domain adaptation, we propose a target-aware sparse source selection mechanism and further design a reliability-aware semantic-geometric structured EEG CutMix regularizer. Unlike conventional random EEG augmentation, the proposed regularizer constrains mixed samples by source reliability, class consistency, Log-Euclidean geometric proximity, and semantic prototype consistency. This makes the augmented samples compatible with the geometry-guided semantic spatio-temporal adaptation framework.
```

中文表达：

```text
为降低多源 EEG 域适应的计算开销并缓解不可靠源域带来的负迁移，本文在目标感知稀疏源域选择的基础上，进一步设计可靠性感知的语义-几何结构化 EEG CutMix 正则化模块。不同于随机 EEG 增强，该模块同时约束源域可靠性、类别一致性、Log-Euclidean 几何邻近性和语义原型一致性，使增强样本与几何引导的语义时空域适应框架保持一致。
```

## 15. 参考边界

后续写论文或报告时应引用已有方向，并避免把已有方法包装为本文原创。

可参考的已有工作类型：

- EEG-Mixup / EEGMatch: EEG emotion recognition 中已有基于 Mixup 的增强与半监督域适应。
- EEG-CutMix / progressive multi-domain adaptation: 已有工作将 EEG-CutMix 用于源域和目标域之间的混合过渡。
- Emotional subspace constrained augmentation: 已有工作将情绪子空间约束用于 EEG 情绪识别数据增强。
- Multi-source EEG domain adaptation: 已有工作关注多源域对齐、公共分支或样本混合。

本文候选创新应严格限定为：

```text
GeoSem-STDA 框架下的可靠源域筛选、语义原型、Log-Euclidean 几何和图连通结构化混合的联合设计。
```
