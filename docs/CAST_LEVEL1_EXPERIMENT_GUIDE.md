# CAST-EEG Level-1 实验说明

## 公平性边界

- E0–E5 沿用 DREAMER Rapid Pilot-3：目标 S12/S2/S4、每个目标固定 6 个源域、seed=42、CLIP 原型与源域 adapter 不变、对齐损失关闭。
- CAST Level-1 不实例化原 GeoSem 编码器，因此不计算 SPD、Log-Euclidean、ReSGCA、UOT 或条件对齐。
- 均衡小样本只从源域按类别抽取，目标域始终保持全量且只用于评估。抽样索引及 SHA-256 会写入 `balanced_source_manifest.json`。
- 当前 Rapid 协议用目标准确率选择 best epoch。代码没有用目标标签训练，但这种选 epoch 方式不属于严格 held-out 模型选择；结果只能与同协议结果比较。
- 冻结实验是模块机制验证，不代替正式端到端消融。冻结必须先加载真实检查点；没有检查点时程序会拒绝 `--cast_freeze_loaded`。

## 变体与数据流

| 变体 | 新增结构 | 表征流 |
|---|---|---|
| E0 | 无 | R4 Direct DE → Adapter → CLIP |
| E1 | Strong-DE | Direct DE projection → residual MLP → Adapter → CLIP |
| E2 | 动态通道选择 | channel weights → reweighted DE → Strong-DE |
| E3 | 稀疏空间图 | E2 + learned top-k DE graph；`LayerNorm(base + spatial mean)` |
| E4 | 多尺度时序 | E3 spatial tokens → Conv1d 3/5/7 → learned scale weights → one MHSA；`LayerNorm(base + temporal mean)` |
| E5 | 跨注意力与门控 | temporal query × spatial key/value → `LayerNorm(base + beta × context)`，beta 初值 0.1 |

DREAMER 输入为 `[B,3,14,5]`。E1–E5 最终表征均为 `[B,128]`，经 6 个源域 adapter 后映射为 `[B,512]` 与固定 CLIP 文本原型比较。

## 推荐运行顺序

项目已经在 `.idea/runConfigurations/` 中配置好以下 PyCharm 运行项：

- `CAST L1 00 Smoke E5`
- `CAST L1 10 Screen E0` 至 `CAST L1 15 Screen E5`
- `CAST L1 20 Full E0` 至 `CAST L1 25 Full E5`
- 原有的 `Rapid Pilot-3 R0` 至 `Rapid Pilot-3 R4` 保持不变

在 PyCharm 右上角运行配置下拉框选择对应项目，再点击绿色三角或按 `Shift+F10` 即可。不要直接对编辑器里的任意 Python 文件按运行；应确认右上角显示的是上述命名配置。

在 PyCharm 中把脚本设为：

`experiments/crossSubject_geosem_stda_sgda.py`

工作目录设为项目根目录，然后把以下参数放入 Run Configuration 的 Parameters。

### 1. 极小冒烟（不是实验结果）

```powershell
--dataset_name dreamer --rapid_pilot3 --rapid_variant e5_strong_de_full_st --rapid_smoke_test --cast_balanced_screening --cast_samples_per_class 2 --epochs 1 --batch_size 4 --device cuda:0
```

### 2. 均衡小样本筛选

逐个替换 variant 为 E0–E5 的完整名称：

```powershell
--dataset_name dreamer --rapid_pilot3 --rapid_variant e0_r4_de_only --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --device cuda:0
--dataset_name dreamer --rapid_pilot3 --rapid_variant e1_strong_de --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --device cuda:0
--dataset_name dreamer --rapid_pilot3 --rapid_variant e2_strong_de_channel --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --device cuda:0
--dataset_name dreamer --rapid_pilot3 --rapid_variant e3_strong_de_channel_graph --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --device cuda:0
--dataset_name dreamer --rapid_pilot3 --rapid_variant e4_strong_de_channel_graph_multiscale --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --device cuda:0
--dataset_name dreamer --rapid_pilot3 --rapid_variant e5_strong_de_full_st --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --device cuda:0
```

筛选结果位于 `results/dreamer_cast_level1_screening_n128/`。至少同时检查 mean best Acc、macro-F1、balanced accuracy、两类 recall 与预测类别比例，不能仅按 Acc 晋级。

### 3. 正式全量运行

去掉 `--cast_balanced_screening` 和手工 `--epochs` 后，Rapid 配置默认运行 100 epochs：

```powershell
--dataset_name dreamer --rapid_pilot3 --rapid_variant e5_strong_de_full_st --device cuda:0
```

正式结果位于 `results/dreamer_cast_level1/`。应对 E0 和所有需要报告的 E1–E5 分别执行同一命令，只替换 variant。

## 冻结式单模块验证

先保存前一级模型固定终轮检查点。这里保存的是固定 final epoch，不是按目标标签挑出的 best epoch：

```powershell
--dataset_name dreamer --rapid_pilot3 --rapid_variant e1_strong_de --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --cast_save_final_checkpoint --device cuda:0
```

然后以 E1 检查点初始化 E2，冻结所有名称和形状匹配的已加载参数。`{target}` 会分别替换为 12、2、4：

```powershell
--dataset_name dreamer --rapid_pilot3 --rapid_variant e2_strong_de_channel --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --cast_pretrained_checkpoint "results/dreamer_cast_level1_screening_n128/E1_strong_de/final_checkpoint_subject{target}.pt" --cast_freeze_loaded --device cuda:0
```

此时 E1 已有参数被冻结，只有 E2 新增的动态通道模块参与梯度更新。E2→E3、E3→E4、E4→E5 可按同一方式替换前一级目录和当前 variant。日志与 `run_config.json` 会记录加载及冻结的参数张量数量。

注意：冻结实验与端到端实验回答不同问题。前者回答“在固定旧模块时新增模块能否带来信息”，后者回答“整套结构联合优化后的最终性能”。
