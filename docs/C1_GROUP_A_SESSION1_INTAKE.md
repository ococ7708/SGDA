# C1 组员 A（Session 1）结果接收与合规审计

接收日期：2026-09-27
结论：**核心实验数据通过；执行 provenance 条件合规，存在缺项和环境偏差。**

本文以实际运行目录、JSON、checkpoint 和环境文件为主要证据。`交付说明.txt` 只作为组员声明来源，不用于替代机器核验；声明与实际数据冲突时以实际数据为准。

## 1. 接收材料及完整性

| 材料 | 字节数 | SHA-256 |
|---|---:|---|
| `seediv_c1_session1_592351bfb649e601.zip` | 3229572058 | `7694C5B6DF24B6A1C7827E52239521AE47FB9131A9A167685E8E35B26A2C95F3` |
| `sgda_env.yml` | 5934 | `29E5C0A7D3410483E66333F133B9663DEDCD9507B87E436E2835DE8E3B9023DF` |
| `交付说明.txt` | 1583 | `A239BC2E87553277B5DC4CEC249585ED29A1EF0065537EEEAEE82D2CA9F58F1A` |

ZIP 使用 7-Zip 22.01 完整测试，结果为 `Everything is Ok`：295 个目录、1622 个文件、解压大小 3,674,087,182 字节。

环境文件和交付说明的保留副本位于：

```text
artifacts/c1_intake/group_a_session1_20260927/
```

## 2. 实际运行数据审计

- effective-config hash：`592351bfb649e601`，与当前正式配置一致。
- 实际运行矩阵：Session 1 × E0–E5 × seeds 42/43/44 × targets 1–15，共 270 cells。
- 每个变体 45 cells；不存在其他 session、variant、seed 或 target 目录。
- 270 个 cell 均含且非空：`run_manifest.json`、`metrics.json`、`epoch_metrics.json`、`target_best.pt`、`last.pt`、`COMPLETE`。
- 270 个 `COMPLETE` 均为 `PASS`；所有 metrics 均为有限数值。
- 270 个 manifest 的 file/effective config 与当前 `configs/seediv_c1_full45.json` 解析结果逐字段相同；均为 `run_kind=formal`、`smoke=false`、`device=cuda:0`。
- 每个 `epoch_metrics.json` 均为 200 epochs。
- 全部 270 个 `target_best.pt` 均可由 PyTorch 加载，含 model、mechanism 和 manifest；checkpoint epoch/metrics 与对应 `metrics.json` 完全一致，并对应 200-epoch 历史中首次达到最高 target accuracy 的 epoch。异常数为 0。
- 组员目录的 `fold_results.csv` 含 270 行，`summary.json` 正确列出其余两个 session 的 540 个缺失 cells。
- 复制后对 1620 个原始运行文件逐个比较 SHA-256：缺失 0、差异 0。没有修改 checkpoint、metrics、历史或 manifest。

实际 Session 1 target-best accuracy 汇总如下；这是从 270 个 `metrics.json` 重新计算的接收统计，不依赖交付说明中的手写数字。

| Variant | n | Mean accuracy |
|---|---:|---:|
| E0 | 45 | 69.3040% |
| E1 | 45 | 69.1878% |
| E2 | 45 | 69.3649% |
| E3 | 45 | 69.5586% |
| E4 | 45 | 68.9027% |
| E5 | 45 | 68.8778% |

交付说明把 E3 写为 `0.6959`；实际均值为 `0.6955859969`，四位小数应为 `0.6956`。项目汇总使用实际 JSON 值。

## 3. 与预设要求逐项对照

| 预设要求 | 实际证据 | 判定 |
|---|---|---|
| 组员 A 只运行 Session 1 的 15 targets、E0–E5、seeds 42/43/44 | 270 个目录及 manifest 全部匹配 | 通过 |
| 使用冻结正式配置，不按 target 1 调参 | 270 个 effective config 逐字段相同，hash 一致 | 配置层面通过 |
| 每个 run 训练 200 epochs | 270 份 epoch history 均为 200 条 | 通过 |
| 保存 COMPLETE、manifest、metrics、target-best checkpoint | 270/270 齐全，checkpoint 全量可读 | 通过 |
| 同一 best checkpoint 报告 Accuracy/Macro-F1/BA 等 | checkpoint metrics、metrics.json、epoch history 全部一致 | 通过 |
| 失败/OOM/NaN 不得伪装成 0 分 | 计划矩阵 270/270 完成，metrics 有限，无缺失 cell | 数据层面通过 |
| 上传前运行 partial 汇总 | 自带 270 行汇总可复算；项目复算一致 | 通过 |
| 提供 conda 环境导出 | `sgda_env.yml` 存在 | 通过 |
| 提供 Git commit SHA | 实际材料和 manifest 均无 SHA | **缺失** |
| 提供数据路径和数据文件 hash 清单 | manifest 不记录数据路径；未交付数据 hash 文件 | **缺失** |
| 首个真实 run 记录 GPU 型号、每 epoch 时间、峰值显存和 checkpoint 占用 | manifest 只证明 `cuda:0`；history 无时间/显存字段。首个 E0 run 的 `target_best.pt` 为 5,377,126 字节、`last.pt` 为 5,427,239 字节；GPU 型号只见于组员文字声明 | **部分缺失** |
| 正式矩阵前先完成真实 smoke 六组 | 交付中没有 smoke 目录或 smoke 日志 | **无法证明** |
| 原则上不改代码或配置；改动需先停跑报告 | 配置完全一致；组员声明改过两个本地路径文件且已报备，但未交付源码快照、diff、Git SHA 或批准记录 | **无法独立证明仅改路径/已批准** |
| 使用统一环境 | 环境为 Python 3.10.8、torch 2.1.2+cu118；与本项目 `sgda_py3.11` 名称和本机执行环境不一致 | **环境偏差** |
| 不把 target-best 称为独立验证选模 | manifest 明确写有 target-best 警告；本报告也按 target-best 表述 | 通过 |

组员声明使用 RTX 4090 D、只修改数据/CLIP 本地路径，并用互不重叠 runner 并行执行。实际结果可以证明运行单元没有重叠且配置一致，但不能从结果文件独立证明 GPU 型号、源码改动范围或事前批准过程。并行执行本身不改变预设实验矩阵，不作为数据拒收理由。

## 4. 合并状态

只把 270 个原始 Session 1 运行目录合并到：

```text
results/seediv_c1/592351bfb649e601/
```

组员 A 自带的 `fold_results.csv` 和 `summary.json` 没有覆盖项目文件。合并 Session 1 和 Session 2 后，使用项目当前汇总脚本从 540 份实际 `metrics.json` 重新生成汇总：

```text
complete=540
missing=270
```

剩余 270 cells 应全部来自 Session 3。最终三人结果收齐前只能使用 `--allow-partial`，不能将当前 540-cell 汇总作为完整 C1 结论。

## 5. 处置建议

本批次无需因结果文件结构或 best-checkpoint 错误而重跑；这些核心项已全量通过。正式复现归档前，应向组员 A 补收：

1. 实际 Git commit SHA，或无法提供时保留“源码版本不可完全追溯”的限制；
2. 数据文件路径及 hash 清单；
3. 正式实验前六组真实 smoke 的日志或结果目录；
4. 首个真实 run 的 GPU 自动采集信息、每 epoch 时间和峰值显存记录；
5. 两个路径文件的实际 diff 或运行时源码快照，以及路径改动获批记录。

Python 3.10 环境偏差不会自动证明数值结果无效，但它降低了与统一 py3.11 环境的严格可复现性，必须在最终实验记录中保留，不应写成“完全相同环境”。
