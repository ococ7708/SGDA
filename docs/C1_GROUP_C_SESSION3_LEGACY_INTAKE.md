# C1 组员 C（Session 3）旧协议结果接收记录

接收日期：2026-09-27

结论：**不合并到正式 C1 结果；必须使用当前代码重新运行。**

## 1. 实际接收内容

源目录：`C:\Users\oc200\Desktop\数据结果\sessionC\bd6c9b46c9db30ae\`

- 文件数：540
- 总字节数：1,218,542,453
- 文件树复合 SHA-256：`54DDD5E12ECDA96E1C74E75652D7C46E5765DE1955719337D91AD2FD77C595D6`
- 运行数：90
- 范围：Session 3，E0–E5，targets 1–15，**仅 seed 42**
- 每个 run 均含 `COMPLETE`、manifest、metrics、200-epoch history、`last.pt` 和 `target_best.pt`

## 2. 机器核验

这90个旧协议 run 的内部完整性通过：

- 90/90 `COMPLETE=PASS`；
- 90/90 history 为200 epochs；
- 90/90 metrics 为有限值；
- 90/90 `target_best.pt` 可读取；
- checkpoint epoch/metrics 与外部 `metrics.json` 及首次最高 target accuracy 一致；
- manifest 的 session、target、variant、seed 与目录一致。

旧协议 seed-42 的描述性均值如下。它们只用于识别本批数据，**不得与正式 A/B 结果拼表或用于论文主结论**。

| Variant | Legacy Session 3 seed-42 accuracy |
|---|---:|
| E0 | 74.531% |
| E1 | 73.213% |
| E2 | 74.341% |
| E3 | 72.963% |
| E4 | 74.806% |
| E5 | 74.358% |

## 3. 拒绝合并原因

### 3.1 配置/实现哈希不一致

本批目录和 manifest 对应 `bd6c9b46c9db30ae`。该值可由修复前提交 `cd987e5` 的旧配置重新计算得到。当前正式 A/B 结果使用 `592351bfb649e601`，对应修复后的提交 `872b65c`。

旧 manifest 只有 `config` 字段，不含当前要求的：

- `file_config`
- `effective_config`
- `effective_config_hash`
- `run_kind`
- `fusion_mode`
- `head_sharing`
- `training_path`

这不是单纯的目录名差异。`872b65c` 包含会影响预测结果的实质修正：

1. 正式默认改为各 source branch 在自己的 embedding 上先计算 logits，再按 source weights 融合；旧版把融合后的同一 embedding 送入各分支头。
2. centroid 在显式 `model.eval()` 下、使用非 shuffle loader 计算；旧版存在 Dropout/loader 状态影响。
3. 有效配置与命令行覆盖统一进入哈希，避免文件配置与实际运行参数不一致。
4. resume 时先恢复 RNG，再创建 shuffle iterator。

因此不能把 `bd6...` 重命名成 `592...`，也不能只转换 manifest 后使用。

### 3.2 计划矩阵不完整

正式 Session 3 需要：

```text
6 variants × 3 seeds × 15 targets = 270 runs
```

本批只有 seed 42 的90个 runs，缺 seed 43、44 共180个 runs。即便实现版本正确，也仍不满足正式交付矩阵。

## 4. 处置

- 未把任何文件复制或合并到 `results/seediv_c1/592351bfb649e601/`。
- A/B 正式结果保持 `complete=540, missing=270`。
- 原始 Session C 目录保持不变，未修改、删除或覆盖。
- 本批只能标记为 `legacy / rejected for formal aggregation`。

## 5. 正确重跑要求

组员 C 必须从 GitHub `main` 的提交 `872b65c` 或其代码等价后续版本开始，确认以下命令计算出的正式哈希为 `592351bfb649e601`：

```powershell
python -c "from utils.seediv_c1_c2_protocol import *; print(canonical_hash(resolve_c1_config(load_config('configs/seediv_c1_full45.json'))))"
```

输出正确后运行完整 Session 3：

```powershell
python experiments/run_c1_assignment.py --session 3 --device cuda:0 --execute
```

不要加 `--seeds 42`；默认配置会运行42、43、44三个 seeds。旧 `bd6...` checkpoint 与新实现不兼容，不能 resume 到正式目录。重跑交付的顶层目录必须是：

```text
results/seediv_c1/592351bfb649e601/
```

且应含270个 Session 3 `COMPLETE`。
