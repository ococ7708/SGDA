# SEED / SEED-V 单 Session 探索性实验

## 科研定位

在 SEED 与 SEED-V 上各选一个 Session 做跨被试实验，适合作为代码/预处理可行性检查及外部数据集探索；它不能单独证明跨 Session 稳定性，也不是当前 SEED-IV C1 优化的正式结果。当前范围固定为 SEED Session 2、SEED-V Session 1。SEED 是三分类、15 名被试；SEED-V 是五分类、16 名被试。两套数据分别训练、分别报告，不能把分类准确率直接合并或当作同一任务的配对样本。

当前只做代码准备与 Smoke：SEED Session 2 target 1、SEED-V Session 1 target 1 各运行 2 epoch，验收数据加载器、标签映射、维度和端到端入口；本轮不启动全量 LOSO。后续另行安排时，完整探索范围分别为 SEED Session 2 的 15 个 target、SEED-V Session 1 的 16 个 target，200 epochs、batch 64、seed 42、sample_length=3、stride=1，沿用 GeoSem-STDA / sparse reliability / ReSGCA 设置。每个 target 的其余被试为 source；两个数据集绝不互相作为训练数据。

### 解释边界

共享 runner 为历史可比性按每个 outer target 的目标标签逐 epoch 评价并记录目标准确率最高的 epoch。这个选择本身接触了评价标签，因此该运行仅可作为探索/筛查，最高 epoch 指标会偏乐观。若要作论文确认性结果，应另外冻结源域验证选择规则或固定最终 epoch，并在看目标结果前确定规则。此处不改变原 runner 的模型算法，只提供受控的一 session 启动方式。

### 本轮 Smoke 记录

| 数据集 / Session | Fold | 结果 | 说明 |
|---|---|---|---|
| SEED / Session 2 | target 1, seed 42, 2 epochs | 完成；smoke acc 0.4923、Macro-F1 0.4891 | 仅工程验证 |
| SEED-V / Session 1 | target 1, seed 42, 2 epochs | 完成；smoke acc 0.3963、Macro-F1 0.3549 | 仅工程验证 |

SEED-V 首次 smoke 暴露类别标签为 `float64` 导致 `np.bincount` 报错；类别权重函数现在显式将标签转为 `int64` 并检查类别范围，修复后重跑通过。加载 pickle 时仍有 NumPy 2.4 的 `dtype(align=0)` 弃用警告，但当前数据读取与训练成功。上表数值不作模型效果解释。

## 执行

先确认 `data_utils/constants/path_mapper.py` 指向本机数据目录。SEED-V 预处理文件 `{1..16}_{data,label}.npy` 应位于 `SEED_V/unzipped_DE_features/`；加载器原本直接在 `seedv_de_lds` 根目录找文件，因此路径映射已修正到实际数据子目录。

PowerShell 中先用已安装 PyTorch 的项目 Python 做 Smoke：

```powershell
$env:SGDA_PYTHON = "C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe"
.\experiments\run_seed_seedv_session1_exploratory.ps1 -Stage Smoke
```

脚本固定为 SEED Session 2、SEED-V Session 1。本轮只执行上面的 Smoke；未来完整 LOSO 必须另行安排后再显式启动：

```powershell
.\experiments\run_seed_seedv_session1_exploratory.ps1 -Stage SingleSessionLOSO
```

只跑其中一个数据集时传 `-Datasets @("seed")` 或 `-Datasets @("seedv")`。SEED 与 SEED-V 分别写入各自的结果目录，按各自 run 的 `run_config.json`、训练日志、epoch log 和 subject CSV 追踪。

## 三人分工

- 组员 C：负责 SEED Session 2 和 SEED-V Session 1 两项外部数据集工作。当前两个 smoke 已完成；未来如另行安排，分别完成 SEED 15-target LOSO 与 SEED-V 16-target LOSO。另独立复核输入路径、session/target 范围、固定参数、输出文件和类别映射，复算每个数据集单独的汇总表。特别核查 SEED-V 五类 label/text 映射和少数类表现。
- 组员 A、B 的 C1 A/B 工作包与对应三-session共同范围见 README 的“Immediate three-member assignment”；两名成员都不按 session 拆分 C1。

这次分工是按两个外部数据集做一次单 Session 的方法可行性尝试，不是说它们分别拥有“一个被试/一个 run”的正式样本数。当前交付止于代码和 smoke 验收，不含全量训练；未来全量结果才包含对应数据集全部 LOSO targets。
