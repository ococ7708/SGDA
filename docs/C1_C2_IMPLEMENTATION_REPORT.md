# Task 06 C1/C2 实施与验收记录（2026-09-24）

## 实施范围

本次只完成并验证模型、训练入口、配置、结果汇总与三人交接；没有启动 200 epoch 正式矩阵，也没有覆盖任何历史结果。原始 `experiments/seediv/crossSubjects_seediv.py` 相对当前 Git HEAD 的 diff 为零。

核心映射：

- C1 数学模块：`models/context_affective_metric.py`
- C1 端到端入口：`experiments/seediv_c1_end_to_end.py`
- C2 router：`models/pairwise_utility_router.py`
- C2 action/consistent solve：`utils/pairwise_consistency.py`
- C2 OOF 包训练入口：`experiments/seediv_c2_utility.py`
- 冻结配置：`configs/seediv_c1_full45.json`、`configs/seediv_c2_development9.json`、`configs/seediv_c2_full45.json`
- 三人分工：`docs/C1_C2_TEAM_HANDOFF.md`

## 测试记录

### 数学测试

命令：

```powershell
C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe tests\run_c1_c2_task06_tests.py
```

结果：4/4 PASS。覆盖低秩/稠密一致与梯度、谱界/关系矩阵、E0/E1/E2/E3/E4/E5 控制、checkpoint state 重载、pairwise utility/null action/常数平移不变、T/V/U 与 OOF 被试隔离。

### Synthetic acceptance

`experiments/c1_c2_acceptance.py` PASS：C1 六变体 logits 均为 `[12,4]` 且有限；C2 feature 为 `[12,3,6,13]`，utility 为 `[12,3,6]`。

### 首个真实 fold smoke

Session 1 / Target S1 / seed 42 / 每组 2 epochs 全部执行成功，输出在 `results/seediv_c1_smoke/<config_hash>/`，不与正式目录共用：

| Variant | smoke best accuracy | 状态 |
|---|---:|---|
| E0 | 0.625 | PASS |
| E1 | 0.500 | PASS |
| E2 | 0.500 | PASS |
| E3 | 0.500 | PASS |
| E4 | 0.625 | PASS |
| E5 | 0.500 | PASS |

这些是每类仅 2 个样本的工程 smoke，绝不能作为实验结论。另以 seed 99 做了 1 epoch 中断 + `--resume` 续至 epoch 2，恢复成功；期间发现并修复 CUDA RNG state 的 CPU ByteTensor 恢复问题。

## 已知边界

- C1 使用历史兼容的 target-best 报告规则，manifest 明确标注其非出版级独立模型选择。
- 当前 normalization 沿用并显式标注历史 transductive per-subject z-score；所有 C1 组必须一致。
- C2 入口消费 OOF evidence package，不负责假装已有 teacher 证据；开发阶段必须先实际生成 subject-level OOF 包。
- 正式 SGDA 原法复现仍需按原文件单列运行；为保护基线，本次没有修改它。
