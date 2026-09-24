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
| E4 | 0.500 | PASS |
| E5 | 0.500 | PASS |

这些是每类仅 2 个样本的工程 smoke，绝不能作为实验结论。表中数值已按审查修订后的 `branch_logits` 和 eval-mode 质心重跑；最终默认 effective-config hash 为 `a7faebfdefa970de`。另以 seed 99 做了 1 epoch 中断 + `--resume` 续至 epoch 2，恢复成功；期间发现并修复 CUDA RNG state 的 CPU ByteTensor 恢复问题。

## 已知边界

- C1 使用历史兼容的 target-best 报告规则，manifest 明确标注其非出版级独立模型选择。
- 当前 normalization 沿用并显式标注历史 transductive per-subject z-score；所有 C1 组必须一致。
- C2 入口消费 OOF evidence package，不负责假装已有 teacher 证据；开发阶段必须先实际生成 subject-level OOF 包。
- 正式 SGDA 原法复现仍需按原文件单列运行；为保护基线，本次没有修改它。

## 2026-09-24 代码审查修订

- 修复质心计算模式：先显式执行 `model.eval()`/`mechanism.eval()`，再在 `no_grad` 下计算源质心，保证质心与目标特征都关闭 Dropout。
- 默认融合改为 `branch_logits`：目标的第 k 个 adapter 表示先交给训练时对应的第 k 个 C1 head，再融合 logits。旧的“先融合表示、再让全部 head 处理同一个表示”仅保留为 `legacy_fused_embedding` 显式消融，不能与新版默认结果混合。
- 新增唯一 `effective_config`：JSON、CLI 覆盖和默认值先合并，模型、优化器、manifest 与结果目录 hash 全部读取同一个对象；学习率、tau、rank、gamma 等 CLI 变化会改变 hash。
- 默认使用独立 source heads + `branch_logits`；另提供单独配置 `configs/seediv_c1_shared_fused.json`，实现共享 head + 源侧 uniform fused representation 直接监督。该配置是接口消融，不能覆盖默认主表；其 E2 首 fold 两轮 smoke 已通过。
- 每轮保存 `mechanism_gradient_norm_mean`、`metric_H_fro_mean`、`metric_H_sample_variance`、`centered_logit_rms`、`prediction_flip_rate` 与源权重熵。默认 E2 smoke 在 epoch 2 的机制梯度范数约 0.0516、H Frobenius 均值约 0.0165、logit 扰动 RMS 约 0.00421，说明模块有梯度且产生非零作用；这仍不是效果结论。
- 更正 C2 状态：当前只完成数学组件、router 和带声明校验的证据包消费者；自动 OOF teacher、V/U 流水线、完整系统 best 与全部对照尚未完成。
- 上一轮六变体 smoke 使用旧融合接口，仅是历史工程记录；修订后 smoke 必须写入新的 effective-config hash 目录并重新验收。
