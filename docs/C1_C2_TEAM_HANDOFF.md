# Task 06 — C1/C2 三人实验交接说明

## 当前可交付状态

- 已实现 C1 的 E0–E5 数学头、低秩度量、共同平面旋转和多源分支融合。
- 已实现端到端 SEED-IV C1 训练入口、目标 best checkpoint、同 checkpoint 的 Accuracy/Macro-F1/Balanced Accuracy/recall/confusion matrix。
- 已实现 C2 pairwise residual、真实 CE utility label、U1/U2 一致性求解、共享 MLP router 和 OOF 声明校验训练入口。**C2 目前只是基础组件及证据包消费者，不是完整正式实验流水线**：尚未自动训练 OOF teachers、完成 V/U 选择评价及全部对照；声明校验也不能替代 checkpoint 级独立审计。
- 原文件 `experiments/seediv/crossSubjects_seediv.py` 未修改。原 SGDA 结果须单列，不能与 C1 文件互相覆盖。
- 数学单元测试 4/4 通过；synthetic C1/C2 acceptance smoke 通过。尚未代替真实 EEG 首 fold smoke，也未运行 200-epoch 正式实验。

## 统一环境与首次验收

在仓库根目录运行：

```powershell
C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe tests\run_c1_c2_task06_tests.py
C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe experiments\c1_c2_acceptance.py --config configs\seediv_c1_full45.json
```

负责人先执行固定真实 smoke（session 1 / target 1 / E0–E5，各 2 epochs）：

```powershell
C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe experiments\run_c1_assignment.py --session 1 --targets 1 --variants E0,E1,E2,E3,E4,E5 --seeds 42 --device cuda:0 --smoke --execute
```

只有真实数据 smoke 六组均生成 `COMPLETE` 后再分发正式任务。不要根据 target 1 的结果改配置。

## 三位组员分工

三人只按 session 切分，避免重复和漏跑：

| 组员 | 固定任务 | 正式命令 |
|---|---|---|
| 组员 A | Session 1，15 targets，E0–E5，seed 42/43/44 | `python experiments/run_c1_assignment.py --session 1 --device cuda:0 --execute` |
| 组员 B | Session 2，15 targets，E0–E5，seed 42/43/44 | `python experiments/run_c1_assignment.py --session 2 --device cuda:0 --execute` |
| 组员 C | Session 3，15 targets，E0–E5，seed 42/43/44 | `python experiments/run_c1_assignment.py --session 3 --device cuda:0 --execute` |

若先按任务书做 seed 42 首轮，在命令后加 `--seeds 42`。E1/E2/E4 使用已记录的 EMOD V-A 文件及“中心化最小二乘映射到 CLIP 后 QR 正交化”；不得自行改 V-A、prompt、rank、gamma 或 tau。E5 使用 B_proto，不能称为语义基底。

## 每位组员必须交付什么

每位组员交回整个目录 `results/seediv_c1/<config_hash>/` 中属于自己 session 的内容，并附：

1. `run_manifest.json`、`metrics.json`、`target_best.pt`、`COMPLETE`，不得只截图最终准确率。
2. 每个 run 的 session、target、variant、seed、best epoch、best Accuracy、同轮 Macro-F1 和 Balanced Accuracy。
3. 失败 run 清单及完整 traceback；OOM/NaN 记为 failed，不能写成 0 分。
4. 首个真实 run 的 GPU 型号、每 epoch 时间、峰值显存和 checkpoint 磁盘占用；禁止估算准确率。
5. 本人运行时的 Git commit SHA、conda 环境导出和数据路径/数据文件 hash 清单。
6. 明确声明是否改过代码或配置。原则上答案必须是“没有”；任何改动先停跑并报告。

上传前自查：

```powershell
python analysis/summarize_c1_results.py --allow-partial
```

总负责人收齐三份后，去掉 `--allow-partial`；缺任何计划 cell 时汇总程序会失败，不会把不完整结果冒充完整实验。

## C1 接口消融（与默认主表分开）

默认配置是“独立度量头 + 各分支先打分再融合 logits”。共享度量对照使用独立配置，训练时直接监督 uniform fused source representation，目标推理时使用距离权重融合表示：

```powershell
python experiments/seediv_c1_end_to_end.py --config configs/seediv_c1_shared_fused.json --variant E2 --session 1 --target 1 --seed 42 --device cuda:0 --smoke
```

共享配置与默认配置具有不同 effective-config hash，结果不可覆盖或拼接。`tau` 是类别 logits 温度，`fusion_tau` 是源权重温度，两者必须分别报告。共享度量不自动代表概率已校准。

## C2 的交付边界

C2 必须在固定 9-fold 开发协议上，以 subject-level OOF teacher 生成 evidence package。窗口随机切分不合格。每个 `.npz` 必须配套 provenance JSON，列出每个 `held_out` 与 `teacher_train`；`seediv_c2_utility.py` 会拒绝 teacher 看过 held-out subject 的包。

注意：当前检查只能验证 provenance 文件中的集合声明自洽，不能独立证明 teacher checkpoint 的真实训练数据。正式 C2 还需要自动 OOF teacher 训练、checkpoint/训练样本 hash 绑定、V 选模/U 终评、逐轮完整系统 best 保存，以及置信度、距离和普通 MoE 对照；完成这些之前不得写“C2 正式验证完成”。

C2 每 fold 至少交付：reference/expert logits、真实标签及 sample/session/subject/trial/window ID、source distance、prototype pair feature、OOF provenance、router checkpoint、Q0–Q3/U1/U2 的 CE/Accuracy/Macro-F1/BA、负迁移率和动作率。C2 正式 45-fold 只有在任务书开发门槛通过后才启动。

## 禁止事项

- 不改原 SGDA 源码来“复现”；另行运行原法并单独保存。
- 不用 target 表现调配置，不删负结果，不把 seed/head 当独立被试。
- 不覆盖任何历史 `results`；本阶段使用带 config hash 的新目录。
- 不把 synthetic acceptance 当真实 EEG smoke，不把 target-best 称为独立测试集模型选择。
