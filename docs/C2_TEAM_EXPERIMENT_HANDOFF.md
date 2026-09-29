# C2 与外部单 Session 实验：组员接口及交付清单

## 目标与边界

- C1 优化探索现在冻结：不要启动新的 C1 A/B 或共享度量头实验。
- 主线只验证 SEED-IV C2：冻结同一 E0 参考模型后，检查当前窗口证据是否能预测有效修正，以及历史信息是否超出普通 EMA 和容量匹配对照。
- C2 按 Session 1、2、3 分别进行 LOSO；不跨 Session 混合训练。每种方法覆盖全部 15 个 target 和 seed 42/43/44，即每方法 135 个外层 run。
- SEED Session 2 与 SEED-V Session 1 是独立的补充探索，不是当前 C2，也不与 SEED-IV 合并统计。
- 组员只提交可复核的原始 run 产物、日志及本组汇总；最终合并、配对分析和总报告由项目负责人完成。

## 开跑前的硬性门槛

目前审计到 C1 E0 的 Session 1、2 各 45 个 parent，共 90/135；Session 3 的 45 个 parent 不在当前已审计目录中。旧的单折 C2 pilot 属于旧的两 Session 配置，不能替代新三 Session 配置的 pilot。**在实际补齐并审计全部 135 个精确匹配的 E0 parent、重新通过三 Session C2 pilot 并由项目负责人签署 acceptance 以前，不启动全量 C2。**如果组员另有 Session 3 E0 产物，先按下文挂载并审计；不允许用别的配置、seed 或重新训练的 E0 替代。

所有成员先做数据预检、代码/测试复核和结果目录权限检查。只有负责人明确宣布门槛通过后，才能运行正式组别命令。不要用 smoke 指标判断方法有效，也不要挑 target 子集。

## 数据目录接口

无需编辑 `data_utils/constants/path_mapper.py`，通过环境变量给每台机器注入本地根目录。默认目录只适用于项目负责人的机器。

```powershell
$env:SGDA_SEEDIV_DATA_ROOT = 'E:\datasets\SEED_IV'
$env:SGDA_SEED_DATA_ROOT   = 'E:\datasets\SEED'
$env:SGDA_SEEDV_DATA_ROOT  = 'E:\datasets\SEED_V\unzipped_DE_features'
# 可选：此目录本身必须直接包含 seed42/session1_target01/target_best.pt 等子目录
$env:SGDA_C1_E0_ROOT       = 'E:\shared\results\seediv_c1\592351bfb649e601\E0'
```

实际结构约定：

- SEED-IV 根目录下有 `eeg_feature_smooth/label.mat`，且 `eeg_feature_smooth/1`、`2`、`3` 中各有 subject 1–15 的单个 MAT 文件。
- SEED 根目录下有 `ExtractedFeatures/label.mat` 和 45 个 subject/session MAT 文件。
- SEED-V 根目录下直接有 `1_data.npy`、`1_label.npy` 至 `16_data.npy`、`16_label.npy`。
- E0 parent 根目录下直接有 `seed42/session1_target01/target_best.pt` 这类目录。其 run manifest/metrics 必须与 C1 config hash `592351bfb649e601` 相符。

预检命令只检查目录和文件数量，不读取/修改数据：

```powershell
python experiments\check_dataset_interface.py seediv
python experiments\check_dataset_interface.py seed
python experiments\check_dataset_interface.py seedv
```

SEED-IV C2 代码测试：

```powershell
python tests\test_seediv_c2_pipeline.py
```

将各预检 JSON、测试输出、Python/PyTorch/CUDA 版本、GPU 型号及 `git rev-parse HEAD` 交给负责人。不要把 EEG、CLIP 权重、checkpoint 或含隐私的数据提交到 Git。

## C2 三人分工

各方法严格覆盖相同的 3 sessions × 15 targets × 3 seeds。方法组由配置 `team_assignments` 唯一规定，运行器会校验负责人和方法列表；每个组独立写报告/summary，不覆盖其他组。

| 成员 | 方法 | 主要工作与交付重点 |
|---|---|---|
| A | `B0`, `B1` | 固定 E0 融合基线和因果概率 EMA。交付逐折 B0 对原 E0 的复现检查、parent epoch/哈希、B1 平滑结果、样本级最终融合错→对/对→错诊断。 |
| B | `C2-current`, `C2-current-Huber` | OOF/source-dev teacher 证据、证据缓存及来源审计；当前特征 MSE 主方法和同结构 Huber 对照。交付四个 teacher 的实际训练/留出样本名单、缓存与 SHA-256，并将同一缓存移交 C。 |
| C | `C2-current-EMA`, `C2-history`, `C2-current-matched` | 在相同证据上检验输出平滑、严格过去历史增益和容量匹配。交付 trial reset、时序因果、整段/分块一致性及特征分布检查。优先复用 B 交付且哈希核验通过的 teacher/cache。 |

正式运行示例（负责人通过门槛并通知开始后才执行）：

```powershell
python experiments\seediv_c2_experiment.py group --owner A --execute --device cuda:0
python experiments\seediv_c2_experiment.py group --owner B --execute --device cuda:0
python experiments\seediv_c2_experiment.py group --owner C --execute --device cuda:0
```

每组命令自动使用完整三 Session 范围及配置中指定的方法；不要加 `--sessions`、`--targets`、`--seeds` 或自行改 `--methods`。每名组员先从负责人取得当前 config、代码 commit、pilot acceptance 与 parent audit；运行产物须记录这些精确版本。若组员工作区不能访问共同的结果目录，按负责人指定的目录复制原始产物，不要仅交截图或手抄汇总。

### 每位成员必须交付

1. 运行身份：Git commit、干净/脏工作区状态、C2 config 文件与 hash、implementation hash、parent config hash、每折实际 parent checkpoint SHA-256、数据 fingerprint、运行环境/GPU。
2. 完成性：完整的 session/target/seed/method 清单、每折 PASS/FAIL、失败日志与错误原因；不得隐藏失败折或只报成功折。
3. 模型选择：冻结 E0 的 `parent_best_epoch`；router 的 `router_best_epoch`；source-dev 选择 checkpoint；正式 best-accuracy checkpoint，以及同一 checkpoint 的 Macro-F1、balanced accuracy、CE、混淆矩阵。B0/B1 不重新训练 E0。
4. 逐样本数据：sample/trial/window 时间索引、真值、参考预测、最终预测、所选源/类别对动作或 no-op、预测 utility、实际 utility。提供能复算的 CSV/Parquet 和计算脚本。
5. 机制统计：accuracy、Macro-F1、BA、CE、动作/no-op 比率、负实际效用率、最终融合错→对和对→错的绝对样本数与比例；不以分支平均翻转率代替最终融合统计。
6. 训练与证据：完整 epoch history、模型 checkpoint、teacher train/held-out/source-dev subject 和 sample ID、标准化拟合范围、数据 fingerprint、cache manifest 和 hash。router 训练期间 teacher 参数必须冻结。
7. 组内分析：按 session 给出所有 target/seed 的结果、均值/标准差及失败情况；当前/历史配对必须对应同一个 session-target-seed 和同一个缓存。把原始运行目录一并交付，负责人需要从原始记录复算。

严禁目标标签进入 router 训练、history 特征、超参选择或 source-dev checkpoint 选择。目标标签仅可用于预先规定的最终评价和 best-accuracy 记录；报告应明确目标 best 选择带来的额外选择机会。

## SEED 与 SEED-V 补充探索

这部分使用仓库 GeoSem-STDA exploratory runner，不是 SEED-IV C2；固定 SEED Session 2、SEED-V Session 1，seed 42，各自全部 targets（SEED 15 人、SEED-V 16 人）。既有 target 1、2-epoch smoke（SEED acc .4923、SEED-V acc .3963）仅证明工程入口可运行，**不能证明对应 Session 或 target 难以识别**。要总结难例，必须先完成所选 session 的全 target LOSO，再依据逐被试及类别指标识别低表现 subject/class；结论只适用于这一个 session。

先完成对应数据预检，再单独启动：

```powershell
python experiments\check_dataset_interface.py seed
python experiments\check_dataset_interface.py seedv
$env:SGDA_PYTHON = (Get-Command python).Source
.\experiments\run_seed_seedv_session1_exploratory.ps1 -Stage SingleSessionLOSO -Datasets @('seed')
.\experiments\run_seed_seedv_session1_exploratory.ps1 -Stage SingleSessionLOSO -Datasets @('seedv')
```

两套数据分别保存环境/代码/data fingerprint、所有 target 训练日志和 checkpoint、逐被试及逐类别混淆矩阵、Acc/Macro-F1/BA/CE、完整失败清单、可复算脚本。由于现有 runner 使用目标标签逐 epoch 记录并取最佳结果，此结果属于探索/筛查，不应当作无偏确认性测试。不得拼接 SEED 与 SEED-V 的准确率或共用一个总体均值。

## 最终整合责任

由项目负责人保管各组原始 run bundle，核对 config/commit/parent/data/cache hashes，复算全部指标并生成统一结果表。C2 分别汇报 Session 1、2、3，再给跨 session 汇总；基于相同 session-target-seed 配对比较方法，统计不确定性按被试聚类，不能把 135 个 run 或重叠窗口当成独立受试者。SEED 和 SEED-V 各自单列。最终需要回答：C2-current 是否改善冻结 E0、哪些动作贡献纠错/造成伤害；history 是否超过普通 EMA 和容量匹配对照。若收益不一致或审计门槛失败，如实汇报负结果/不确定性并停止扩展，不临时挑有利折。
