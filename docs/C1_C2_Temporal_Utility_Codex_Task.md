# C1/C2 升级方案与 Codex 实验任务书

日期：2026-09-27。状态：研究设计与实现任务，未运行新实验。主指标：best accuracy。

## 1. 目标与依据

保留固定 CLIP 情绪锚点、Strong-DE/Channel/Graph backbone、C1 条件化有界度量和 C2 类别对迁移效用主线。首轮升级只检验：历史判别证据能否提高源证据的效用预测，超出普通输出平滑？暂不同时加入新的 backbone、GRU、对比损失、多动作路由或跨数据集训练。

代码依据是此前会话审阅的 ococ7708/SGDA 提交 872b65c4d272a551a9e1e58e6c48610836d3db70，不表示当前 main 未变化。Codex 必须先核实 checkout、AGENTS.md、工作区修改及实际调用链，再适配路径。不能把历史代码审阅或本任务中的拟议设计写成已完成结果。

已知历史实现：C1 主配置 cast_level1 / e3_strong_de_channel_graph，共享 h=128 维，源 adapter 后投影到固定文本原型空间，类别加权 CE+C1 正则；旧源内 GeometricRegularizationLoss 未启用；C1 的 prediction_flip_rate 是源分支平均，不是最终融合翻转率。C2 已有特征、效用标签、Huber router、单动作及多动作组件，但 OOF teacher 闭环需核实完善。

## 2. 文献依据与边界

以下是针对本项目的定向检索，不是 TAFFC 全刊系统综述；不能声称已证明全球首创。

| 文献 | 核实层级 | 对设计的作用 |
|---|---|---|
| CFSDBN, TAFFC 17(2), 2026, DOI 10.1109/TAFFC.2026.3667858 | 用户两份 PDF 正文，方法与实验段 | 已有 3D ResNet/Bi-GRU/GAT/通道选择与图可视化；我们不主打通道或动态图新颖性。其被试独立 70/10/20 划分、5 次随机划分，不等同 SEED-IV 单 session LOSO。 |
| AMA-EEG, TAFFC 2026, DOI 10.1109/TAFFC.2026.3728157 | IEEE 条目、作者当前 README | EEG 与文本/图像语义对齐已经有直接邻近工作。作者 README 当前记载 FACED 跨被试 10 折、SEED LOSO，默认三模态；不能把它直接写为与固定类别文本原型 EEG-only 推理同一协议。 |
| Prompt-Guided Domain Generalization for EEG Emotion Recognition, TAFFC 2026, DOI 10.1109/TAFFC.2026.3658346 | IEEE 摘要 | 学习源域提示并用 cross-attention 融合域特异信息。我们的差异需落在“逐候选动作的损失改善监督”，不能只写自适应融合源知识。 |
| PCL-TDGCN, TAFFC 2026, DOI 10.1109/TAFFC.2026.3671020 | 作者高校书目及官方仓库 | 原型对比与时间动态图已有组合，未审阅完整公式和协议，不作精细优劣判断。 |
| DAEST, arXiv:2411.04568 | 作者原文及官方仓库；此处按作者预印本引用 | 关注 EEG 状态转移及可解释时空成分。借鉴“解释必须对应具体可测量量”，不推断慢情绪等于静态 EEG。 |
| HEDN, arXiv:2511.06782 | 作者摘要/仓库，预印本 | 已有按源迁移难度区分知识迁移策略；作为 C2 邻近工作，不能宣称属于已核实 TAFFC 正式论文。 |

参考链接：
- https://doi.org/10.1109/TAFFC.2026.3667858
- https://github.com/jianjiez100-sys/AMA-EEG
- https://doi.org/10.1109/TAFFC.2026.3728157
- https://doi.org/10.1109/TAFFC.2026.3658346
- https://www.fst.um.edu.mo/personal/fwan/
- https://github.com/YYingDL/PCL-TDGCN
- https://arxiv.org/abs/2411.04568
- https://arxiv.org/abs/2511.06782

## 3. 不改变的实验口径

沿用 best accuracy 主指标，不要求改报最后 epoch。每个 run 保存 best_epoch、best_accuracy 和该 checkpoint 的 Macro-F1、Balanced Accuracy、混淆矩阵、逐样本预测；并列 best 取最早 epoch。不得分别挑不同 epoch 的 F1/BA 拼接成同一 checkpoint 结果。训练 epoch 上限、评估频率和预算保持一致。

论文协议文字：For each subject-session run, we report the maximum target-set accuracy across the prespecified training epochs; companion metrics are computed at the same selected checkpoint.

这是一种明确的 target-best 报告方式，不写成 source-validation-selected checkpoint，也不把 best 定义成 last epoch。target-best 仅为预先规定的事后报告指标；目标标签不得用于部署 checkpoint 选择、超参数/动作选择或实验决策。实际部署结果另报仅由 source-dev 选择的 checkpoint。目标标签不进入梯度、历史特征或 router 推理。源域留出训练仍必须隔离，以使 utility 标签具有明确含义。

固定 checkpoint 的后处理诊断没有新增 encoder best epoch：继承 parent_best_epoch，标记 evaluation_type=frozen_checkpoint_diagnostic。后续完整 router 训练可记录预先限定 router epochs 内的 target-best，并单独记 router_best_epoch；该 target-best 仅作事后报告，不能用于部署、超参/动作选择或实验决策。保留 source-dev 选定 router 的结果作为部署/机制验证，勿混淆两者。

## 4. 方法：局部证据 + 历史判别背景

### 4.1 首轮采用低维、类别对相关的历史

不立即输入 128 维 h 的整段历史。先复用当前 build_router_features 的全部合法特征，再增加下列可解释量。所有预测均在修正前计算，避免新策略的输出递归成为自己的输入。

记参考融合 logits 为 l0_t，专家 k 的 logits 为 lk_t，类别对 (a,b) 统一 a<b：

    m0_t = l0_t[a] - l0_t[b]
    mk_t = lk_t[a] - lk_t[b]
    dk_t = mk_t - m0_t

对 m0、mk、dk 各自维护严格过去的 EMA 摘要。处理窗口 t 时读取 t-1 状态，完成预测后再更新：

    mean_t = alpha * mean_(t-1) + (1-alpha) * value_t
    second_t = alpha * second_(t-1) + (1-alpha) * value_t^2
    variance_t = max(second_t - mean_t^2, 0)

新增特征为过去 mean(m0)、mean(mk)、mean(dk)、std(dk)、当前 dk-历史 mean(dk)、has_history。避免为首个窗口使用当前值冒充历史。初始摘要填零，mask=0，首次更新用当前值初始化；后续 mask=1。必要的特征标准化只在 router train fit。

alpha 初始固定 0.8，只在源域 dev 比较 {0.5,0.8,0.95}；先核实窗口 stride 的实际秒数，报告有效历史跨度，不能把帧数默认当秒。每个 subject/session/trial 重置，每个 epoch/eval pass 重置；禁止跨 trial。相邻滑窗可能共享原始帧，若有时间收益再增加间隔至无重叠的历史对照，以区分重复观测与更长时间信息。

首轮不把 subject ID、trial ID、clip ID、绝对时间位置作为 router 输入。这些仅用于隔离、排序、reset 和审计，避免利用固定刺激顺序。

### 4.2 C2 候选动作

保留已有动作定义以便复现，导出其方程和单元测试；若需要标准化新动作，必须版本化为 action_v2，不覆盖旧版。

可用的明确新动作定义如下（只有旧定义无法满足任务时启用）：

    Delta_kab = eta * clip(mk - m0, -D, D)
    l_action = l0 + 0.5 * Delta_kab * (e_a - e_b)

它使 a-b margin 增加 Delta，保持 logits 总和不变。虽然只改两个 logits，softmax 的所有类别概率都可能改变，因此 utility 必须用完整多类 CE：

    u_kab = CE(l0, y) - CE(l_action, y)
    u_noop = 0

eta、D 从现有代码继承并锁定；若现有代码无参数，试运行默认 eta=0.25、D=2（logit 单位，纯工程初始值，非文献最优）。最终选择只在 source-dev 完成，不用目标标签挑动作幅度。未加权 CE 作为新 utility 的统一解释口径，旧加权形式若存在须作为独立 legacy 条件，不默默替换。

逐样本枚举 K*6 候选，预测效用，最大值>0 才执行单动作，否则 no-op；并列按固定 pair/source 顺序。不加入多个置信度的手工乘积。不以当前真实类别筛选候选，训练和推理都枚举六个类别对。

router 参数共享于所有 source/pair；特征不依赖固定 K 维向量，不用源编号 embedding，使 OOF 的 8 源、dev 的 12 源及正式 14 源可兼容。类别对编码可保留；所有特征方向遵守 a<b。

## 5. 可成立的理论陈述

这些是待核对实现后可写入附录的数学性质，不是泛化或准确率保证，也不单独作为新定理贡献。

### 命题 A：C1 的受限变形

若 B^T B=I，H=H^T 且 ||H||_2 <= gamma<1，则 M=I+BHB^T 有：

    (1-gamma)||v||^2 <= v^T M v <= (1+gamma)||v||^2
    condition_number(M) <= (1+gamma)/(1-gamma)

证明：|v^T BHB^T v| <= gamma ||B^T v||^2 <= gamma ||v||^2。若 gamma=0.5，条件数上界为 3。必须检验实际 H 是谱范数约束；逐元素 tanh 有界本身不足以证明该上界。

若 v 与 span(B) 正交，Mv=v。但归一化分类分数的分母仍受原型范数影响，不能说“补空间内所有分类行为完全不变”。每个 context 固定时 M 定义正定几何；context-dependent M 不是自动满足全局距离公理的单一距离函数。

### 命题 B：效用回归与决策

令 F 为合法可观测信息，v(a,F)=E[u(a)|F]。平方损失总体最优解为 v，要求二阶矩有限和足够表达能力。Huber 的总体目标通常不是条件均值，因此历史 Huber 代码保留为对照，不能直接沿用“无偏期望效用”解释。

若对全部候选动作有 |vhat-v|<=epsilon，且 no-op 精确为 0，选择预测最大效用动作 ahat，则：

    v(astar,F)-v(ahat,F) <= 2*epsilon

证明：在最优动作与所选动作上各插入一次 vhat，中间差非正。该界是条件期望 CE 效用的界，不是逐样本 realized utility 的保证，不是 accuracy 界。实际模型未建立统一 epsilon 时，只报告误差和经验 regret，不能宣称安全保证。

### 命题 C：历史信息的理想决策价值

令当前信息 sigma-field 为 G，加入真实可获得历史后为 H，G 包含于 H；相同有限动作集合且效用可积。则：

    E[max_a E[u_a | H]] >= E[max_a E[u_a | G]]

原因：更多信息的最优策略可以忽略历史；也可由 max 的凸性和条件期望塔式性质证明。它说明研究历史信息有决策论动机，不证明有限样本 MLP 一定改善。训练域到目标域的条件效用稳定性仍是实证假设。

### 可解释量

C1：记录 M 的特征值、谱范数、六个类别对的 normalized prototype metric cosine，以及同 checkpoint 下的真实 margin 改变量。QR 后的 B 轴不直接称为纯 valence/arousal。优先使用基底旋转不变的类别对几何图。

C2：导出预计效用、实际效用、source、pair、动作幅度、no-op、最终错→对/对→错。拟合校准图与零线附近误差；选定动作平均实际效用、负效用率、action rate、realized oracle gap 分开报告。realized oracle 只作事后上界，不是可部署策略。

理论对象 v 是条件均值；实证可观测 u 是一次实现，单样本二者不应混为一谈。个体 u 误差有标签随机性。

## 6. 分阶段最小实验

### Stage A：无需重新训练 encoder 的诊断

使用可获得的 E0/E2 best checkpoints；缺失时列出所需路径，禁止虚构。按原配置复算 parent best accuracy，容差由确定性/数值精度说明。

1. 同一 checkpoint 内开启/关闭 C1，固定相同源融合权重，统计最终融合而非分支平均翻转。此为直接 head 作用，不等于 E0 vs E2 的端到端训练效应。
2. 固定 C2 候选动作，枚举 true-label oracle、no-op，导出可纠正错误比例、被破坏比例及完整 CE utility 分布。另算 accuracy oracle，它与 CE oracle 不等价。
3. 在源域留出诊断上判断动作是否有足够空间。若动作几乎不能纠正错误，先修动作/专家互补性，不继续训练更大 router。目标 oracle 只供事后描述，不用于挑超参。
4. 检查 trial/time metadata、滑窗重叠、原有 LDS/运行归一化是否使用未来。如果原预处理使用未来，只能说新增路由因果，不称全系统在线因果。

### Stage B：低成本 OOF teacher 流水线

先 SEED-IV Session 1，targets [1,8,15]（SEED-IV 数据包中的 subject ID，从1开始，不是 Python 零基数组索引；代码必须在 manifest 同时写明 ID 到内部索引映射；这里是拟议的固定编号，并非按结果选容易/困难），seed=42。已有预先锁定 pilot 列表时优先沿用，并在 manifest 说明。epoch/batch/DE length/stride/优化器等继承现有正式配置，不能默认旧 DREAMER pilot 参数。

每个 outer target t：
- 剩余14个被试中确定性选2个 source-dev D，另外12个为 R。对每个 outer target，按 subject ID 升序列出其余14人，使用 `numpy.random.default_rng(20260927).permutation` 打乱后取前2人为D、剩余12人为R；分别按 target 独立生成。实际生成名单必须写入配置/manifest并冻结，后续以名单为准，不能仅靠种子临时重建或重抽。
- R 按被试分3折，每折4人。各 teacher 仅由另外8人训练（共享 encoder、adapter、projection、标准化和质心全部重新由合法训练数据构建），对4名 heldout 生成 OOF 记录。
- 所有 OOF teachers 均不能使用 outer target t 或 source-dev D 的有标签/无标签训练数据。OOF 被试不参与 teacher checkpoint 挑选；整个 pilot 统一采用一种 checkpoint 策略：固定训练预算的固定 teacher epoch，或 teacher-train 内部验证选择；写入配置并对所有 teacher 一致执行。
- 另训练1个 R=12人的 teacher，用于在 D 上按预先列出的诊断检查机制并选择 router 超参。D 既不在 router train，也不在其 OOF teachers 的训练集中；D 上诊断不得用来更改已冻结的outer-target结果、动作定义或切分。
- 因此每 outer target 共4次新 teacher 训练任务（3个8人训练的OOF teacher + 1个12人R teacher），3 targets 共12次训练任务，不是12个OOF teacher。teacher checkpoint 策略须在pilot配置中统一锁定：固定训练预算/epoch，或统一采用teacher-train内部验证；不得逐teacher临时切换，也不能以OOF标签选择teacher。
- 正式 outer target 推理可复用对应14源的原 best teacher checkpoint。其 adapter 数不同于 OOF，必须依赖共享且 K 无关的 router。保存 train/dev/deploy 的 K 并检查特征分布漂移；不能把由14源训练的共享 encoder 加“去掉一个adapter”当 OOF。
- 若成本有限先只跑 target1 的4个 teacher 完成集成，通后扩到另外2个；已有 manifest 可证明隔离的 checkpoint 才能复用。

这比14个 leave-one-source-out teacher 更便宜，但小 pilot 仅判断方向。3折 teacher 的8源与正式14源差异要在报告列为实际限制，不声称完全同分布。

### Stage C：共享同一证据缓存的最小对照

| ID | 方法 | 用途 |
|---|---|---|
| B0 | 冻结参考融合，无额外动作 | 基线 |
| B1 | B0 + 因果概率 EMA | 普通平滑基线 |
| U0 | 当前窗口 utility router，MSE | 无历史效用选择 |
| U1 | U0 + 输出概率 EMA | 直接检验平滑能否解释收益 |
| U2 | 当前+过去 margin 摘要 router，MSE；无输出平滑 | 首轮核心升级 |
| U2-current | 与U2完全同结构；每个历史槽按固定映射复制对应的当前类别对 margin 特征（mean(m0)←m0-current、mean(mk)←mk-current、mean(dk)←dk-current、std(dk)←0、current-minus-history←0）；不读取真实过去值。history mask 使用与U2相同的可用历史规则，使首窗仍为0、后续为1 | 参数/输入维度匹配对照 |

执行顺序与负责人固定如下：A交付B0/B1及Stage A诊断；B交付U0-MSE（并提供同配置的legacy-Huber损失对照）；C先跑U2。先完成这四项的接口和数据检查，之后C再补跑U1与U2-current；若U2相对U0没有达到预设的source-dev信号门槛，则仍须完成U1/U2-current最小诊断，或在manifest中明确登记停止原因，不得把未运行写成阴性结果。这样Stage C表中的六种方法均有归属，B0/B1不由B或C重复训练。所有router变体采用相同动作、batch、训练预算、初始化种子。训练候选效用时，每个样本总权重相同，其K×6个候选再均分该样本权重；长trial或源数量多不得无意获得更高权重。

若U2有信号，再作真实历史 vs 滞后/无重叠历史；同trial打乱历史仅作有分布偏移的辅助诊断，不能作为单独因果证明。禁用未来双向历史作为可部署主方法。

source-dev 调参采用平均被试 accuracy 与 realized utility 的预先声明规则：先最大化 mean accuracy，并列优先 mean utility 高、模型简单者。初轮 alpha 默认0.8、no-op阈值0；若需要搜索，最多3个alpha，其他参数固定。MSE特征标准化、early stop/epoch选择仅source-dev；target-best仅作事后报告，部署checkpoint由source-dev选择。

### Stage D：扩大条件

只有 source-dev 上 U2 优于 U0 且超过普通平滑/容量匹配解释，才扩完整3 sessions×15 targets、至少3 seeds。实验资源不足时如实保留 pilot 定位，不能写稳定泛化提升。

完整统计以被试为聚类单位，同一被试3 sessions非独立；可对每被试平均session/seed后的方法差进行配对检验/被试bootstrap。提供mean-best、标准差定义、每被试配对差，不能把大量重叠窗口当独立样本。

C1时间升级放在下一阶段：比较 static semantic metric、current-context semantic metric、history-context semantic metric；补rank=2 prototype-subspace随机方向对照，另可有ambient random。语义基底位于类别差空间内，ambient random可能是过弱对照。重复随机基底种子，不以单次随机失败证明语义优越。所有基底对照保持rank/gamma/head结构一致。

### 三位组员的顺序实验分工（按方法，不按 Session）

三位组员依次交接，不把 Session 1/2/3 分给不同组员。Pilot 阶段所有方法使用完全相同的固定数据范围：SEED-IV Session 1、outer targets `[1, 8, 15]`、backbone seed `42`、同一份角色划分和预先锁定的 checkpoint 选择规则。C 的所有 router 方法必须复用 B 交付的同一证据缓存与 teacher checkpoint。这样组间比较的是不同实验方法，而不是不同 Session。

固定 pilot 范围为3个 outer targets。每个 outer target 的14个非目标被试拆为 source-dev `D=2` 与 router/OOF pool `R=12`；R 按被试分3折，每折4人，每个 OOF teacher 只用其余8人训练。分组由负责人一次性确定并保存为 `configs/seediv_c2_temporal_pilot.json` 或配套 split JSON；固定 `split_seed=20260927`，之后不按结果重抽。现有 `source_roles()` 的 T/V/U=8/3/3 分割不是这里的 D/R 协议，不能直接复用或混称。outer target 和 D 均不能进入 OOF teacher 训练或 router 训练标签。

| 顺序/负责人 | 独立负责的实验 | 输入与边界 | 必交付内容及通过条件 |
|---|---|---|---|
| 1. 组员 A：基线与可行性诊断 | 完成 Stage A；建立 B0、B1 和 utility-oracle/no-op 诊断；对可用 E0/E2 checkpoint 做冻结 checkpoint 的 C1 head-off/head-on 诊断 | 只用固定 pilot folds；不重训 encoder；C1 head-on/off 使用相同 checkpoint、相同 source 权重。目标 oracle 仅作事后诊断，不能用于选参数 | 代码与命令；每 fold 的数据/标签/窗口索引清单；E0/E2 checkpoint/config hash；B0/B1 指标；CE oracle 可纠正/破坏比例、accuracy oracle 独立结果、utility 分布；最终融合翻转四格表；预处理未来信息与窗口重叠审计；`stage_a_manifest.json` 和可复算 CSV/JSON。若 checkpoint 缺失，列出确切路径，不得用新训模型冒充冻结诊断。负责人签字后冻结 split、action 定义和基线口径，交给 B。 |
| 2. 组员 B：OOF teacher 与当前窗口 utility 基线 U0 | 完成 Stage B，并在共同证据缓存上实现/运行 U0（current-window features + MSE）；保留 legacy Huber 仅作损失对照，不改变动作定义 | 接收 A 冻结的 split/action manifest。先完成 outer target 1 的3个 OOF teachers + 1个 R=12 source-dev teacher，通过隔离检查后再扩 targets 8、15；pilot 总计12个 teacher 训练任务。D 只用于 router 选模/机制检查，不用于 router train；outer target 只作最终评价，不能用于选择参数 | teacher 训练命令、日志、checkpoint 和 SHA-256；每个 teacher 的实际训练 subjects、标准化 fit subjects、checkpoint 选择来源及数据 fingerprint；OOF evidence `.npz` 与逐 fold provenance；K=8/12/14 特征分布检查；U0 MSE/Huber checkpoint、source-dev 选择记录、outer-target 结果；OOF exclusion 自动审计报告。不得只提交声明式 provenance JSON。通过后冻结共享 evidence package 和 U0 参数，交给 C。 |
| 3. 组员 C：历史 utility 实验 U1/U2 与容量对照 | 完成 Stage C；在 B 冻结的同一 evidence package 上比较 U0、U1、U2、U2-current；不重训或筛选 teacher | U1=U0+因果输出概率 EMA；U2=当前特征+严格过去 margin EMA；U2-current=同结构但历史槽替换为预先定义的当前特征匹配对照。共享动作、候选集合、MSE 主损失、batch、初始化种子、router 训练预算；不得把 U2 特有调参带回 U0 | 因果历史实现和测试；每个 sample 的 history features/mask 复算表；四方法同缓存配对结果；source-dev 选模记录；outer-target CE/accuracy/Macro-F1/BA、realized utility、action/no-op、负效用率、realized oracle gap、校准图及每被试差值；失败案例与日志；`temporal_comparison_manifest.json`。只有 U2 同时优于 U0、U1 和 U2-current 的预设门槛，才提交 Stage D 扩大实验建议。 |

“顺序”表示证据和配置逐步冻结，不表示后一个组员可以改前一个人的方法。任何必要修订都要新增配置版本和哈希，重跑受影响对照，并保留旧结果。A/B/C 都须遵循第9节的共同交付清单；各自的专属交付以上表为准。

### Stage D 的扩大分工

若 Stage C 通过预注册门槛，完整实验仍使用所有3个 sessions、15个 outer targets 和至少3个 seeds。此时仍按方法包分工，不按 Session 切开：

- 组员 A 负责全部 folds 的 B0/B1、冻结 C1 诊断和总体数据审计，覆盖 Session 1–3；
- 组员 B 负责全部 folds 的 OOF teacher/evidence package 与 U0，覆盖 Session 1–3；
- 组员 C 负责全部 folds 的 U1/U2/U2-current 路由比较，覆盖 Session 1–3；
- 负责人冻结总 manifest、核对跨方法 checkpoint/split/hash、合并结果并做 subject-cluster 统计。各组员交付自己负责方法的完整矩阵，不是一个 Session。

完整阶段的目标数量、teacher 训练预算和 GPU 时间必须先由三人按 pilot 实测汇总，再决定是否启动；不得把三个 pilot outer targets 的表现外推成完整45-fold结果。

## 7. Codex 实现清单

先读取仓库指令，保护现有实验与用户改动。以下路径为建议新增文件名，不假定已存在。

1. `docs/C1_C2_AUDIT.md`：记录commit、配置hash、现有损失/动作/融合公式、metadata来源、checkpoint来源及偏差。
2. `experiments/export_pairwise_evidence.py`：有序导出h/分支logits/参考logits/融合权重，支持冻结checkpoint与C1 ablation。
3. `experiments/build_source_oof.py`：实现上述3折隔离与source-dev；训练进程读取显式数据manifest并记录实际加载的subject集合及数据fingerprint，不能仅做声明字符串检查。
4. `models/causal_evidence_history.py`：严格过去的EMA与mask；支持完整trial缓存和在线逐窗两种一致接口。
5. 扩展现有 `seediv_c2_utility.py`：MSE/Huber开关、current/history/current-capacity对照、shared K-agnostic router；冻结teacher，禁止router训练更新teacher。
6. `experiments/evaluate_utility_actions.py`：最终融合预测、CE/accuracy oracle、flip四格表、动作效用及校准。
7. `configs/seediv_c2_temporal_pilot.json`：保存split、seed、alpha、action版本、checkpoint策略、budget和variant，支持dry-run打印run manifest。
8. `docs/C1_C2_TEMPORAL_RESULTS.md`：从真实产物生成方法、实验和限制。无数据时只写待运行，不生成假指标/假SOTA。

### 实现模块责任人

| 负责人 | 首要代码模块 | 验收交接 |
|---|---|---|
| A | `experiments/export_pairwise_evidence.py`、Stage A 诊断/审计脚本、C1几何性质测试、固定 split/action manifest | 输出可复算冻结 checkpoint 诊断、B正交/H谱界/M正定/H=0恢复baseline测试与通过审查的split/action manifest；目标 oracle 只作事后汇总诊断，不得影响方法或fold选择。 |
| B | `experiments/build_source_oof.py`、证据缓存 schema、teacher/data provenance 和 OOF 审计 | 输出通过 subject exclusion 的真实 OOF 包及训练记录；负责人检查实际样本集合/hash 后，才交给 C。 |
| C | `models/causal_evidence_history.py`、`experiments/evaluate_utility_actions.py`、U1/U2/U2-current 比较 runner | 输出因果历史特征、统一证据缓存比较、预测与效用评估；必须通过第8节 future/trial reset/chunk/action/utility 测试。U0由B负责，不重复训练。 |

共用 `seediv_c2_utility.py` 或增加 temporal 配置时，由 B 先冻结 package/config 接口，C 在该接口上实现历史对照。未经负责人记录的新字段、动作或切分不得由后续成员私自改变。

证据缓存最小字段：sample_id, subject_id, session_id, trial_id, window_start, window_end, stride_seconds, class_order, label, teacher_id, teacher_train_subjects, teacher_select_subjects, checkpoint_hash, config_hash, split_role, h（可选）, branch_logits, reference_logits, fusion_weights。标签与router输入分离；导出实际可用source列表，不补虚假源。

动作表最小字段：sample_id, source_id, class_a, class_b, base_margin, expert_margin, delta, predicted_utility, realized_utility, chosen, noop, base_prediction, action_prediction, final_prediction。另存history_features和history_mask用于复算。

## 8. 必需验收测试

- future isolation：修改t之后的全部窗口，t及以前历史路由不变。
- trial reset：两个trial分别推理与拼接推理一致；不同batch/chunk边界结果一致。
- OOF exclusion：实际encoder训练样本/标准化fit样本不含heldout、dev、outer target。
- action identity：delta=0/no-op时logits和utility精确恢复基线；修改量符号、pair orientation正确。
- utility exactness：缓存utility与直接重算完整多类CE一致；正utility不被错误当作必然纠正分类。
- fused flip：直接逐样本比较最终argmax，按样本计数，最后不满batch无偏。
- metric property：实际B正交、H谱界、M正定，以及H=0恢复baseline。
- label separation：修改目标标签只能改变评估指标/best选择，不能改变该epoch输入、历史、动作和梯度。
- frozen teacher：router backward后teacher无梯度/参数更新；训练/推理source数变化接口可运行。

## 9. 必交付结果与论文写作任务

交付可运行代码、固定配置、manifest、README复现实验命令、关键测试输出、真实results.csv/json与图表。若当前环境无数据/GPU/checkpoint，完成实现和合成验收，并给出一条最小pilot启动命令及准确缺件清单；不宣称实际EEG性能已经验证。

### 每位组员的共同交付包

每位组员按自己负责的方法提交独立目录 `results/c1_c2_temporal/<config_hash>/<owner>/<experiment_id>/`，不得覆盖其他实验。共同交付包必须包含：

1. **配置和来源**：冻结配置 JSON、canonical/effective config hash、Git commit SHA、conda `environment.yml`、GPU 型号/驱动/CUDA/PyTorch 版本、运行命令、起止时间、每 epoch 时间、峰值显存、checkpoint 磁盘占用。
2. **split 与数据 provenance**：outer target、source-dev D、R/OOF folds、teacher train/heldout subjects、实际加载的 subject/trial/window IDs、标准化拟合 subject、数据文件路径与 SHA-256。目标被试标签不得进入训练特征或 router 标签。
3. **逐 run 产物**：`run_manifest.json`、日志与完整 traceback（如失败）、`metrics.json`、逐 epoch CSV/JSON、预测和效用明细、best checkpoint（若该实验有训练）、`COMPLETE`。best accuracy 的 Macro-F1、BA、混淆矩阵必须来自同一个 checkpoint；并列 best 取最早 epoch。
4. **结果表**：每个 outer target/fold 的 Acc、Macro-F1、BA、CE、utility、action rate、no-op rate、negative utility rate、错误翻转四格表；平均值和标准差定义；每被试配对差。不能只交最终均值、截图或经过手工筛选的 folds。
5. **失败与偏差**：明确列出缺失 cell、OOM/NaN/中断、配置偏离和任何代码改动；失败不得写为0分。完整矩阵未完成时标记 partial，不得写 complete。
6. **复现说明**：一条从仓库根目录可执行的命令、输入文件清单、预期输出路径、运行前检查、如何复算本组 CSV/图表。

### 每位组员的专属交付

| 负责人 | 专属目录/核心结果 | 完成判据 |
|---|---|---|
| A | `results/c1_c2_temporal/<hash>/A_stage_a/`：B0/B1、C1冻结checkpoint诊断、oracle/no-op feasibility、预处理时间审计 | split/action manifest 冻结；最终融合 head-off/head-on 诊断可从逐样本 logits 复算；oracle/accuracy oracle 分开；结论只用于诊断，不用目标标签选超参。 |
| B | `results/c1_c2_temporal/<hash>/B_oof_u0/`：真实 OOF teacher、evidence cache、U0 MSE/Huber 基线 | pilot 3 targets 的 OOF train/heldout 集合经独立检查无交叉；每条证据能追溯 teacher/checkpoint/data hash；source-dev 和 outer target 均不进入 router training；U0 结果可复算。 |
| C | `results/c1_c2_temporal/<hash>/C_temporal/`：U1、U2、U2-current 的同缓存对照 | 输入证据缓存和动作定义与 B 完全相同；history 严格过去且 trial reset；报告 source-dev 选择和 outer-target 结果；future isolation、trial reset、chunk consistency 测试全通过。 |

### 负责人总验收包

负责人收到 A/B/C 三包后交付 `results/c1_c2_temporal/<hash>/final/`，包括完整性检查清单、跨包 provenance/hash 对照、缺失/失败 run 表、机器生成的主表和 paired-difference 表、图表源数据、分析脚本与执行日志。只有完整 Stage D 矩阵通过后，才将文档状态改为完整实验结果；否则标题和表格均标注 pilot/partial。

图表：
1. C1六类别对几何变化与实际混淆改善对照；相同checkpoint head-off/head-on与端到端E0/E2分开。
2. C2 source×pair预计/实际效用图、校准图、拒绝/执行比例。
3. 一个预先指定trial的当前margin、历史margin、动作、真实标签时间线；同时包含成功和失败案例。trial标签只作标签线，不声称真实逐秒情绪轨迹。
4. 各方法每被试best accuracy配对差及最终错→对/对→错。

论文Methods应明确：输入信息、候选动作、utility监督、OOF训练、因果历史、no-op、best报告协议。Theory只写上述假设下能证明的性质。Results每条结论对应实际表格，不把未做对照写成支持证据。Discussion区分任务内判别解释、泛化证据和神经生理解释。

建议贡献表述（计划语气，结果成立后再改为完成时）：

Contribution 1 — Context-Adaptive Affective Metric：在固定情绪语义锚点下，学习EEG context条件化的低秩有界判别几何，并通过类别对几何、分类margin和受控基底对照检验其判别作用。历史context为待验证扩展，不预先列为已证实贡献。

Contribution 2 — History-Informed Pairwise Transfer Utility Learning：利用源域留出被试上的候选动作损失改善，学习当前和历史证据条件下的迁移效用，在样本、源与情绪类别对层面选择单次修正或保持参考预测，并检验该效用选择是否优于置信度、距离及普通时间平滑。

最终主张是“可追溯的判别收益选择”。不使用“恢复真实情绪状态”“证明因果脑连接”“保证准确率提升”“首次使用时间/语义/度量”等超出证据的表述。
