# C1 E2 metric learning-rate focused screen

## Method constraints and theoretical properties

This is a narrowly scoped optimizer intervention on the existing E2 metric mechanism. The eventual experiment is three separate SEED-IV session-wise LOSO studies; sessions are never mixed in training. Within each session all 15 targets and seeds 42, 43, 44 are covered. DE-LDS preprocessing, subject normalization, backbone, adapter, semantic prototypes, fusion rule, rank, temperature, regularizers, data split, 200-epoch backbone budget, batch size 64, and accuracy-based best-checkpoint selection remain unchanged. The only intended model-training change is a distinct optimizer group for the metric mechanism, with `metric_lr = learning_rate * metric_lr_multiplier`; the backbone remains at the original learning rate. The multiplier is part of the resolved/effective configuration and therefore changes the configuration hash.

The mechanism matrix is parameterized so its spectral norm is bounded by `gamma`. For the implemented metric form `M = I + B H B^T`, with orthonormal `B`, symmetric `H`, and `||H||_2 <= gamma < 1`, all eigenvalues of `M` lie in `[1-gamma, 1+gamma]`. Thus M is positive definite. This is a mathematical property of the parameterization, not an empirical accuracy claim. Setting the final metric-generator projection to zero gives `H=0` and recovers the base cosine metric score; the test checks numerical equivalence. Context shuffling is an intervention diagnostic: it tests dependence on context but does not imply causal validity or better target generalization.

No new network module is introduced. Fusion weights, rank and temperature are fixed. The existing E2 target-best rule is retained for comparability; because that rule selects using the target evaluation accuracy, these scores are exploratory and optimistic, not confirmatory paper estimates.

## Experimental settings

The formal A/B JSON configurations cover 3 sessions × 15 targets × 3 seeds = 135 paired runs per phase, using original E2 as the reference (parent configuration hash `592351bfb649e601`). A local preflight found only 90/135 E2 parent folds complete; all 45 Session 3 E2 parent checkpoints/metrics are missing. The full A/B experiment is blocked until those exact parents are completed and audited. The current engineering smoke is deliberately much smaller: target 6, seed 42, 2 epochs in each of Sessions 1, 2 and 3. Target 6 is retained because prior Session 1 evidence indicated a difficult fold; smoke performance is not scientific evidence and does not narrow the eventual all-target scope.

| Phase | Metric LR multiplier | Gamma | Backbone LR | Metric LR | Epochs / batch | Purpose |
|---|---:|---:|---:|---:|---:|---|
| A | 3.0 | 0.50 | 0.001 | 0.003 | 200 / 64 | Initial screen |
| B | 3.0 | 0.25 | 0.001 | 0.003 | 200 / 64 | Gamma-only follow-up, gated on A |

Both phases use the same full-session scope, with A setting multiplier 3.0 / gamma 0.5 and B holding multiplier 3.0 while changing gamma to 0.25. The launcher checks all 135 parent E2 folds before starting. B starts only after all 135 A folds finish and the predeclared gate passes: average over seeds within each subject-session cell; require overall accuracy and Macro-F1 deltas > 0, at least 23/45 cells positive for each, at least 2/3 session means positive for each, and nonnegative balanced-accuracy delta in every session. This is a descriptive screening rule, not an inferential test.

The previous target-focused A attempt was interrupted after seed 42 completed and seed 43 had begun. Seed 42 itself is a complete exploratory run under the earlier config hash: S1/T6 accuracy 0.3861 at best epoch 157, versus baseline E2 accuracy 0.4807. Seed 43 stopped at saved epoch 71 (best-so-far 0.4421 at epoch 49), and seed 44 was never launched. The single complete seed may be reported as a one-fold exploratory result, but it is excluded from the new all-session matrix and cannot satisfy its gate; the incomplete seed 43 is not a completed result. The new all-session configurations have distinct effective hashes. No B run has been performed. Per epoch, the code records actual weighted CE, regularization, objective, metric-head gradient magnitude, H spectral norm, and `||H||_2/gamma`. Evaluation additionally records final fused flips, corrected/harmed samples and net correction rate.

Checkpoint evaluation supports the trained metric, `H=0`, and within-subject context permutation. In the permutation mode only the metric-generator context input is shuffled with an explicit random seed; EEG classification embeddings and fusion weights are held fixed. Paired tabular outputs, class-pair confusion changes, and two mechanism figures are written by `analysis/analyze_c1_metric_lr_screen.py` when the required runs are complete.

## Results and limitations

### Measured results

Only smoke tests are in scope now. The C1 E2 smoke completed for Session 1, 2 and 3 at target 6 / seed 42 (two epochs each). Effective hash: `fcb505a9e03ac832`; both optimizer groups were recorded as backbone LR `0.001` and mechanism LR `0.003`. Smoke accuracies were 0.2500, 0.6250 and 0.2500, respectively; these tiny smoke subsets are strictly engineering checks, not comparable performance estimates. H spectral maxima were below the configured `gamma=0.5` in all three smoke runs.

The new all-session A/B matrix is not run. The earlier complete seed42 result is retained as a single-fold exploratory observation; the partial seed43 checkpoint is excluded. No full A/B performance claims are available. The phase gate and paired analysis outputs must be generated only after the separately scheduled full experiment has completed.

### Interpretation boundaries

- Established mathematically: the stated spectral constraint yields a positive-definite metric under the orthonormal-basis assumptions; zero H recovers the base metric score.
- Measured only after completed runs: paired score changes, final-fusion corrections, gradient/norm behavior, and context-shuffle sensitivity.
- Hypotheses, not facts: a larger mechanism learning rate may help the mechanism escape near-zero initialization; lowering gamma may reduce unstable distortions. Either can fail or trade off accuracy and correction/harm rates.

For any future full run, seeds repeat each subject/session condition; they do not create additional independent participants. Statistical inference should cluster at the subject level and report sessions separately as well as the predeclared aggregate. The unchanged target-best epoch rule uses target labels, so its scores remain exploratory/optimistic. SEED Session 2 and SEED-V Session 1 smoke checks are separate datasets and must not be pooled with this SEED-IV C1 experiment.
