# DEAP MMD Lambda Ablation: lambda_max = 0.0

Run:

```text
results/results_deap_geosem_stda/runs/ep200_bs64_lr0p001_lmda0p0_tau0p07_topk6_seed42_20260807_134405
```

Protocol:

```text
dataset = DEAP
target_subject_ids = 3, 14, 20, 23, 32
source candidates per target = 31
source_selection = sparse_reliability
final source count = 6
epochs = 200
batch_size = 64
lr = 0.001
lambda_max = 0.0
mmd_type = marginal
mmd_schedule = monotonic
seed = 42
```

The senior target-best reporting protocol is preserved.

## Target-Best Results

| Target | Best Epoch | Best Acc | Best Macro-F1 |
|---:|---:|---:|---:|
| S3 | 5 | 60.69 | 51.09 |
| S14 | 1 | 69.72 | 68.97 |
| S20 | 29 | 60.28 | 59.14 |
| S23 | 117 | 60.00 | 37.50 |
| S32 | 1 | 64.03 | 63.67 |

Summary:

```text
Best Acc = 62.94 +/- 4.12
Best Macro-F1 = 56.08
```

## Final-Epoch Diagnostic

| Target | Final Acc | Final Macro-F1 | Final - Best Acc |
|---:|---:|---:|---:|
| S3 | 56.25 | 51.61 | -4.44 |
| S14 | 56.67 | 54.91 | -13.06 |
| S20 | 43.89 | 43.85 | -16.39 |
| S23 | 55.69 | 37.95 | -4.31 |
| S32 | 28.89 | 25.84 | -35.14 |

Summary:

```text
Final Acc = 48.28 +/- 12.08
Final Macro-F1 = 42.83
```

## Interpretation

This run confirms that simply removing MMD does not solve the late-stage degradation problem. Several targets reach their best performance very early, especially S14 and S32, then degrade substantially by epoch 200.

This suggests the current instability is not only caused by strong marginal MMD. Other factors may also contribute:

```text
source overfitting
prototype CE saturation
source reliability selected too early
target pseudo-label instability
DEAP valence label noise
```

The result supports moving the next main experiment to ReSGCA and SEED-IV, where emotion classes are more semantically stable and class-conditional alignment should be more meaningful.
