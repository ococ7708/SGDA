# MMD Experiment Plan

This plan keeps the senior target-best protocol unchanged. It only changes the training MMD term and its schedule.

## Why MMD Is the Priority

The current GeoSem-STDA sparse reliability baseline reaches its best accuracy early and then degrades toward the final epoch. RSG-CutMix v1 did not improve the result, which suggests that the main problem is probably not sample quantity but unstable or excessive domain alignment.

Current baseline:

```text
L = L_proto + lambda_mmd * L_mmd
lambda_mmd -> 0.3
MMD = marginal mean alignment
```

Potential issue:

```text
late-stage marginal MMD may pull target embeddings toward unreliable or class-mismatched source distributions
```

## Step 1: Lambda Ablation

Script:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_mmd_lambda_ablation.ps1
```

Runs:

```text
lambda_max = 0.0
lambda_max = 0.05
lambda_max = 0.1
lambda_max = 0.2
lambda_max = 0.3
```

Fixed settings:

```text
mmd_type = marginal
mmd_schedule = monotonic
target_subject_ids = 3, 14, 20, 23, 32
source_selection = sparse_reliability
epochs = 200
batch_size = 64
lr = 0.001
```

Goal:

```text
check whether stronger MMD causes stronger late-stage degradation
```

Expected interpretation:

- If `0.05` or `0.1` is better than `0.3`, current MMD is too strong.
- If `0.0` is better than all MMD settings, the current alignment direction is harmful.
- If `0.2` is best, MMD helps but should be weaker than the current baseline.

## Step 2: Scheduled MMD

New schedule options:

```text
monotonic: original sigmoid warm-up toward lambda_max
warmup_hold: linear warm-up, then hold
warmup_decay: linear warm-up, hold, then decay to lambda_min
```

Recommended first scheduled run:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_classaware_mmd_full.ps1
```

Settings:

```text
mmd_schedule = warmup_decay
mmd_warmup_ratio = 0.2
mmd_hold_ratio = 0.5
lambda_max = 0.2
lambda_min = 0.05
```

Meaning:

```text
0%-20% training: lambda linearly increases from 0 to 0.2
20%-50% training: lambda stays at 0.2
50%-100% training: lambda linearly decays from 0.2 to 0.05
```

## Step 3: Class-Aware MMD

New option:

```text
--mmd_type class_aware
```

Source class centers use true source labels. Target class centers use soft pseudo labels from text prototypes:

```text
q_t = softmax(z_target @ text_prototypes / proto_tau)
```

The class-aware MMD aligns class centers rather than only global means:

```text
L_mmd = sum_c q_c * || mean(z_source^c) - mean(z_target^c) ||^2
```

This is designed to avoid class-mismatched marginal alignment.

## Step 4: Confidence-Gated MMD

New option:

```text
--mmd_confidence_gate none|soft|threshold
```

Modes:

```text
none: use all target soft pseudo labels
soft: weight target pseudo labels by max softmax confidence
threshold: use target samples only when max confidence >= threshold
```

Recommended first setting:

```text
mmd_confidence_gate = soft
```

This is less brittle than a hard threshold when target pseudo labels are still noisy.

## Code Safety

Defaults preserve the old baseline:

```text
mmd_type = marginal
mmd_schedule = monotonic
mmd_confidence_gate = none
lambda_min = 0.0
```

Therefore, existing scripts keep their original behavior unless these new flags are explicitly set.

