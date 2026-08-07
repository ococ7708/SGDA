# Reliability-Aware Semantic Conditional Alignment

This note records the current innovation direction implemented on top of GeoSem-STDA.

## Core Judgment

The current baseline does not fully solve MMD negative transfer. It reduces multi-source risk through semantic prototypes, sparse source reliability, and adaptive source fusion, but its training alignment can still be dominated by marginal MMD:

```text
L = L_proto + lambda(t) * L_marginal_mmd
```

This can over-align target samples in the late stage because global MMD does not know whether a target sample is low-valence or high-valence.

## New Training Objective

The new option is:

```text
--mmd_type sca
```

SCA means Semantic Conditional Alignment:

```text
L = L_proto + lambda(t) * L_SCA
```

For each selected source domain k and class c:

```text
source center:  mean(z_source_k | y=c)
target center:  weighted mean(z_target_k, weight = uncertainty * prototype probability)
```

The target class probability comes from text prototypes:

```text
q_t = softmax(z_target @ text_prototypes / proto_tau)
```

The uncertainty weight uses normalized entropy:

```text
u_t = 1 - H(q_t) / log(C)
```

Low-confidence target samples therefore contribute less to class-conditional alignment.

## Source Reliability

The existing sparse reliability module is retained:

```text
31 candidate source subjects -> selected Top-K reliable sources
```

The selected source weights are reused in SCA, so the model controls:

```text
who to align with: source reliability
what to align: semantic class centers
how strongly to align: lambda schedule
which target samples to trust: entropy uncertainty
```

## Dynamic Marginal-Conditional Balance

Early pseudo labels can be noisy, so SCA does not need to start as fully conditional alignment.

The implemented SCA loss is:

```text
L_SCA = (1 - mu) * L_marginal + mu * L_conditional
```

where `mu` ramps from `sca_mu_start` to `sca_mu_end`:

```text
sca_mu_start = 0.0
sca_mu_end = 1.0
sca_mu_warmup_ratio = 0.5
```

This means early training is safer and later training becomes more class-aware.

## Adaptive Alignment Schedule

The recommended schedule is:

```text
mmd_schedule = warmup_cosine_decay
lambda_max = 0.2
lambda_min = 0.05
mmd_warmup_ratio = 0.2
mmd_hold_ratio = 0.5
```

Meaning:

```text
0%-20%: alignment increases
20%-50%: alignment holds
50%-100%: alignment decays by cosine toward lambda_min
```

This directly targets the observed late-stage degradation.

## Recommended DEAP Run

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_sca_full.ps1
```

This keeps the senior 5-target full protocol:

```text
target_subject_ids = 3, 14, 20, 23, 32
source candidates = 31
source_selection = sparse_reliability
epochs = 200
batch_size = 64
lr = 0.001
```

## Dataset Suitability

The SCA idea is most suitable for datasets with stable categorical emotion labels, because it relies on class-conditional semantic centers.

Expected suitability:

```text
SEED-IV / SEED-V: strongest fit
SEED: strong fit
DEAP: useful but harder to improve
DREAMER: useful but likely unstable due to smaller subject count and valence/arousal-style labels
```

Reason:

- SEED-family datasets have clearer discrete emotion categories, so text prototypes and class-conditional alignment have stronger semantic anchors.
- DEAP valence binary labels are noisier and subject self-rating based, so the target pseudo labels are less stable.
- DREAMER is smaller and also uses dimensional emotion ratings, so it is more suitable as a secondary validation dataset than the first proof dataset.

For the current codebase, DEAP remains the easiest implementation target because the GeoSem-STDA script is already complete. For stronger final results, the next dataset adaptation should prioritize SEED-IV or SEED-V.
