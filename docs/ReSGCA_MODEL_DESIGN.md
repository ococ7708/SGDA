# Reliability-Guided Semantic-Geometric Conditional Alignment

This document records the ReSGCA model option implemented on top of GeoSem-STDA.

## Motivation

The current baseline uses a semantic prototype space and sparse source reliability, but the domain alignment term can still behave like global MMD:

```text
source distribution <-> target distribution
```

This can cause negative transfer when the target subject has noisy valence labels or when source and target have different class-conditional EEG geometry.

ReSGCA changes the alignment question from:

```text
Are source and target globally similar?
```

to:

```text
Which source, which class, which target samples, and which geometric structure should be trusted for transfer?
```

## Transfer Trust

For source domain `k`, target sample `j`, and class `c`, ReSGCA uses:

```text
trust(k,c,j) = source_reliability(k)
             * semantic_membership(j,c)
             * semantic_confidence(j)
             * geometric_consistency(k,c,j)
```

In code:

```text
semantic_membership = softmax(z_target @ text_prototypes / proto_tau)
semantic_confidence = 1 - entropy(semantic_membership) / log(num_classes)
geometric_consistency = exp(- geo_weight * d_geo / geo_tau)
```

The geometry distance is measured in the current Log-Euclidean tangent representation `R`, which is already computed by GeoSem-STDA:

```text
d_geo = mean((R_target - mean(R_source | y=c))^2)
```

## Objective

ReSGCA keeps a clean two-term objective:

```text
L = L_proto + lambda(t) * L_ReSGCA
```

The conditional alignment term aligns source class semantic centers with target class centers weighted by transfer trust.

To avoid early pseudo-label noise, ReSGCA keeps the SCA marginal-to-conditional ramp:

```text
L_ReSGCA = (1 - mu) * L_marginal + mu * L_trusted_conditional
```

Recommended settings:

```text
mmd_type = resgca
mmd_schedule = warmup_cosine_decay
mmd_confidence_gate = entropy
lambda_max = 0.2
lambda_min = 0.05
sca_mu_start = 0.0
sca_mu_end = 1.0
sca_mu_warmup_ratio = 0.5
resgca_geo_tau = 1.0
resgca_geo_weight = 1.0
```

## Recommended DEAP Run

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_resgca_full.ps1
```

This preserves the senior full protocol:

```text
target_subject_ids = 3, 14, 20, 23, 32
source candidates per target = 31
source_selection = sparse_reliability
epochs = 200
batch_size = 64
lr = 0.001
```

## Ablation Chain

Recommended ablation order:

```text
M0: marginal MMD baseline
M1: class_aware MMD
M2: SCA with entropy uncertainty
M3: SCA with sparse source reliability
M4: ReSGCA with geometry trust
M5: ReSGCA + warmup_cosine_decay
```

Use `--resgca_geo_weight 0.0` to disable geometry trust while keeping the rest of ReSGCA unchanged.

## Dataset Suitability

Expected fit:

```text
SEED-IV / SEED-V > SEED > DEAP > DREAMER
```

Reason:

- ReSGCA is class-conditional and benefits from stable discrete emotion labels.
- SEED-family datasets have clearer semantic classes, so text prototypes and conditional alignment should be stronger.
- DEAP valence is binary and self-rating based, so target pseudo labels can be less stable.
- DREAMER is smaller and dimensional-label based, which makes geometry-conditional pseudo-label alignment more fragile.

DEAP remains the fastest implementation target because the current GeoSem-STDA pipeline is already complete. For a stronger final paper result, SEED-IV or SEED-V should be prioritized once the DEAP feasibility run is understood.

## Novelty Boundary

Do not claim that this is the first EEG emotion recognition method combining SPD geometry and prototypes. Prior SPD prototype domain adaptation work already exists.

The intended contribution is narrower and safer:

```text
language semantic prototypes
+ source reliability
+ target uncertainty
+ Log-Euclidean geometry trust
=> selective, interpretable conditional transfer
```
