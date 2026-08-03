# GeoSem-STDA Current Ablation Brief

This document summarizes the current new model status so that collaborators or another AI agent can continue ablation experiments from the same protocol.

## Project

- Local project path used for the current run: `D:\大学\脑机接口\SGDA_py3.11`
- Repository target: `https://github.com/ococ7708/SGDA.git`
- Dataset currently tested: DEAP
- Task currently tested: valence binary classification
- Current new model name: GeoSem-STDA sparse reliability

## Files Added for the New Model

- `models/geosem_stda.py`
- `experiments/deap/crossSubject_geosem_stda_deap.py`
- `experiments/deap/run_geosem_stda_deap_5target_full.ps1`

## Model Structure

The current GeoSem-STDA model contains:

1. Geometry module
   - SPD covariance construction.
   - Shrinkage covariance for numerical stability.
   - Matrix logarithm / Log-Euclidean mapping.
   - Tangent-space deviation from a geometric reference.
   - Geometry-guided adjacency construction.

2. Dynamic graph module
   - Combines geometry adjacency `A_geo` and learnable adjacency `A_learn`.
   - Uses a learnable gate to fuse geometric and data-driven graph structure.

3. Spatio-temporal encoder
   - Encodes DEAP EEG samples shaped as `[sample_length, channels, num_freq_bands]`.
   - Current DEAP shape assumption: `[9, 32, 5]`.

4. Semantic prototype space
   - Semantic space is still present.
   - Uses CLIP text vectors as class prototypes.
   - Projects EEG features to semantic prototype space.
   - Uses prototype classification / contrastive objective.

5. Multi-source adapters
   - Multi-source branches are still present.
   - Full 31-source final training is avoided by target-aware source reliability selection.
   - Each target starts with 31 candidate source subjects and keeps 6 selected source subjects for final training.

6. Target-aware sparse source reliability
   - Warm-up model is trained with all candidate sources.
   - Reliability score uses semantic marginal distance, semantic conditional distance, and source classification proxy.
   - Selected source losses and MMD alignment are weighted by reliability weights.

7. Losses
   - Source prototype classification loss.
   - Source-target MMD loss.
   - Optional source reliability weighting.
   - Class-weighted source classification loss.

## Full Current Protocol

- Dataset: DEAP
- Label: valence
- Classification: binary
- Fixed target subjects: `3, 14, 20, 23, 32`
- Candidate source subjects per target: 31
- Final selected source subjects per target: 6
- Full data: yes
- Full training protocol: yes
- Random seed: 42
- Target labels: used for evaluation according to the existing senior-code protocol
- Summary metric: best target epoch accuracy, following the original script convention
- Per-epoch accuracy: logged in `epoch_log.csv`
- Final epoch accuracy: diagnostic only

Important protocol note: Do not change the existing best-target summary protocol unless the experiment owner explicitly asks for a strict unsupervised domain adaptation variant.

## Hyperparameters

```text
dataset = deap
label_used = valence
LM = clip
num_classes = 2
channels = 32
num_freq_bands = 5
text_dim = 512

epochs = 200
batch_size = 64
lr = 0.001
weight_decay = 0.0001
seed = 42

sample_length = 9
stride = 3

lambda_max = 0.3
proto_tau = 0.07
fusion_tau = 0.5
topk = 6
shrinkage = 0.1
spd_eps = 0.00001
geometry_batch_size = 256

st_dim = 128
graph_dim = 64
adapter_bottleneck = 32
heads = 4
dropout = 0.3

subject_zscore = true
class_weighted_loss = true
sample_subset = stratified

source_selection = sparse_reliability
reliability_warmup_epochs = 5
sparse_rho = 0.85
sparse_k_max = 6
source_weight_tau = 0.5

rel_marg_weight = 1.0
rel_cond_weight = 1.0
rel_val_weight = 0.2

eval_interval = 1
log_interval = 1
```

## Exact Current Launcher

PowerShell:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_full.ps1
```

Equivalent command:

```bash
python experiments/deap/crossSubject_geosem_stda_deap.py \
  --target_subject_ids 3 14 20 23 32 \
  --source_selection sparse_reliability \
  --epochs 200 \
  --batch_size 64 \
  --lr 1e-3 \
  --reliability_warmup_epochs 5 \
  --sparse_k_max 6 \
  --sparse_rho 0.85 \
  --eval_interval 1 \
  --log_interval 1 \
  --lambda_max 0.3 \
  --proto_tau 0.07 \
  --fusion_tau 0.5 \
  --topk 6 \
  --shrinkage 0.1 \
  --spd_eps 1e-5 \
  --geometry_batch_size 256 \
  --st_dim 128 \
  --graph_dim 64 \
  --adapter_bottleneck 32 \
  --heads 4 \
  --dropout 0.3 \
  --weight_decay 1e-4 \
  --sample_subset stratified
```

## Current Experiment Results

Run directory:

```text
results/results_deap_geosem_stda/runs/ep200_bs64_lr0p001_lmda0p3_tau0p07_topk6_seed42_20260802_234859
```

Main output files generated locally:

- `target_subject_results.csv`
- `epoch_log.csv`
- `run_config.json`
- `subject_results_deap_geosem_stda_valence_ep200_bs64.csv`

Per-target result:

| Target Subject | Candidate Sources | Final Sources | Acc | Macro-F1 | Micro-F1 |
|---:|---:|---:|---:|---:|---:|
| S3 | 31 | 6 | 66.67 | 66.52 | 66.67 |
| S14 | 31 | 6 | 71.11 | 71.00 | 71.11 |
| S20 | 31 | 6 | 57.50 | 57.26 | 57.50 |
| S23 | 31 | 6 | 54.72 | 51.77 | 54.72 |
| S32 | 31 | 6 | 60.00 | 59.90 | 60.00 |

Mean result:

```text
Accuracy = 62.00 +/- 6.75
Macro-F1 = 61.29 +/- 7.59
Micro-F1 = 62.00 +/- 6.75
```

## Same-Target Comparison With Existing Experiments

The following comparison uses the same five target subjects: `3, 14, 20, 23, 32`.

| Method | Covered Targets | Acc | Macro-F1 |
|---|---:|---:|---:|
| Original SGDA | 5/5 | 67.50 +/- 9.64 | 65.34 +/- 10.18 |
| Riemann average bs8 | 5/5 | 67.00 +/- 7.16 | 64.28 +/- 8.66 |
| Riemann adaptive bs8 | 5/5 | 66.50 +/- 7.20 | 63.80 +/- 8.85 |
| Tangent-aux bs64 | 5/5 | 64.66 +/- 7.20 | 55.76 +/- 15.54 |
| DE-Riemann average | 5/5 | 62.41 +/- 5.06 | 56.00 +/- 7.32 |
| DE-Riemann adaptive | 5/5 | 62.06 +/- 4.63 | 55.71 +/- 7.52 |
| Current GeoSem-STDA sparse | 5/5 | 62.00 +/- 6.75 | 61.29 +/- 7.59 |

## Final-Epoch Diagnostic

The summarized result follows the original best-target epoch protocol. The final epoch values show some late-training degradation:

| Target | Best Epoch | Best Acc | Epoch 200 Acc | Gap |
|---:|---:|---:|---:|---:|
| S3 | 24 | 66.67 | 57.50 | -9.17 |
| S14 | 16 | 71.11 | 65.56 | -5.56 |
| S20 | 23 | 57.50 | 47.50 | -10.00 |
| S23 | 134 | 54.72 | 50.00 | -4.72 |
| S32 | 107 | 60.00 | 55.00 | -5.00 |

Interpretation: the model can run and reduce computation, but late training suggests overfitting or accumulated negative transfer.

## Selected Sources

Each target used 31 candidate source subjects and selected 6 for final training:

| Target | Selected Sources |
|---:|---|
| S3 | S9, S11, S12, S13, S21, S27 |
| S14 | S3, S6, S8, S10, S30, S31 |
| S20 | S1, S6, S12, S16, S28, S32 |
| S23 | S6, S13, S17, S20, S28, S31 |
| S32 | S8, S15, S20, S25, S28, S30 |

Observed issue: selected-source weights are close to uniform, usually around `0.16-0.17`, and the source accuracy proxy saturated at `1.0`. This means source pruning reduces computation, but the reliability score still needs better discrimination.

## Feasibility Judgment

Current conclusion:

- Feasible as a prototype and ablation base.
- Useful for reducing multi-source computation from 31 final source branches to 6.
- Macro-F1 is competitive with some Riemann variants.
- Accuracy is not yet better than original SGDA.
- Late-epoch degradation indicates overfitting or negative transfer.

Recommended wording:

```text
GeoSem-STDA sparse reliability can run under the complete DEAP 5-target protocol and reduces final multi-source training from 31 candidate source domains to 6 selected source domains. The model shows useful semantic-space behavior through improved Macro-F1 compared with several Riemann variants, but its current accuracy is lower than original SGDA and final-epoch diagnostics show late-training degradation. Therefore, the current model should be treated as a feasible ablation baseline rather than the final superior model.
```

## Recommended Next Ablations

1. Source selection ablation
   - `source_selection = none`
   - `source_selection = fixed_top_m`
   - `source_selection = sparse_reliability`

2. Reliability sharpness
   - `source_weight_tau = 0.1`
   - `source_weight_tau = 0.2`
   - `source_weight_tau = 0.5`

3. Sparsity strength
   - `sparse_rho = 0.70`
   - `sparse_rho = 0.80`
   - `sparse_rho = 0.85`
   - `sparse_k_max = 4, 6, 8`

4. Overfitting diagnosis
   - inspect `epoch_log.csv`
   - compare best epoch vs final epoch
   - consider smaller learning rate, stronger dropout, or fewer epochs only as a separate ablation

5. Fair baseline
   - rerun original SGDA on the same five target subjects with matching preprocessing where possible
   - keep original senior-code best-target protocol for direct comparison unless explicitly running strict UDA

