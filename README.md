# SGDA / GeoSem-STDA for EEG Domain Adaptation

This repository contains the SGDA-based EEG cross-subject domain adaptation codebase and the current GeoSem-STDA prototype used for DEAP feasibility and ablation experiments.

The current focus is DEAP valence binary classification under a subject-independent domain adaptation protocol.

## Current New Model

The new model is tentatively named **GeoSem-STDA sparse reliability**.

Main files:

- `models/geosem_stda.py`: GeoSem-STDA model components.
- `experiments/deap/crossSubject_geosem_stda_deap.py`: DEAP cross-subject experiment script.
- `experiments/deap/run_geosem_stda_deap_5target_full.ps1`: fixed 5-target full-protocol launcher.
- `experiments/deap/run_geosem_stda_deap_5target_rsg_cutmix_full.ps1`: fixed 5-target RSG-CutMix launcher.
- `experiments/deap/run_geosem_stda_deap_5target_sca_full.ps1`: fixed 5-target SCA launcher.
- `experiments/deap/run_geosem_stda_deap_5target_resgca_full.ps1`: fixed 5-target ReSGCA launcher.
- `docs/GEOSEM_STDA_ABLATION_BRIEF.md`: complete current experiment summary for ablation collaborators.

## Model Overview

GeoSem-STDA currently includes:

- SPD covariance construction with shrinkage.
- Log-Euclidean geometric reference and tangent-space deviation.
- Geometry-guided dynamic graph convolution.
- Temporal/spatio-temporal EEG encoder.
- Semantic prototype space with CLIP text vectors.
- Multi-source adapters.
- Target-aware sparse source reliability selection.
- Prototype classification loss and source-target MMD alignment loss.

The multi-source branch is still present, but the current experiment reduces the final training sources from 31 candidate subjects to 6 selected sources for each target subject.

An optional RSG-CutMix module can be enabled with `--use_rsg_cutmix`. It is disabled by default so the existing baseline protocol remains unchanged.

## Data

Raw and preprocessed EEG datasets are not included in this repository.

The DEAP path is configured in:

```text
data_utils/constants/path_mapper.py
```

Collaborators should edit `path_mapper["deap"]` to point to their local DEAP preprocessed directory.

## Environment

The code was run with a conda environment using PyTorch.

Minimal packages:

```bash
pip install -r requirements.txt
```

If using CUDA, install the PyTorch build matching your local CUDA version from the official PyTorch installation page.

## Reproduce Current 5-Target Experiment

PowerShell:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_full.ps1
```

Equivalent Python call:

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

## Current Protocol

- Dataset: DEAP
- Task: valence binary classification
- Targets: subjects 3, 14, 20, 23, 32
- Candidate sources per target: 31
- Final selected sources per target: 6
- Epochs: 200
- Batch size: 64
- Seed: 42

The current reporting keeps the original senior-code protocol: target evaluation is logged each epoch and the summarized result uses the best target epoch. Final epoch accuracy is kept in `epoch_log.csv` for diagnosis.

## Current Result

For the 5 fixed DEAP target subjects:

```text
Accuracy = 62.00 +/- 6.75
Macro-F1 = 61.29 +/- 7.59
Micro-F1 = 62.00 +/- 6.75
```

Per-target results:

| Target | Candidate Sources | Final Sources | Acc | Macro-F1 | Micro-F1 |
|---:|---:|---:|---:|---:|---:|
| S3 | 31 | 6 | 66.67 | 66.52 | 66.67 |
| S14 | 31 | 6 | 71.11 | 71.00 | 71.11 |
| S20 | 31 | 6 | 57.50 | 57.26 | 57.50 |
| S23 | 31 | 6 | 54.72 | 51.77 | 54.72 |
| S32 | 31 | 6 | 60.00 | 59.90 | 60.00 |

See `docs/GEOSEM_STDA_ABLATION_BRIEF.md` for the complete comparison and recommended next ablations.

## Run RSG-CutMix

PowerShell:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_rsg_cutmix_full.ps1
```

This keeps the same 5-target full protocol and adds the optional reliability-aware semantic-geometric structured EEG CutMix loss.

## Run SCA

PowerShell:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_sca_full.ps1
```

This keeps the same senior 5-target full protocol and replaces marginal MMD with reliability-aware semantic conditional alignment:

```text
mmd_type = sca
mmd_schedule = warmup_cosine_decay
mmd_confidence_gate = entropy
lambda_max = 0.2
lambda_min = 0.05
```

See `docs/SCA_MODEL_DESIGN.md` for the theoretical motivation and dataset suitability judgment.

## Run ReSGCA

PowerShell:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_resgca_full.ps1
```

This keeps the same senior 5-target full protocol and uses reliability-guided semantic-geometric conditional alignment:

```text
mmd_type = resgca
mmd_schedule = warmup_cosine_decay
mmd_confidence_gate = entropy
lambda_max = 0.2
lambda_min = 0.05
resgca_geo_tau = 1.0
resgca_geo_weight = 1.0
```

See `docs/ReSGCA_MODEL_DESIGN.md` for the model formulation, ablation chain, and dataset suitability judgment.

## Run MMD Experiments

The MMD experiments keep the senior target-best reporting protocol unchanged.

Lambda ablation:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_mmd_lambda_ablation.ps1
```

This runs:

```text
lambda_max = 0.0, 0.05, 0.1, 0.2, 0.3
mmd_type = marginal
mmd_schedule = monotonic
```

Recommended class-aware scheduled MMD:

```powershell
.\experiments\deap\run_geosem_stda_deap_5target_classaware_mmd_full.ps1
```

This runs:

```text
mmd_type = class_aware
mmd_schedule = warmup_decay
lambda_max = 0.2
lambda_min = 0.05
mmd_confidence_gate = soft
```
