# SGDA / GeoSem-STDA for EEG Emotion Recognition

This repository contains the SGDA-based cross-subject EEG emotion recognition code and the current **GeoSem-STDA** model.

The current priority is **best target accuracy** under the senior SGDA protocol. Target labels are used only for epoch-wise evaluation, and each subject result reports the best target epoch.

## Current Complete Model

The unified model is implemented in:

```text
models/geosem_stda.py
experiments/crossSubject_geosem_stda_sgda.py
```

The default launchers use the latest stable GeoSem-STDA configuration:

- SPD covariance geometry with shrinkage.
- Log-Euclidean tangent deviation features.
- Geometry-guided dynamic graph convolution.
- Multi-scale temporal encoder and attention pooling.
- CLIP semantic text prototypes.
- Multi-source source-domain adapters.
- Target-aware sparse reliability source selection.
- Reliability-guided semantic-geometric conditional alignment (`--mmd_type resgca`).

DEAP also keeps a HUT/RCUOT launcher because the recent DEAP trial used a lighter unbalanced-transport alignment. This is switchable and documented below.

## Why This Direction

Recent EEG emotion recognition work still points to the same useful directions for this project:

- multi-source domain adaptation and source selection to reduce negative transfer;
- graph/spatial modeling for EEG channels;
- class-aware or pseudo-label-aware alignment instead of only marginal MMD.

Representative recent references:

- [MSGDAN: Multi-source Selective Graph Domain Adaptation Network for cross-subject EEG emotion recognition](https://doi.org/10.1016/j.neunet.2024.106742), Neural Networks 2024.
- [Spectral-Spatial Attention Alignment for Multi-Source Domain Adaptation in EEG-Based Emotion Recognition](https://ieeexplore.ieee.org/document/10509712/), IEEE Transactions on Affective Computing 2024.
- [DAPLP: Unsupervised Domain Adaptation With Pseudo-Label Propagation for Cross-Domain EEG Emotion Recognition](https://ieeexplore.ieee.org/document/10944516/), IEEE Transactions on Instrumentation and Measurement 2025.
- [MS-DCDA: Multi-Source EEG Emotion Recognition via Dynamic Contrastive Domain Adaptation](https://arxiv.org/abs/2408.10235).

Our current model follows these ideas but keeps a different implementation: semantic prototypes + SPD geometry + reliability-guided conditional alignment.

## Data Paths

Datasets are not included in this repository. Set local paths in:

```text
data_utils/constants/path_mapper.py
```

Required keys:

```python
path_mapper = {
    "deap": ".../DEAP/data_preprocessed_python/data_preprocessed_python",
    "seed_de_lds": ".../SEED/",
    "seediv_de_lds": ".../SEED_IV/",
    "seedv_de_lds": ".../SEED_V/",
    "dreamer": ".../DREAMER/DE_processed_1s.npy",
}
```

CLIP text encoder path is configured in:

```text
data_utils/text_to_vector.py
```

Current local default:

```text
D:/大学/脑机接口/local_clip_model
```

## Environment

Use the conda environment with PyTorch:

```powershell
conda activate sgda_py311
pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching your GPU and CUDA version.

## SGDA Protocol

All launchers below follow the same protocol:

- Cross-subject leave-one-subject-out within each evaluated session.
- Target subject data participates in unsupervised adaptation without labels.
- Target labels are used only for evaluation after each epoch.
- The main reported metric is `best_acc`.
- `macro_f1` and `micro_f1` are saved at the same best epoch.
- Full training uses 200 epochs unless explicitly changed.

Output is written to:

```text
results/results_<dataset>_geosem_stda/runs/<run_id>/
```

Important files:

```text
epoch_log.csv
subject_results_<dataset>_geosem_stda.csv
run_config.json
```

## Direct Full Runs

Run from the repository root after activating conda.

### DEAP

```powershell
.\experiments\deap\run_geosem_stda_deap_sgda_full.ps1
```

Default DEAP setting:

```text
task = valence binary classification
sample_length = 9
stride = 3
epochs = 200
batch_size = 64
mmd_type = hut
lambda_max = 0.02
final selected sources = Top-6 reliability sources
```

### SEED

```powershell
.\experiments\seed\run_geosem_stda_seed_sgda_full.ps1
```

Default SEED setting:

```text
sessions = 1, 2, 3
classes = 3
sample_length = 3
stride = 1
epochs = 200
batch_size = 128
mmd_type = resgca
final selected sources = Top-6 reliability sources
```

### SEED-IV

```powershell
.\experiments\seediv\run_geosem_stda_seediv_sgda_full.ps1
```

Default SEED-IV setting:

```text
sessions = 1, 2, 3
classes = 4
sample_length = 3
stride = 1
epochs = 200
batch_size = 64
mmd_type = resgca
final selected sources = Top-6 reliability sources
```

### SEED-V

```powershell
.\experiments\seedv\run_geosem_stda_seedv_sgda_full.ps1
```

Default SEED-V setting:

```text
sessions = 1, 2, 3
classes = 5
sample_length = 3
stride = 1
epochs = 200
batch_size = 64
mmd_type = resgca
final selected sources = Top-6 reliability sources
```

The unified script reshapes SEED-V flat features from `[L, 310]` to `[L, 62, 5]` before model input.

### DREAMER

```powershell
.\experiments\dreamer\run_geosem_stda_dreamer_sgda_full.ps1
```

Default DREAMER setting:

```text
session = 1
task = valence binary classification
sample_length = 3
stride = 1
epochs = 200
batch_size = 64
mmd_type = resgca
dreamer_ea = true
final selected sources = Top-6 reliability sources
```

For arousal:

```powershell
python experiments\crossSubject_geosem_stda_sgda.py --dataset_name dreamer --dreamer_labeltype aro
```

## Smoke Test

Before running all subjects, run one subject for a few epochs:

```powershell
python experiments\crossSubject_geosem_stda_sgda.py `
  --dataset_name seediv `
  --session_ids 1 `
  --target_subject_ids 1 `
  --epochs 3 `
  --reliability_warmup_epochs 1 `
  --sparse_k_max 3
```

If this succeeds, run the corresponding full `.ps1` script.

## Useful Switches

Use the default `.ps1` scripts for the main experiment. Change these only for ablation or comparison.

| Purpose | Parameter |
|---|---|
| Run fixed target subjects | `--target_subject_ids 1 4 7 10 13` |
| Run one session | `--session_ids 1` |
| Random target subset | `--random_target_count 5 --target_seed 42` |
| Disable source selection | `--source_selection none` |
| Use class-aware MMD | `--mmd_type class_aware` |
| Use ReSGCA | `--mmd_type resgca` |
| Use HUT/RCUOT | `--mmd_type hut --uot_epsilon 0.10 --uot_tau_t 0.7 --uot_n_iter 12 --no_hut_agreement_mass` |
| Evaluate by feature centroids | `--eval_classifier senior_feature` |
| Use reliability fusion in evaluation | `--reliability_fusion` |

## Team Requirements

Group members should:

1. Use the latest `main` branch.
2. Activate the correct conda environment.
3. Check `path_mapper.py` before running.
4. Run one smoke test first.
5. For formal results, run the dataset `.ps1` script without changing hyperparameters.
6. Report `best_acc`, `macro_f1`, `micro_f1`, `best_epoch`, and the path to `run_config.json`.
7. Do not compare results from different `sample_length`, `stride`, source count, or epoch settings as the same protocol.

## Current DEAP Partial Progress

The latest DEAP 15-target protocol has completed these targets:

| Target | best_acc | best_epoch | final_acc |
|---:|---:|---:|---:|
| 2 | 44.72% | 51 | 35.28% |
| 3 | 70.28% | 166 | 64.72% |
| 6 | 60.42% | 102 | 52.92% |

Current completed-target mean best accuracy is about 58.47%.
