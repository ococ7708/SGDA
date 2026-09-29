# SGDA / GeoSem-STDA EEG Emotion Recognition

This repository contains the original SGDA implementation, the GeoSem-STDA research model, and the current SEED-IV C1/C2 experiments. Dataset files and pretrained CLIP weights are not included.

## Current priority: C1 frozen; prepare C2 and external-session handoff

C1 optimization is frozen; do not launch further C1 A/B or shared-head exploration. Current main work is to establish whether C2's evidence-driven correction improves the frozen E0 reference, and whether historical evidence adds value beyond smoothing and a capacity-matched control. The C2 and supplementary SEED/SEED-V experiments are not being run by the repository owner in this preparation step; team members receive an explicit data interface, gated run commands, and delivery checklist in [docs/C2_TEAM_EXPERIMENT_HANDOFF.md](docs/C2_TEAM_EXPERIMENT_HANDOFF.md).

The C2 code supports per-machine data roots via `SGDA_SEEDIV_DATA_ROOT`, `SGDA_SEED_DATA_ROOT`, and `SGDA_SEEDV_DATA_ROOT`; optional `SGDA_C1_E0_ROOT` points to the exact E0 parent-results root. `experiments/check_dataset_interface.py` checks expected file layouts without modifying or loading the data. Full C2 remains gated: the currently audited exact E0 parent set has Sessions 1/2 (90/135 cells) but lacks the 45 Session 3 cells; the new three-session pilot must also pass and be owner-accepted before any full runs. The old two-session pilot is not a substitute.

## External single-session smoke: SEED and SEED-V

The separately scoped external checks are SEED Session 2 and SEED-V Session 1. Smoke target 1 / seed 42 / two epochs completed for both; the observed accuracies were 0.4923 and 0.3963, respectively, and are engineering-only. Before running on another machine, point `seed_de_lds` and `seedv_de_lds` in `data_utils/constants/path_mapper.py` to that machine's feature directories. To run the smoke:

```powershell
$env:SGDA_PYTHON = "python"
.\experiments\run_seed_seedv_session1_exploratory.ps1 -Stage Smoke
```

The future 15-target SEED Session 2 and 16-target SEED-V Session 1 LOSO runs are deferred and must be launched separately. Keep datasets, class spaces, metrics, and summaries separate. Details and assignments are in [docs/SEED_SEEDV_SINGLE_SESSION_EXPLORATION.md](docs/SEED_SEEDV_SINGLE_SESSION_EXPLORATION.md).

## C2 experiment scope

The C2 study asks whether held-out source evidence can predict which source/class-pair correction will improve the frozen E0 reference, and whether strictly past margin summaries add value beyond the current window and ordinary probability smoothing. The scope is **SEED-IV Sessions 1, 2, and 3, each evaluated separately**—never pooled for training—with all 15 subjects as outer targets and seeds 42, 43, and 44. This is 135 outer cells per method (not 135 total across all methods). Each C2 outer cell needs four inner teacher fits; those are cached and shared across that cell's C2 variants.

The frozen parent is the C1 E0 configuration `592351bfb649e601`. The new three-session C2 config has a distinct hash and records the exact parent checkpoint SHA-256 per run. The existing parent audit covers 90 E0 artifacts in Sessions 1 and 2; **the 45 Session 3 E0 parent checkpoints are currently missing from the audited path**. Consequently, the three-session parent audit is incomplete, and C2 full execution is blocked until those exact-config E0 runs are completed and audited. The previously passed single-fold pilot belongs to the old Sessions 1–2 configuration and is retained only as historical engineering evidence; it does not validate this expanded config or unlock the new matrix. No full three-session matrix has run.

The new engineering pilot is fixed to Session 3 / target 1 / seed 42, so it can exercise the newly added session path. It cannot run until that exact E0 parent checkpoint is available and audited. The old Session 1 pilot remains inspectable but is not accepted as a substitute. After the Session 3 pilot passes and the owner signs its acceptance record, the full 135-cell-per-method matrix can start—only after all 135 E0 parents are present. Group A/B/C assignments and exact per-member deliverables are in the handoff document above.

## Model at a glance

The SEED-IV reference model is `GeoSemSTDA`: a Strong-DE / channel-selection / graph backbone produces a shared representation; source-specific adapters map it toward frozen CLIP text prototypes for the four emotion classes; the original source branches are fused at the logits level using the E0 distance-based rule. C2 freezes that complete reference and adds a shared router that scores candidate source/class-pair actions from reference and expert logits, source distances, and class-pair prototype features.

For each source and each of the six unordered class pairs, the router predicts the full multiclass cross-entropy improvement

`utility = CE(reference_logits, label) - CE(action_logits, label)`.

At inference, the action with the greatest predicted positive utility is applied, with a no-op when no candidate is predicted to help. The action is a bounded pairwise logit residual; there is no label-based candidate filtering and no product of confidence scores. The first comparison uses current-window features. The history variant additionally reads only prior reference/expert margins within the same trial; state is updated only after the current decision and resets at every trial. A probability EMA is a separate smoothing control.

The internal evidence protocol partitions the 14 outer sources into a deterministic 2-person source-dev set and a 12-person OOF/router pool. Three teachers each train on eight pool subjects and produce evidence for four unseen pool subjects; one 12-source teacher produces source-dev evidence for the two dev subjects. The outer target and dev subjects are excluded from OOF teacher fitting. Router inputs accept the resulting 8-, 12-, or 14-source evidence. All methods use the same split and cached teachers for a given outer fold/seed.

Normalization follows the parent per-subject rule: each evaluated subject contributes only its own unlabeled session moments. Because the existing preprocessing can use an entire session and DE-LDS, this experiment does not claim end-to-end online causality. The router history itself is strictly causal within each trial.

## C2 quick start

Configure local data roots with `SGDA_SEEDIV_DATA_ROOT`, `SGDA_SEED_DATA_ROOT`, and `SGDA_SEEDV_DATA_ROOT`; set the optional `SGDA_C1_E0_ROOT` to mount the exact frozen E0 result directory. Configure the CLIP model path in `data_utils/text_to_vector.py`. Use the locally prepared Python 3.11/PyTorch environment with a matching CUDA build; this checkout does not include a portable conda environment export.

From the repository root in PowerShell:

```powershell
$PY = "python"
& $PY experiments\seediv_c2_experiment.py audit
& $PY experiments\seediv_c2_experiment.py plan
& $PY tests\test_seediv_c2_pipeline.py
```

These commands recheck parent artifacts (and report any missing Session 3 cells), print the plan, and run synthetic tests. The old Session 1 pilot used target 1, seed 42, and B0, B1, C2-current, and C2-history; it took about 33 minutes on an NVIDIA GeForce RTX 5060 Laptop GPU but is bound to the old two-session config. The new pilot uses Session 3 / target 1 / seed 42 and must be run only after its E0 parent exists:

```powershell
& $PY experiments\seediv_c2_experiment.py pilot --execute --device cuda:0
```

Inspect `artifacts/seediv_c2_s1s2s3_parent_audit.json`, the new report at `artifacts/seediv_c2_s1s2s3_pilot_report.json`, and outputs under `results/seediv_c2_e0_s1s2s3/<config-hash>/<implementation-id>/`. The old independent pilot audit script is tied to the previous two-session pilot and is not a verifier for the new run. Do not launch full execution until all 135 parent E0 cells pass audit and the project owner reviews the new pilot and records every checklist item in `artifacts/seediv_c2_s1s2s3_pilot_acceptance.json` using [`docs/templates/seediv_c2_pilot_acceptance_TEMPLATE.json`](docs/templates/seediv_c2_pilot_acceptance_TEMPLATE.json). The record must match the exact pilot-report hash and current code/config fingerprints. The runner checks these gates before full execution.

The other planned contrasts are C2-current-EMA, C2-current-matched (same router capacity with the history slot replaced by a fixed current-feature mapping), and C2-current-Huber. The initial pilot is deliberately limited to the core path; complete the contrast implementation checks before expanding the matrix.

## Core configuration and evidence

- Frozen parent C1 config: `configs/seediv_c1_full45.json` (verify its effective hash before use).
- C2 config: `configs/seediv_c2_e0_s1s2s3.json`.
- Runner: `experiments/seediv_c2_experiment.py`.
- C2 history and output EMA: `models/causal_evidence_history.py`.
- Shared pairwise router: `models/pairwise_utility_router.py`.
- Deterministic split, hash and run identity helpers: `utils/seediv_c2_protocol.py`.
- Detailed protocol and group delivery contract: `docs/SEEDIV_C2_E0_EXPERIMENT_PLAN.md`.
- Current implementation audit/status: `docs/SEEDIV_C2_IMPLEMENTATION_AUDIT.md`.

Each run is expected to retain config/parent/data hashes, actual inner splits, sample and time indices, teacher training records, evidence caches, epoch logs, both source-dev and target-report artifacts, sample-level predictions/actions/utilities, standard classification metrics, action/no-op and negative-utility rates, corrected/harmed prediction counts, and a reproducible run status. Final summaries are reported separately for each session and combined, with paired comparisons on matching session-target-seed cells; subject is the inferential cluster.

## C2 member assignments

Members own methods, not sessions. Every member covers Sessions 1, 2, and 3 separately for their assigned method family, using the same frozen E0 parent, all targets, all seeds, split protocol, and output schema. Each method therefore has 135 outer cells.

| Member | Work package |
|---|---|
| A | B0 reference and B1 probability-EMA controls; frozen-reference checks; sample-level final-fusion diagnostics. |
| B | OOF/source-dev teachers and provenance; evidence cache; C2-current MSE and current-structure Huber contrast. |
| C | Strict history variant; C2-current-EMA; capacity-matched current-feature control; history, reset, and chunked-inference checks. |

The [team handoff](docs/C2_TEAM_EXPERIMENT_HANDOFF.md) defines the machine data interface, group-only run commands, per-run deliverables, and final integration contract. Before the owner accepts the new Session 3 pilot and all 135 E0 parent cells pass audit, member work is limited to data preflight, code/protocol review, and documented acceptance fixes; do not start the full matrix. No member may alter the shared config, data, or code silently; changes require a versioned config and a recorded commit. Do not split sessions across members because method comparisons must cover the same complete scope.

## Original SGDA and other experiments

Original SGDA code remains available in `experiments/seediv/crossSubjects_seediv.py` and the dataset-specific launchers. GeoSem-STDA code is in `models/geosem_stda.py` and `experiments/crossSubject_geosem_stda_sgda.py`. The broader DEAP, SEED, SEED-IV, SEED-V and DREAMER launchers are separate from this C2 study; their options and output conventions should be checked against each launcher before use.

For the separate external-dataset checks, see [`docs/SEED_SEEDV_SINGLE_SESSION_EXPLORATION.md`](docs/SEED_SEEDV_SINGLE_SESSION_EXPLORATION.md) and [`experiments/run_seed_seedv_session1_exploratory.ps1`](experiments/run_seed_seedv_session1_exploratory.ps1). The scope is SEED Session 2 and SEED-V Session 1; it is not part of C2 and its scores must not be pooled with SEED-IV or with each other.

## Data and local model paths

Datasets are not included. Set the local dataset roots in `data_utils/constants/path_mapper.py` and the CLIP model path in `data_utils/text_to_vector.py`. Do not commit private EEG data, pretrained weights, or machine-specific secrets. The C2 run manifest records data fingerprints rather than copying the data into the repository.

## Research and interpretation boundary

The old two-session engineering pilot and provenance audit passed, but they do not validate the expanded three-session config, and one target is not a scientific efficacy study. Do not claim a C2 gain from either pilot. The comparison is intended to distinguish: (1) whether C2-current improves on frozen E0 and which predicted actions account for changes; and (2) whether history adds value over probability smoothing and a capacity-matched current-only control. Report per-subject paired results and uncertainty clustered by subject; overlapping windows and the 135 run cells are not independent participants.
