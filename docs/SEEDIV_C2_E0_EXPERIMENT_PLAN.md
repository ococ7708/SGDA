# SEED-IV C2 on the Frozen E0 Reference

Status: three-session scope configured; execution is blocked pending the missing Session 3 C1 E0 parent runs and a new Session 3 pilot. Do not interpret this plan as a completed C2 experiment. Sessions remain separate LOSO evaluations and are never pooled for training.

## 1. Questions and scope

The study answers two questions:

1. Does C2-current, trained to predict source-evidence utility, improve the existing frozen E0 reference, and which selected actions explain any changes?
2. Does strict past-margin history add value beyond current-window evidence, probability smoothing, and a capacity-matched current-only control?

The official scope is SEED-IV DE-LDS, Sessions 1, 2, and 3 evaluated separately. Each session has 15 outer target subjects and seeds 42, 43, and 44: 135 outer target/seed cells per method, with no cross-session training. Cross-dataset and clinical experiments are out of scope.

The parent is the C1 E0 pipeline. Before any run, recompute the effective hash from its contents and validate every required parent manifest/checkpoint. The reviewed parent config hash is `592351bfb649e601`; the expanded three-session C2 config has a new identity. The previous audit verified 90 Session 1/2 E0 cells, but all 45 Session 3 E0 cells are missing from the audited parent path at this revision. Each C2 run records `parent_config_hash` and the SHA-256 of its exact parent checkpoint. Never trust a directory name alone. Full C2 remains blocked until Session 3 E0 is completed under this same parent config and all 135 cells pass audit.

## 2. Frozen E0 protocol

The implementation inherits the Session 1/2 C1 E0 settings and actual run manifest, including:

- DE-LDS, four-class label order, the same source data version, `sample_length=3`, `stride=1`, and the same data/sample ordering;
- original subject-wise standardization;
- Strong-DE / channel / graph backbone, original source adapters, fixed CLIP prototypes, and original `branch_logits` fusion;
- one outer target held out; the other 14 subjects are the source set;
- backbone teachers use the original 200-epoch, batch-64 E0 optimizer and loss settings;
- seeds 42, 43, and 44.

The parent E0 checkpoint is fixed per session-target-seed. Do not retrain a replacement E0. B0 must reproduce that exact run’s stored metrics and confusion matrix before any teacher fitting for the fold. Report any tolerance or mismatch; a mismatch blocks the fold. A missing parent cell must be listed exactly and cannot be substituted by a different checkpoint.

C2 uses new experiment/run identities. Preserve the parent E0 results. Router training budget is separately fixed at 100 epochs, batch 256, Adam, learning rate 0.001. This is an added module budget, not equivalent total compute to the backbone’s 200 epochs.

## 3. Internal OOF teachers and router evidence

For each outer target, deterministically divide its 14 source subjects into source-dev set `D` of two and router/OOF pool `R` of twelve. Use `split_seed=20260927`, algorithm `seedsequence_session_target_v1`, and record the actual subject IDs. The same split is reused across C2 variants and backbone seeds for the same session-target. Do not include the outer target or D in OOF teacher fit, standardization fit, or router gradient updates.

Split R into three subject folds of four. For each fold, fit an E0-family teacher on the other eight R subjects and export evidence for the four held-out subjects. Fit a fourth 12-source teacher on all of R and export source-dev evidence for D. The four teachers use the fixed original 200-epoch backbone budget. Do not use OOF held-out labels for teacher checkpoint selection. No outer-target data enters any teacher fit. OOF/R labels alone supervise the router's utility loss. D labels are evaluation-only for the source-dev checkpoint comparison; D evidence is never included in router gradient updates. Outer-target labels are used only for final system evaluation and the predeclared descriptive epoch report.

Teacher/evidence provenance must be based on actual data and checkpoints, not a self-declared held-out list. Record for every teacher:

- outer session/target/seed, train/evaluation subject IDs, actual fit sample IDs and their fingerprint;
- per-subject raw and normalized feature fingerprints, label/trial fingerprints, and evaluation sample/time indices;
- normalization rule and scope, training log, checkpoint path/hash, teacher and prototype identities;
- evidence rows, feature/label fingerprints, source count, cache hashes, and a machine-checkable train/evaluation disjointness result.

The shared router must work with K=8 OOF sources, K=12 source-dev sources, and K=14 outer-target source experts. Measure/report the feature shift between these counts. Evidence can be cached across C2 variants after all identities and hashes validate. Reuse across outer targets only if each receiving fold’s inclusion/exclusion protocol is independently proven; otherwise train/cache separately.

### Normalization and causality boundary

Follow the original per-subject normalization rule. The evaluated subject may contribute its own unlabeled observations to its own normalization moments, but it cannot contribute to shared training statistics or learned parameters. Record the exact feature and sample scope. Since the parent may normalize using an entire subject/session and DE-LDS may use temporal smoothing, do not call the entire system online-causal. The C2 history state is causal with respect to window order within each trial; that narrower claim is valid only after its tests pass.

## 4. Methods

| ID | Definition | Purpose |
|---|---|---|
| B0 | Frozen original E0 reference | Common baseline and exact re-evaluation |
| B1 | B0 plus causal probability EMA (`output_ema_alpha=0.8`) | Test whether ordinary smoothing is sufficient |
| C2-current | Current-window features; MSE utility router | Test basic C2 mechanism |
| C2-current-EMA | C2-current followed by output probability EMA | Separate utility routing from output smoothing |
| C2-history | Current features plus strict-past margin summaries; MSE | Test added temporal evidence |
| C2-current-matched | Same input structure as history, history slot replaced by a fixed current-feature mapping | Control parameter/capacity increase |
| C2-current-Huber | Current structure with the original Huber objective | Isolate loss choice |

Engineering sequence: first implement/verify B0, B1, C2-current, and C2-history. Complete the other contrasts before matrix expansion. The new pilot is fixed to Session 3, target 1, seed 42 and those four methods; it trains four teachers total and reuses their cache for both routers. This pilot cannot run until its matching Session 3 E0 parent is present. The old Session 1 pilot is retained as historical evidence only; its config/code fingerprints do not unlock the expanded matrix. Until all 135 E0 parents are audited and the project owner accepts the new pilot report, member tasks are limited to protocol review and resolving documented acceptance failures—no member starts the full 135-cell-per-method matrix. The pilot is an engineering acceptance check, never a basis for choosing favorable folds or changing fixed parameters.

### Utility and actions

For each example, source, and each of six unordered class pairs, the utility label is the full multiclass CE change:

`u(k,a,b) = CE(reference_logits, y) - CE(action_logits(k,a,b), y)`.

Enumerate all source/pair candidates. At inference choose the candidate with the greatest predicted utility if it is positive; otherwise take a no-op. Do not use the true class to filter candidates; do not multiply confidence quantities into a hand-built transfer weight; do not combine multiple actions in this first study.

For diagnosis only, also compute an oracle single-action upper bound that uses the realized full-CE utility to choose among those same candidates after the deployable prediction is fixed. Label its outputs explicitly as label-dependent diagnostics; never use oracle actions in router training, model selection, or inference.

Fixed action parameters are `action_alpha=0.25`, `residual_clip=2.0`, and `action_mode=single`. The inference decision must be made before labels are read; labels may be used afterward for evaluation and actual-utility diagnostics. B0/B1 no-op behavior must preserve probability and prediction (within numeric tolerance; probability equivalence is acceptable if logits are centered).

### History and smoothing

History consists of past mean margins for reference and source experts, past disagreement variability, current-minus-past margin difference, and a history mask. Read the state before acting on the current window; only then update it. The first window has no history. Reset at each subject/session/trial. Preserve the actual trial ID and window/time index; never infer time from shuffled batch order. Fixed `history_alpha=0.8`; first-round parameters are locked and not searched.

`C2-current-matched` uses the same structured router input dimension as history but replaces each historical slot with a fixed, documented transform of current features. This mapping is frozen in config and tested so it cannot read previous windows.

## 5. Selection, reporting, and fairness

B0/B1 evaluate the fixed parent checkpoint and record `parent_best_epoch` from the parent metadata. The router is evaluated every epoch as a complete system. Keep both (a) the predeclared target-accuracy-maximum checkpoint for descriptive reporting, with companion F1, balanced accuracy, and confusion matrix from the same epoch, and (b) the source-dev-selected router, which is the deployable/mechanistic comparison. Tied maximum accuracy retains the earliest epoch. The target-label path must never enter router gradients, history input, action selection, utility prediction, or hyperparameter selection.

Because target reporting over router epochs adds a model-selection opportunity, report the current-versus-history comparison on the same source-dev-selected router as well as the descriptive per-method target report. C2-current and C2-history must share the same teacher/cache, seeds, router training budget, and evaluation frequency. Save complete epoch history and the source-dev reload check.

Every method/run reports accuracy, macro-F1, balanced accuracy, CE, confusion matrix, sample-level predictions and actions, predicted and realized selected utility, no-op/action rates, negative realized-utility rate, corrected errors, harmed correct predictions, and source-count feature drift. Compute flip diagnosis from the **final fused system’s per-sample predictions**, not branch-average flips.

Summaries show Session 1, Session 2, Session 3, then the combined scope, paired on matching session-target-seed cells. Inferential uncertainty clusters by subject; 135 cells, windows, and overlapping windows are not independent people.

## 6. Run gates and commands

From the repository root (PowerShell), first run only the read-only checks:

```powershell
$PY = "C:\Users\oc200\anaconda3\envs\sgda_py311\python.exe"
& $PY experiments\seediv_c2_experiment.py audit
& $PY experiments\seediv_c2_experiment.py plan
& $PY tests\test_seediv_c2_pipeline.py
```

The first two commands validate the parent artifacts and print the execution plan. The test command covers synthetic protocol invariants. Then, only the project owner starts the one real-data pilot:

```powershell
& $PY experiments\seediv_c2_experiment.py pilot --execute --device cuda:0
```

Inspect `artifacts/seediv_c2_s1s2s3_parent_audit.json` and the new report `artifacts/seediv_c2_s1s2s3_pilot_report.json`. Full execution is unlocked only after all 135 E0 parent cells pass audit, every applicable pilot acceptance item below passes, and the project owner records acceptance in `artifacts/seediv_c2_s1s2s3_pilot_acceptance.json`, based on [`templates/seediv_c2_pilot_acceptance_TEMPLATE.json`](templates/seediv_c2_pilot_acceptance_TEMPLATE.json). The record must include the current config hash, implementation hashes, and exact new pilot-report SHA-256. Do not run or distribute the 135-cell-per-method matrix before that gate. Full execution requires an explicit execution flag, all valid parent checkpoints, a passing pilot report, and the signed acceptance record.

Acceptance checklist:

1. B0 re-evaluates the corresponding E0 cell exactly within stated tolerance.
2. OOF exclusions are checked at actual sample/checkpoint level; D and target do not leak into OOF or router fit.
3. Modifying future windows cannot alter current/past history features or decisions.
4. Trial reset works; full-trial and chunked streaming outputs agree.
5. No-op preserves probabilities and predictions.
6. Cached utility equals direct full-class CE recomputation.
7. Final fused prediction changes count corrected errors and harmed predictions by sample.
8. Router fitting does not change teacher parameters/checkpoint hashes.
9. MSE versus Huber, single action, and history configuration each take effect independently.
10. If resume is supported, persist sampler state and verify resumed training consistency.

The pilot also records peak memory, wall time, throughput, and observed feature/cache sizes. A failed check blocks the affected fold and matrix. The project owner records the pilot config hash, implementation hashes, report hash, PASS/FAIL for each checklist item, any approved corrective code/config version, and acceptance date before full runs begin. The CLI rejects missing, incomplete, stale, mismatched, or non-PASS acceptance records. Do not repair results manually or change the fixed fold after observing target metrics.

## 7. Three-member method-based assignment

All three members use the same Sessions 1, 2, and 3 separately, all 15 targets, seeds 42/43/44, frozen E0 checkpoints, config, split generator, and cache/run schemas. The work is assigned by method, not by session; every method covers all 135 cells. These full-scope assignments begin only after all 135 parent cells are audited and the project owner accepts the new engineering pilot. Before that acceptance, do not start the full matrix. Coordinate file ownership and integrate only versioned commits/configs; never make silent changes to shared preprocessing or method parameters.

### Member A — fixed reference and diagnosis

- Verify effective parent hash and the exact parent checkpoint SHA for each run.
- Export frozen E0 fused-reference logits, every source expert’s logits, source distances, prototype-pair features, sample IDs, and trial/window indices.
- Implement/run B0 and B1; validate B0 against stored E0 output and check B1 EMA/reset/chunk behavior.
- Produce the frozen-reference action/oracle diagnostic without using the target labels for decisions.
- Deliver per-sample final-fusion `wrong→right` and `right→wrong` records, no-op invariant report, metrics, config/data/model hashes, logs, and failed/missing-cell inventory.

### Member B — OOF evidence and current utility

- Generate the deterministic D/R and three 4-person OOF folds; train three 8-source teachers plus the 12-source D teacher per outer-fold/backbone-seed.
- During pilot acceptance only, run those four teacher fits for Session 3/target 1/seed 42; expand to remaining cells only after all E0 parents pass audit and written owner acceptance.
- Verify actual training sample sets and excluded samples; write teacher logs, data fingerprints, checkpoint SHA values, evidence caches, and provenance.
- Implement/evaluate C2-current MSE, then C2-current-Huber with the identical evidence, router seeds, and schedule.
- Report K=8/12/14 feature drift, utility agreement to direct CE, action/no-op rates, negative-utility events, corrected/harmed predictions, and source-dev/target-report artifacts.

### Member C — temporal and smoothing controls

- Implement strict-past history, state/trial reset, future-window noninterference, and chunk equivalence.
- Implement C2-history MSE, C2-current-EMA, and the capacity-matched current control with fixed documented mapping.
- Confirm that decisions use pre-update evidence; router selection never consumes target labels.
- Compare history against current-only on the shared cache and source-dev-selected checkpoints; quantify whether history adds over EMA and matched capacity.

### Each member’s run package

For every assigned method and every matching cell, deliver:

1. Commit SHA, immutable config/hash, parent config hash and exact parent checkpoint SHA, environment, GPU, data version/fingerprints.
2. Actual outer/inner split, teacher IDs, fit/evaluation subject/sample IDs, time index and their hashes.
3. Full training/evaluation logs, checkpoint files and their hashes, parent/router metadata, and completion/failure status.
4. Per-sample reference/final predictions, labels for evaluation, action/no-op and source/class pair, predicted and realized utility.
5. Accuracy, macro-F1, balanced accuracy, CE, confusion matrix, action/no-op/negative-utility rate, corrected/harmed count, and reproducible summary script.
6. Exact traceback and missing-artifact list for failed cells; never fill a failed result with zero or omit an unfavorable target.

## 8. Current code map and model flow

`experiments/seediv_c2_experiment.py` runs audit/plan/pilot/full; `models/geosem_stda.py` implements the E0 backbone/adapter/fusion family; `models/pairwise_utility_router.py` defines the shared variable-source-count router; `models/causal_evidence_history.py` maintains past margins and probability EMA; `utils/seediv_c2_protocol.py` controls config identities and inner splits. `tests/test_seediv_c2_pipeline.py` exercises synthetic C2 invariants.

Data flows as: subject-window DE-LDS → frozen shared Strong-DE/channel/graph encoder → per-source adapter + fixed CLIP prototypes → original branch-logit fusion (reference) → enumerate K×6 pair actions → shared router predicts complete CE utility → apply one positive action or no-op → compute evaluation and action diagnostics. OOF labels supervise utility only for examples whose corresponding teacher did not train on that subject. D provides source-only router selection. The outer target is evaluation-only.

The project-level explanation is in [`README.md`](../README.md). Older C1/C2 notes remain historical unless explicitly linked from this plan.
