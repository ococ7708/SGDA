# SEED-IV C2 Implementation Audit

Audit date: 2026-09-28. This report separates artifact/code checks from actual C2 model results; it does not claim that the full experiment has completed. The scope was subsequently expanded to Sessions 1–3, so the old two-session pilot is historical and does not qualify the expanded config.

## Verified inputs

- The effective configuration was recomputed from the stored C1 run manifests and confirmed as `592351bfb649e601`.
- The previous E0 audit covered Session 1 and Session 2, 15 targets, and seeds 42/43/44 (90 parent cells). It verified all 90 manifests, completion markers, metrics, histories, and `target_best.pt` files.
- The expanded current C2 scope is Sessions 1–3 separately (135 runs per method). No Session 3 E0 checkpoint/manifest artifacts were found in the audited parent path at this revision: 45 parent cells are missing. Full C2 and the new Session 3 pilot are therefore blocked until those exact-config E0 runs are supplied and audited.
- Each C2 run records a new configuration identity, parent configuration hash, and exact parent checkpoint SHA-256. The fixed parent checkpoint is used for the corresponding outer-target reference evaluation, not as an OOF teacher initialization. OOF teachers independently reconstruct the immutable CLIP prototypes and semantic basis and verify those values against the parent checkpoint buffers.
- Every member's current assignment is method-based and covers all three sessions; no member is assigned a session exclusively.

## Implemented

- Deterministic inner source split (`D=2`, `R=12`, three 4-person OOF folds), with session-target split reuse across run seeds and methods.
- Three fixed-epoch 8-source E0-family teachers plus one 12-source teacher for source-dev evidence. Encoder and trainable E0 source mechanisms are jointly optimized; sampler seeds follow the parent C1 loader convention. Checkpoints contain both model and mechanism state.
- Per-subject data/sample fingerprints, source MAT hashes, full teacher train/evaluation sample-ID sets in a hashed `.npz` manifest, sample/time IDs, teacher logs/checkpoints, and evidence-cache hashes.
- B0/B1, shared variable-K utility routing, MSE/Huber objectives, single positive predicted-utility action, strict-past trial-reset history, probability EMA, source-dev and target-report artifacts, and final-system per-sample correction/harm counts.
- Parent matrix audit, plan-only mode, explicit pilot/full execution gates, and synthetic tests.

## Checks run

- `py_compile` passed for the C2 runner, history utilities, split/hash utilities, and data loader.
- `tests/test_seediv_c2_pipeline.py`: 14/14 synthetic tests passed (split determinism, K=8/12/14, history future-isolation/reset/chunking, probability EMA chunking, no-op behavior, CE utility equivalence, action residual, MSE/Huber separation, sample-level oracle reporting, one-epoch router output smoke, and full-run acceptance/scope gates).
- `tests/run_c1_c2_task06_tests.py`: 5/5 existing C1/C2 regression tests passed.
- Previous two-session parent artifact audit: PASS, `valid_cells=90`, `failures=[]`. The expanded three-session audit is expected to report 45 missing parents until Session 3 E0 is completed.
- Several preliminary attempts exposed implementation/provenance defects and were stopped; none of their partial outputs was reused as the final pilot. The final code-fingerprinted run completed successfully, and cache/output directories are revision-specific.
- Historical fixed pilot: PASS, 4/4 methods (B0, B1, C2-current, C2-history), Session 1 / target 1 / seed 42 under the old two-session config. B0 reproduced the matching parent E0 accuracy, Macro-F1, balanced accuracy, and confusion matrix exactly. This does not pass the new Session 3 pilot gate.
- Independent audit: PASS (`analysis/audit_seediv_c2_pilot.py`). It verified the four actual train/evaluation sample manifests, the D/R/OOF subject partition and sample non-overlap, all teacher and cache hashes, frozen teacher checkpoints across both router trainings, source-dev checkpoint reload metrics, and per-sample final-fusion correction/harm counts.
- Runtime: 1,986.7 seconds on NVIDIA GeForce RTX 5060 Laptop GPU; peak CUDA allocated 747,180,544 bytes and reserved 792,723,456 bytes. The pilot report hash after attaching the verified implementation fingerprints is `9c418eed3a23ee0e9fc145da8b9f4334427390d02cbd9cec30c1f6dbb5273c71`.
- Full C2 has not run. The historical pilot report and independent audit are not scientific result estimates; the one target was fixed for engineering validation only and does not validate the expanded config.

## Known boundaries and remaining gates

- Parent per-subject normalization uses the full unlabeled session, and DE-LDS may be non-causal. Only the C2 history feature construction is intended to be strictly past-only; do not claim whole-system online causality.
- A true-utility single-action oracle is recorded as a label-dependent diagnostic upper bound only; it is not used by router inference or training decisions.
- Every run and evidence cache binds exact implementation-file SHA-256 values, so partial caches from another code revision are refused.
- Synthetic acceptance covers history future-window/reset/chunk invariance, no-op and cached CE checks, MSE/Huber loss separation, shared source counts, and full-gate enforcement. The real pilot covers the end-to-end B0/B1/current/history path. C2-current-EMA, capacity-matched, and Huber still require their planned real-data method runs after the owner opens the matrix.
- The full 135-cell-per-method matrix is blocked until all 135 matching E0 parents are present and audited, and the owner accepts a new Session 3 pilot bound to the expanded config, current code fingerprints, and exact pilot-report SHA. The runner refuses full execution while the parent audit is incomplete.
- This repository work has not uploaded/pushed changes to GitHub. No full C2 result is reported here.
