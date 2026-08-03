# RSG-CutMix 5-Target Experiment Result

This result evaluates the first implementation of RSG-CutMix v1 under the same senior-code target-best protocol as the GeoSem-STDA sparse reliability baseline.

## Protocol

```text
Dataset = DEAP
Task = valence binary classification
Targets = 3, 14, 20, 23, 32
Candidate sources per target = 31
Final selected sources per target = 6
Epochs = 200
Batch size = 64
Learning rate = 0.001
Seed = 42
Source selection = sparse_reliability
RSG-CutMix = enabled
rsg_prob = 0.3
rsg_lambda_aug = 0.1
rsg_time_min = 2
rsg_time_max = 4
rsg_band_width = 1
rsg_channel_ratio = 0.25
```

Run directory:

```text
results/results_deap_geosem_stda/runs/ep200_bs64_lr0p001_lmda0p3_tau0p07_topk6_seed42_20260803_203335_rsgp0p3_augl0p1
```

## Result

| Target | Baseline Acc | RSG Acc | Delta Acc | Baseline Macro-F1 | RSG Macro-F1 | RSG Final Acc |
|---:|---:|---:|---:|---:|---:|---:|
| S3 | 66.67 | 65.00 | -1.67 | 66.52 | 64.91 | 62.50 |
| S14 | 71.11 | 70.00 | -1.11 | 71.00 | 69.92 | 62.50 |
| S20 | 57.50 | 56.25 | -1.25 | 57.26 | 56.08 | 48.61 |
| S23 | 54.72 | 50.42 | -4.31 | 51.77 | 41.20 | 45.00 |
| S32 | 60.00 | 60.00 | +0.00 | 59.90 | 58.56 | 32.50 |

Mean comparison:

```text
Baseline Acc = 62.00 +/- 6.75
Baseline Macro-F1 = 61.29 +/- 7.59

RSG-CutMix Acc = 60.33 +/- 7.59
RSG-CutMix Macro-F1 = 58.14 +/- 10.91
RSG-CutMix final epoch Acc = 50.22 +/- 12.70
```

## Conclusion

RSG-CutMix v1 did not improve the current GeoSem-STDA sparse reliability baseline.

Observed behavior:

- Target-best Acc decreased from `62.00%` to `60.33%`.
- Macro-F1 decreased from `61.29%` to `58.14%`.
- S23 became much worse, especially Macro-F1.
- S32 kept the same best Acc as baseline but collapsed badly by epoch 200.
- The augmentation loss entered training normally, but it often became small quickly and did not prevent late degradation.

Current judgment:

```text
The idea remains usable as a research direction, but this first v1 setting should not be treated as an effective model improvement.
```

Recommended next changes:

1. Reduce augmentation strength:
   - `rsg_prob = 0.1`
   - `rsg_lambda_aug = 0.03 or 0.05`

2. Delay augmentation:
   - do not apply RSG-CutMix in the first 10-20 epochs
   - enable it only after the semantic space becomes less noisy

3. Use RSG only for high-confidence sources:
   - ignore sources with near-uniform reliability weights
   - apply augmentation only to top 2-3 selected sources

4. Diagnose MMD first:
   - current late degradation still appears dominated by MMD / negative transfer
   - RSG-CutMix alone cannot fix an incorrect alignment direction

