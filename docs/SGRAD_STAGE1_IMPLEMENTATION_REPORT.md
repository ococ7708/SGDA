# SGRAD-EEG STEP 1 Implementation Report

## 1. Files changed

- `models/geosem_stda.py`
- `experiments/crossSubject_geosem_stda_sgda.py`
- `.idea/runConfigurations/SGRAD_*.xml` (local PyCharm configurations)

No historical result directory was modified. No formal training was launched.

## 2. Variant names

- `g0_sgda`
- `g1_sgda_geo_residual`

## 3. Exact difference

`G0_sgda` is the controlled clean SGDA-style representation baseline:

```text
DE [B,3,14,5]
-> StrongDEEncoder
-> h_DE [B,128]
-> six unchanged source adapters
-> unchanged semantic projector
-> fixed CLIP prototypes
```

`G1_sgda_geo_residual` changes only the representation fusion:

```text
source training DE
-> shrinkage covariance
-> one shared source-only Log-Euclidean reference

source/target DE sample
-> tangent deviation R [B,14,14] relative to that same reference
-> Frobenius-isometric symmetric vector (off-diagonals scaled by sqrt(2)) [B,105]
-> small geometry projector
-> g_geo [B,128]

h_final = LayerNorm(h_DE + sigmoid(beta_logit) * g_geo)
```

The scalar gate starts at `beta=0.1`. It is global, not sample-dependent.

G1 does not contain Dynamic Graph, temporal convolution, self-attention, cross-attention, Mamba, ReSGCA, UOT, or conditional alignment.

## 4. Tensor shapes on DREAMER

| Tensor | G0 | G1 |
|---|---:|---:|
| input DE | `[B,3,14,5]` | `[B,3,14,5]` |
| `h_DE` | `[B,128]` | `[B,128]` |
| tangent deviation | not constructed | `[B,14,14]` |
| `g_geo` | absent | `[B,128]` |
| final representation | `[B,128]` | `[B,128]` |
| semantic projection | `[B,512]` | `[B,512]` |

## 5. Parameter counts

| Variant | Total | Strong-DE | Geometry residual | Adapters | Semantic head |
|---|---:|---:|---:|---:|---:|
| G0 | 210,020 | 93,860 | 0 | 50,112 | 66,048 |
| G1 | 240,567 | 93,860 | 30,547 | 50,112 | 66,048 |

The 38 common state tensors in G0 and G1 were verified bit-identical under seed 42. The geometry module is initialized after all shared modules so adding G1 does not perturb common initialization.

## 6. Loss terms

Both variants use exactly the same Rapid classification objective:

```text
L_total = L_prototype_contrastive
L_alignment = 0
```

There is no geometry MSE and no new auxiliary loss. The geometry projector and scalar gate receive gradients only through the unchanged classification loss.

## 7. Source and target information

Targets are fixed to S12, S2, and S4. Each target uses the same six frozen source IDs from `results/dreamer_rapid_pilot3/fixed_sources.json`.

For G1, the shared reference is fitted only from the source samples actually used for that fold. In balanced screening, this means only the selected per-class source subset. Target covariance is mapped using that source reference, but target samples never update the reference.

Target labels are not used in training, geometry construction, source selection, or gradient computation.

## 8. Leakage risk

The geometry path itself has no target-label leakage and no subject-specific tangent reference. All domains share one source-only reference per target fold.

The inherited Rapid Pilot-3 protocol still selects the reported best epoch using target accuracy. This is evaluation/model-selection leakage under a strict held-out definition. It is retained only for direct comparison with current E-series results and is recorded in `run_config.json`.

## 9. Metrics saved at best accuracy

- best accuracy and epoch
- macro-F1 and micro-F1
- balanced accuracy
- recall for class 0 and class 1
- confusion matrix (`TN/FP/FN/TP` on DREAMER)
- predicted and true class ratios
- total and trainable parameter counts
- final accuracy and macro-F1 for diagnostics only

Architecture screening must use Mean Best Accuracy, not final accuracy.

## 10. Interpretability outputs

G1 additionally saves:

- learned scalar `beta`
- mean `h_DE` norm
- mean unscaled geometry-evidence norm
- mean gated geometry-residual norm
- residual/base norm ratio

These values are written to the per-target diagnostics JSON. Geometry is not claimed as a standalone innovation.

## 11. Smoke-test result

Synthetic forward/backward checks passed for G0 and G1. Output shapes were `[B,512]`; G1 beta was initialized to `0.1000000015` and received a finite gradient.

A real DREAMER integration smoke test completed for S12 with two source samples per class, one epoch, and full-target evaluation. It verified:

- shared-source geometry construction
- no graph/temporal modules
- `alignment=0`
- finite loss and beta gradient
- metric and interpretability output serialization

The smoke accuracy is not an experimental result and must not be cited.

## 12. Exact commands

Controlled screening:

```powershell
python experiments/crossSubject_geosem_stda_sgda.py --dataset_name dreamer --rapid_pilot3 --rapid_variant g0_sgda --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --batch_size 64 --seed 42 --device cuda:0
python experiments/crossSubject_geosem_stda_sgda.py --dataset_name dreamer --rapid_pilot3 --rapid_variant g1_sgda_geo_residual --cast_balanced_screening --cast_samples_per_class 128 --epochs 20 --batch_size 64 --seed 42 --device cuda:0
```

Formal full-data commands (do not run until screening is reviewed):

```powershell
python experiments/crossSubject_geosem_stda_sgda.py --dataset_name dreamer --rapid_pilot3 --rapid_variant g0_sgda --target_subject_ids 12,2,4 --epochs 100 --batch_size 64 --seed 42 --device cuda:0
python experiments/crossSubject_geosem_stda_sgda.py --dataset_name dreamer --rapid_pilot3 --rapid_variant g1_sgda_geo_residual --target_subject_ids 12,2,4 --epochs 100 --batch_size 64 --seed 42 --device cuda:0
```

PyCharm provides equivalent named configurations:

- `SGRAD 00 Smoke G1`
- `SGRAD 10 Screen G0`
- `SGRAD 11 Screen G1`
- `SGRAD 20 Full G0`
- `SGRAD 21 Full G1`
