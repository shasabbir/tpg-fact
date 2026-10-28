# TRACE-SCV v7 Pilot Audit

Audit date: 2026-10-27  
Archive: `trace-scv-pilot-outputs.zip`  
Archive size: 10,916,361 bytes  
SHA-256: `c07364083c062cc5c4e9f738af9a1eab7ffad6f16c23c9c7c8e14d04450e13ca`

## Decision

**The replacement pilot fails the preregistered readiness gate. The formal experiment must not run.**

The run is mechanically trustworthy and independently recomputable for the two primary endpoints. It improves citation macro-F1 over the matched no-counterfactual baseline, but the verdict loss exceeds the one-point tolerance and the intervention behavior still contradicts the causal-grounding claim. These are scientific failures, not packaging failures.

## Protocol recovered from the archive

- Profile: `pilot`; evaluation split: development validation.
- Official SciVer test examples evaluated: 0.
- Training: 600 claims, 293 source-paper groups, 295 refuted and 305 supported.
- Validation: 150 claims, 70 source-paper groups, 78 refuted and 72 supported.
- Train--validation source-paper overlap: 0.
- Required figure media coverage: 100% in train (165/165) and validation (44/44).
- External MuSciClaims evaluation: disabled.
- Feature backend: frozen BGE/CLIP, 512 dimensions.
- Seed: 13; grouped bootstrap samples: 1,000.
- Hardware: Tesla T4; all six trained configurations completed.

This is a one-seed, validation-only diagnostic. It is not a final estimate of generalization or stability.

## Integrity and reproducibility checks

| Check | Outcome |
|---|---:|
| ZIP path-safety screen | pass |
| Declared inventory entries | 37 |
| Actual files excluding the inventory itself | 37 |
| Missing, unlisted, byte-count, or SHA-256 mismatches | 0 |
| Recorded code files compared with the workspace | 26/26 exact |
| Aligned unique claim records per configuration | 150/150 |
| Gold label, group, and gold-citation alignment across configurations | exact |
| Prediction/comparison-record agreement | exact |
| Probability rows finite and summing to one | pass |
| Checkpoint/input/configuration fingerprints | 6/6 exact |
| Primary endpoint recomputation | exact for all seven configurations |
| Paired seed comparison recomputation | exact |
| Group-bootstrap comparison recomputation | exact |

The archive does not export per-claim clean and intervention probabilities. Consequently, comprehensiveness, invariance, and swap sensitivity can be checked against the recorded code and aggregate metrics but cannot be independently recomputed from the exported rows. Version 8 must export these rows.

## Independently recomputed primary endpoints

| Configuration | Verdict macro-F1 | Citation macro-F1 |
|---|---:|---:|
| Majority | 0.3243 | 0.0000 |
| Claim-only | 0.5116 | 0.0000 |
| All-pool | **0.5593** | 0.2563 |
| Relevance | 0.4971 | 0.3783 |
| Flat selector | 0.5158 | 0.5756 |
| TRACE-no-CF | 0.5016 | 0.5731 |
| TRACE-SCV | 0.4822 | **0.5905** |

TRACE-SCV minus TRACE-no-CF is -0.0194 verdict macro-F1 and +0.0173 citation macro-F1. The paired source-paper bootstrap reproduces exactly:

- verdict difference: mean -0.0194, 95% interval [-0.0775, 0.0356], Holm-adjusted p = 1.0000;
- citation difference: mean +0.0173, 95% interval [0.0027, 0.0366], Holm-adjusted p = 0.0280.

The interval and p values are pilot diagnostics only: there is one trained seed and the same validation split controlled early stopping.

## Preregistered readiness gate

| Gate item | Evidence | Result |
|---|---:|---:|
| Citation macro-F1 improves over TRACE-no-CF | +0.0173 | pass |
| Verdict loss is no more than 0.0100 | -0.0194 | **fail** |
| Selected-citation comprehensiveness is positive | -0.0068 | **fail** |
| Comprehensiveness improves over TRACE-no-CF | +0.0041 | pass |
| Visual-swap sensitivity improves over TRACE-no-CF | -0.0585 | **fail** |
| Same-modality swap coverage is positive | 0.5533 | pass |
| Removing visual evidence does not improve the verdict | 0.4822 clean versus 0.5334 removed | **fail** |

The pilot passes three of seven checks and fails four. The archive correctly records `pilot_readiness_gate.status = fail` and `scientific_results_validated = false`.

## Additional findings

- The all-pool baseline has the strongest pilot verdict macro-F1 (0.5593), 0.0771 above TRACE-SCV.
- TRACE-SCV is better calibrated than TRACE-no-CF on this validation run (ECE 0.1214 versus 0.2614; Brier 0.5530 versus 0.6680), but calibration is secondary and does not repair failed faithfulness or verdict gates.
- TRACE-SCV's cited-only verdict macro-F1 is 0.4918, but removing its selected citations increases rather than decreases the average gold-label probability.
- Visual removal improves TRACE-SCV verdict macro-F1 by 0.0512. Its visual-swap sensitivity is 0.0284 versus 0.0869 for TRACE-no-CF.

## Data-quality findings

- Claim identifiers are unique and source-paper groups are disjoint.
- No exact or near-perceptual media overlap occurs across train and validation.
- Six exact evidence-content fingerprints occur in both train and validation. The archive reports only the count, not the claim IDs, source-paper groups, gold/non-gold roles, or text. This is a **medium-severity unresolved leakage risk**. It may be harmless repeated boilerplate, but that cannot be established from this archive.
- Within-split repeated content is common because the same paper-local distractors appear in multiple claim packets. That is compatible with paper-grouped splitting; cross-split repetitions still require item-level inspection.

## Code-level diagnosis and required correction

The v7 intervention alignment fix is present, but the visual representation and evaluation are confounded:

1. `features.py` routes every table, figure, or image with available media through the CLIP image encoder. Its caption or serialized table content is used only as a fallback when media is missing. Thus, the model discards useful textual/numeric content precisely when an image exists.
2. The current `no_visual` evaluation masks the entire table/figure/image evidence node. It removes both the visual item and every semantic cue associated with that candidate instead of isolating the pixel channel.
3. The same-modality swap replaces the complete evidence feature. With a single fused slot, it cannot keep the caption/table semantics fixed while changing only the pixels.

These observations do not prove the cause of the failed metrics, but they provide a concrete and testable explanation for weak or harmful visual dependence.

Before one final validation pilot, version 8 should:

1. store separate semantic-text and pixel-visual features for each evidence item;
2. encode captions and serialized tables even when media is available;
3. project and fuse the two channels with separate trainable projections and an explicit availability mask;
4. make visual removal and swaps operate on the pixel channel only;
5. export per-claim clean/deleted/swap probabilities and applied-swap flags;
6. export the six cross-split content overlaps with claim, group, modality, and gold-role metadata;
7. report intervention effects on the visual-grounded subset in addition to the full validation set.

The next pilot is a hard stop. If the same causal and visual-dependence gates fail again, the paper must drop the causal-grounding claim and pivot to a narrower citation-selection study rather than running the formal matrix.
