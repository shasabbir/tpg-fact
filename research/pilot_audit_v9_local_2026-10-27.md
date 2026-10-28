# TRACE-SCV v9 Corrected Local Pilot Audit

Audit date: 2026-10-27  
Run location: `experiments/outputs/pilot`  
Preserved archive: `trace-scv-v9-local-pilot-outputs.zip`  
Archive size: 13,309,503 bytes  
SHA-256: `7ff604d7cde6615889aeb6343302a2c836f24abbd6629200674a3e7d7fc66449`

## Decision

**The corrected pilot is mechanically valid and its data-release gate passes, but the causal-grounding hypothesis fails the prespecified readiness gate. Do not run the causal TRACE-SCV formal matrix.**

This is a one-seed development-validation pilot, not a final result. The official SciVer test was not evaluated.

## Integrity and scope

- Corrected data: 600 training claims from 293 source-paper groups and 150 validation claims from 70 disjoint groups.
- Data release: no source-group overlap, no gold-content overlap, no exact visual overlap, and no near-perceptual visual overlap.
- Required visual media: 165/165 training items and 44/44 validation items present.
- Gold fallbacks: zero in train and validation.
- Artifact inventory: all 36 declared files exist and match their recorded byte counts and SHA-256 hashes.
- Checkpoints: all six trained variant fingerprints independently recompute from their feature hashes and effective configurations.
- Alignment: every variant contains the same 150 unique claims, groups, gold labels, and gold citations.
- Recalculation: verdict macro-F1, citation macro-F1, selected-citation comprehensiveness, and visual-swap sensitivity reproduce exactly from per-claim records for every variant.
- Hardware: AMD Ryzen 7 7700 CPU, 15.19 GB RAM, PyTorch 2.14.1 CPU build; no CUDA device.

The local feature extraction took approximately nine minutes. Training all six one-seed variants took less than thirty seconds after feature extraction, demonstrating that disciplined pilot iteration is practical locally while preserving the frozen encoder outputs.

## Corrected pilot measurements

| Variant | Verdict macro-F1 | Citation macro-F1 | Comprehensiveness | Swap sensitivity |
|---|---:|---:|---:|---:|
| Majority | 0.3243 | 0.0000 | 0.0000 | 0.0000 |
| Claim only | 0.5133 | 0.0000 | 0.0000 | 0.0000 |
| All evidence | 0.5303 | 0.2563 | 0.0243 | 0.0273 |
| Fixed relevance | 0.4750 | 0.2983 | -0.0041 | 0.0038 |
| Flat learned selector | 0.5067 | 0.4536 | -0.0036 | 0.0129 |
| Graph selector without counterfactual losses | 0.5441 | 0.4851 | -0.0160 | 0.1751 |
| Causal TRACE-SCV | 0.5308 | 0.4902 | -0.0076 | 0.0756 |

Relative to the matched graph selector without counterfactual losses, causal TRACE-SCV changes verdict macro-F1 by -0.0133, citation macro-F1 by +0.0051, comprehensiveness by +0.0084 while remaining negative, and swap sensitivity by -0.0996. Visual removal improves the causal model on the visual-grounded subset from 0.5069 to 0.5267 macro-F1.

Four readiness checks therefore fail: verdict preservation, positive comprehensiveness, improved swap sensitivity, and non-improvement after visual removal.

## Falsification outcome

The evidence does not support a causal-grounding contribution. Repeatedly tuning deletion and swap weights after observing this validation failure would weaken the study. The causal claim, causal-readiness gate, and corresponding formal ablation family are retired.

The pilot supports a narrower direction: supervised sparse citation selection is substantially stronger than fixed semantic relevance on this split, but the fixed top-three citation budget is structurally mismatched to the data. Gold evidence-set sizes are variable: the training mean is 1.64 (maximum 7), and 545/600 training claims have only one or two gold items. The next pilot will test a predeclared adaptive citation-cardinality selector, with the official test remaining untouched.
