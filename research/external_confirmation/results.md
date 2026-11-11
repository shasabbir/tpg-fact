# Independent External Confirmation Results

Completed: 2026-11-12  
Dataset: AVerImaTeC organizer static knowledge store  
Protocol code: `218dbf1ac88630a06d5ad3cce3e1cacaeb84ddcc`  
Train-only freeze commit: `b2075d0`  
Train-only freeze SHA-256:
`36e4210a438a9c6c0051d6c4608b9d2fc04709aceaecfe5c9a694518bfa0e314`

## Integrity boundary

The five train-only seeds were frozen before validation preparation. The
freeze accepted only train rows and records `development_outcomes_accessed:
false`. Validation rankings were written and hashed before gold-source labels
were joined. The locked comparison ran once on 152 validation claims. Test
annotations were not accessed, test predictions were not generated, and no
submission was performed.

The validation pre-join ranking contains 18,240 rows and has SHA-256
`b24739c9e85763561cd248eae7b572f1dac2cce0956cfc578220c615bfe719c0`.
The preserved locked claim records contain 3,800 rows and have SHA-256
`eec62a20ba92361d2fc937208304abd06dc15a7f0f3c1e706351b6a47aa14f92`.
An independent reporting pass reproduced every aggregate, all four 10,000-draw
claim-aligned bootstrap comparisons, and the shared-verifier metrics. Its
checked result has SHA-256
`0751ce76ecb7b4abd7c535bea7bb35313643b8f4ae746502763e807734ad915d`.

## Reachability before selection

The official validation store supplied 101,289 text candidates and 49,617
images. The frozen retrieval path retained 100 text and 20 image candidates
per claim before the train-selected packet width was applied. Under exact
normalized source identity, the retained pool reaches 25.48% of annotated
targets, completely reaches 12.50% of claims, and has zero reachable target for
59.87% of claims. Recall@10/20/50/100 is 5.26%, 6.96%, 11.43%, and 23.78%,
respectively. These are diagnostic exact-source values, not the official
semantic evidence score.

One organizer image contained 144 million pixels and triggered PIL's
decompression-bomb warning during SigLIP2 encoding. The image was processed;
there was no exclusion, OOM, or fallback encoder.

## Locked selector results

All values are claim means after averaging each claim over the five train-only
seeds.

| Policy | Precision | Recall | Exact-source F1 | Exact set | Items |
|---|---:|---:|---:|---:|---:|
| Train-selected fixed top-k | 0.0846 | 0.1116 | 0.0857 | 0.0053 | 2.59 |
| Calibrated threshold | 0.1308 | 0.0725 | 0.0857 | 0.0289 | 1.17 |
| Calibrated expected-F1 | 0.1198 | 0.1214 | 0.1029 | 0.0145 | 3.20 |
| Uncalibrated expected-F1 | 0.0306 | 0.1614 | 0.0497 | 0.0000 | 9.93 |
| Completeness-aware switch | 0.1208 | 0.1160 | 0.1013 | 0.0145 | 2.71 |

The paired intervals below use 10,000 claim-aligned bootstrap draws. The
reported bounds are Bonferroni simultaneous 98.75% intervals for the four-test
family, with Holm-adjusted two-sided p-values.

| Comparison | Mean F1 effect | Adjusted interval | Holm p | Decision |
|---|---:|---:|---:|---|
| Threshold - fixed | +0.00003 | [-0.02341, +0.02404] | 0.9989 | Superiority failed |
| Expected-F1 - threshold | +0.01714 | [-0.00148, +0.03905] | 0.0714 | Non-inferiority passed; compactness failed |
| Calibrated - uncalibrated expected-F1 | +0.05316 | [+0.02207, +0.08852] | 0.0008 | Calibration benefit passed |
| Switch - threshold | +0.01555 | [-0.00263, +0.03621] | 0.0714 | Superiority failed |

Expected-F1 returns 3.20 items versus 1.17 for thresholding, a 173.96% increase
rather than the prespecified 15% reduction. The switch effect is positive in
all five seeds but its multiplicity-adjusted interval crosses zero and it
returns 2.71 items, so the full switch gate fails.

## Shared verifier

The frozen two-layer verifier obtains mean macro-F1 0.1573 with calibrated
threshold evidence and 0.1444 with switch evidence. The seed-aligned switch
minus threshold effect is -0.01293 with 95% interval [-0.03242, +0.00687], so
the -0.010 non-inferiority safeguard fails.

| Policy | Accuracy | Macro-F1 | Supported F1 | Refuted F1 | NEI F1 | Conflicting F1 |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated threshold | 0.3158 | 0.1573 | 0.1150 | 0.4710 | 0.0432 | 0.0000 |
| Completeness-aware switch | 0.2842 | 0.1444 | 0.1000 | 0.4300 | 0.0474 | 0.0000 |

The validation distribution contains only four Supported, six Not Enough
Evidence, one Conflicting Evidence/Cherrypicking, and 141 Refuted claims.
Neither policy correctly resolves the single conflicting-class case across the
five seeds. Accuracy is therefore not an adequate headline metric.

## Prespecified decision and remaining boundary

Only the calibration-benefit hypothesis passes. Threshold superiority,
expected-F1 compactness, switch superiority, and verifier non-inferiority fail.
The frozen blind-selection rule therefore falls back to the train-selected
fixed policy family. The official Gemma-3-27B semantic evaluator remains
unavailable on the single RTX 4090 under the organizer's specified execution
condition, so no official evidence score is reported.

The first local fixed-policy export failed because the locked runner had
attached shared-verifier predictions only to threshold and switch rows. This
was an implementation omission, not a scientific rerun. Deterministic
inference from each already frozen verifier was added without refitting,
retuning, or changing selected evidence. All five fixed-family development
exports then passed the official schema with 152 predictions and at most three
evidence items. They were not submitted. A blind test remains authorization
gated.

The external result is a nonreplication of selector superiority and of the
compact expected-F1 operating point. It does replicate the importance of
calibration and strengthens the paper's central diagnostic conclusion:
candidate reachability and calibration must be established before adaptive
set-selection claims are interpreted.
