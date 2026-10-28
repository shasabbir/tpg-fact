# DACS-SCV formal-result audit — 2026-10-28

## Evidence identity

- Returned archive: `dacs-scv-formal-outputs.zip`
- Archive size: 86,517,243 bytes
- Archive SHA-256: `06FE5026B871197B9768DDF31027A7FD47E3F050882084166374229A6B567F7C`
- Executed notebook: `kaggle/TRACE_SCV_Kaggle_after_run.ipynb`
- Frozen code package: DACS-SCV `0.3.2`
- SciVer revision: `5695ac384d247ae5df2a7b73568dc63d189296b0`

The notebook contains eight code cells. All eight have execution counts, none contains an error output, and the final cell identifies the returned formal archive. The archive's internal `scientific_results_validated: false` flag is correct: it was written before independent audit and has not been modified.

## Mechanical and protocol audit

- All 252 declared artifacts match their recorded byte counts and SHA-256 hashes. The only unlisted file is `audit/artifact_inventory.json`, which intentionally cannot include its own hash.
- All 29 recorded code hashes match the frozen local source.
- The formal profile contains the five prespecified seeds (13, 37, 73, 101, and 137), 10,000 source-paper-grouped bootstrap samples, 11 reported variants, and 2,000 official-test claims from 786 source-paper groups.
- The split manifest contains 804 training claims from 325 papers, 196 validation claims from 80 papers, and 2,000 official-test claims from 786 papers. Source-paper intersections are zero.
- The unchanged release gate passes: no source-paper overlap, no exact raw or corrected-perceptual visual overlap, and no exact content overlap involving gold evidence.
- Ten low-Hamming-distance visual pairs are disclosed across split boundaries. Their source papers, fingerprints, evidence identifiers, and serialized captions/tables differ; they are not exact media or corrected-dHash matches. This audit treats them as logged layout-level similarities rather than confirmed duplicates.

`research/audit_formal_outputs.py` independently recomputes verdict macro-F1, citation precision/recall/F1, citation-count metrics, ranking metrics, Brier score, NLL, and ECE from all 100,000 variant--seed claim records. It also replays every aggregate, exact seed comparison, and deterministic grouped-bootstrap comparison. No discrepancy was found. Across 50,000 paired rows, DACS-SCV and its matched fixed-budget outputs have identical claim order, evidence ranking, verdict probability vector, and predicted verdict. Fixed top-one, top-two, and Graph-fixed are exact one-, two-, and three-item prefixes, respectively.

## Main results

The Graph-fixed verifier obtains mean official-test verdict macro-F1 0.5112 and evidence-ranking MAP@10 0.6486 over five seeds. These values are shared by DACS-SCV and the matched fixed-budget outputs because the verifier and ranking are frozen.

DACS-SCV predicts 1.4925 citations on average for a gold mean of 1.6215. It improves citation-count MAE to 0.4954 and exact count match to 0.5565, compared with MAE/exact match of 0.6215/0.4755 for fixed top-one, 0.5725/0.4480 for fixed top-two, and 1.4195/0.0605 for fixed top-three. Better count prediction does not yield the best citation set: DACS-SCV citation macro-F1 is 0.4702, whereas fixed top-two reaches 0.4810.

The prespecified DACS-SCV minus fixed-budget citation-F1 effects are:

| Comparator | Mean effect | Grouped-bootstrap 95% CI | Holm-adjusted p (three-test family) |
|---|---:|---:|---:|
| fixed top-one | +0.0067 | [+0.0013, +0.0120] | 0.0144 |
| fixed top-two | -0.0108 | [-0.0163, -0.0052] | 0.0006 |
| fixed top-three | +0.0133 | [+0.0049, +0.0219] | 0.0056 |

The adaptive-superiority hypothesis therefore **fails**: DACS-SCV improves over top-one and top-three but is worse than the prespecified top-two rule. This is not an inconclusive tie; the grouped interval for the top-two comparison is wholly negative.

## Mechanistic and ablation findings

- Removing the valid-candidate count raises citation macro-F1 from 0.4702 to 0.4747; the full-model minus ablation effect is -0.0044 with grouped 95% CI [-0.0068, -0.0021]. The valid-count feature does not support a positive contribution claim.
- Removing the visual-candidate count yields citation macro-F1 0.4685. The full-model effect is +0.0017 with grouped 95% CI [-0.0002, +0.0037], so a visual-count contribution is unresolved.
- The no-visual-count ablation has the best count MAE (0.4388) and exact count match (0.6126) but not the best citation F1. This reinforces the mismatch between count accuracy and end citation quality.
- Even the gold-cardinality diagnostic averages only 0.4673 citation F1, below fixed top-two. With an imperfect ranking, emitting the annotated number of items is not guaranteed to maximize macro set F1; a global two-item budget can recover additional gold evidence often enough to offset its extra false positives.
- Fixed top-two is better than DACS-SCV in four of five seeds. DACS-SCV's predicted mean count varies from about 1.0 to 2.0 across seeds, identifying cardinality-head stability as a material weakness.
- By gold evidence count, DACS-SCV is competitive for two-item claims (0.6405 citation F1) but underperforms top-two for one-item claims and top-three for claims requiring at least three citations. These strata are descriptive, not independently confirmatory.

## Scientific decision

The formal run is mechanically valid and independently reproducible from the preserved records. It does not validate the intended adaptive-superiority claim. The manuscript must present the result as a controlled negative finding: decoupled cardinality prediction improves annotation-count accuracy, but this objective is misaligned with citation macro-F1 under imperfect evidence ranking and does not beat the best global budget. No post-test tuning or revised adaptive head may be evaluated on the consumed official test and described as confirmatory.

The study remains useful as an empirical analysis of citation cardinality, metric alignment, seed stability, and strict verdict/ranking invariance. Its publication positioning is consequently weaker than a successful-method paper and should emphasize the falsified hypothesis and diagnostic evidence rather than claim a superior model.
