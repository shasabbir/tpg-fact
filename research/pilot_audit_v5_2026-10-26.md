# TRACE-SCV v5 encoder-valid pilot audit

Audit date: 2026-10-26  
Artifact: `trace-scv-pilot-outputs.zip`  
Archive size: 10,914,692 bytes  
SHA-256: `8bf2843ffc2cf952afabfed642d718561e9ea1f718bf08317e482af84769bc19`

## Decision

**Formal-run readiness: blocked pending one method-alignment correction and a replacement validation-only pilot.**

The v5 archive passes every mechanical, provenance, split-integrity, and metric-recomputation check. Its primary matched comparison is promising: relative to TRACE-no-CF, full TRACE-SCV gains 0.0173 citation macro-F1 while losing 0.0079 verdict macro-F1, which is inside the prespecified one-point verdict tolerance. However, the counterfactual-faithfulness behavior that motivates the method remains in the wrong direction, and code inspection identifies mismatches between the training and evaluation interventions. The official SciVer test must remain unevaluated until those issues are corrected using development data.

## Provenance and data checks

- Profile is `pilot`, feature backend is `hf`, dimension is 512, and evaluation split is `validation`.
- The pinned SciVer revision is `5695ac384d247ae5df2a7b73568dc63d189296b0`.
- Training contains 600 claims from 293 source-paper groups: 295 refuted and 305 supported.
- Validation contains 150 claims from 70 source-paper groups: 78 refuted and 72 supported.
- No source-paper group overlaps train and validation; no selected source paper is missing.
- The pilot official-test packet count is zero.
- The artifact inventory lists 37 files. Every listed byte count and SHA-256 digest matches, with no missing or unlisted artifact.
- All 25 recorded code/configuration hashes match the current v5 workspace.
- All six checkpoint fingerprints independently recompute from their exact train/validation feature hashes and training configurations.
- All systems contain 150 aligned, unique predictions. Labels, groups, and gold citations agree across systems; probability vectors are valid; exported predictions agree with comparison records.
- Verdict macro-F1, citation macro-F1, Brier score, NLL, ECE, and every paired group-bootstrap comparison recompute exactly from the exported records.

## Pilot results

| System | Verdict Macro-F1 | Citation Macro-F1 | Comprehensiveness | Sufficiency Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Majority | 0.3243 | 0.0000 | 0.0000 | 0.3243 | 0.0283 |
| Claim-only | 0.5116 | 0.0000 | 0.0000 | 0.5116 | 0.0129 |
| All-pool | 0.5593 | 0.2563 | 0.0334 | 0.5593 | 0.3237 |
| Relevance | 0.4971 | 0.3783 | -0.0077 | 0.4971 | 0.0866 |
| Flat selector | 0.5158 | 0.5756 | -0.0163 | 0.5158 | 0.1674 |
| TRACE-no-CF | 0.5016 | 0.5731 | -0.0109 | 0.4842 | 0.2614 |
| TRACE-SCV | 0.4937 | 0.5905 | -0.0067 | 0.4764 | 0.1138 |

For the prespecified matched comparison against TRACE-no-CF:

- verdict macro-F1 difference: -0.0079 point estimate; grouped-bootstrap mean -0.0082, 95% interval `[-0.0672, 0.0496]`;
- citation macro-F1 difference: +0.0173; grouped-bootstrap interval `[0.0027, 0.0366]`, Holm-adjusted pilot p-value 0.028;
- irrelevant-addition invariance improves from 0.9842 to 0.9949;
- duplicate invariance improves from 0.9768 to 0.9907;
- calibration ECE improves from 0.2614 to 0.1138.

These values are feasibility evidence only: the checkpoint is selected and evaluated on the same bounded validation partition, only one seed is present, and the pilot was not designed for final inference.

## Blocking scientific findings

1. **Cited-evidence deletion is not faithful yet.** TRACE-SCV comprehensiveness is -0.0067. Removing its selected citations slightly increases, rather than decreases, the correct-label probability. It is less negative than TRACE-no-CF but does not establish decisive reliance on the citations.
2. **Visual evidence is currently harmful to the verdict.** TRACE-SCV clean verdict macro-F1 is 0.4937, while visual-removed macro-F1 is 0.5362. Text removal reduces macro-F1 to 0.4187, so the verifier is primarily text-dependent.
3. **Swap behavior has the wrong comparative direction.** Swap invariance is 0.9761 for TRACE-SCV versus 0.9321 for TRACE-no-CF. Because lower invariance means greater response to a decisive visual swap, the full model is less sensitive, contrary to H3.
4. **The strongest verdict baseline is all-pool.** Its verdict macro-F1 is 0.5593, 6.56 points above TRACE-SCV. The sparse bottleneck has not yet demonstrated that its attribution benefit preserves the best attainable verdict behavior.

## Confirmed implementation mismatches

- `training.py` trains decisive deletion by masking all gold evidence (`mask & ~gold`), whereas `evaluation/study.py` evaluates comprehensiveness by masking the model's selected citation set. This does not directly train the claimed selected-citation deletion behavior.
- Both training and evaluation visual-swap paths copy evidence position zero from the next batch row without requiring that the source item be visual or modality-matched. A visual node can therefore receive a text-feature vector while retaining its visual type.
- The external MuSciClaims path runs automatically in the formal profile although the current main Hugging Face release reports no license metadata. A separate unmerged pull-request card says CC BY 4.0, but that is insufficient under the study's own dataset gate. The external evaluation must be disabled unless the main release or authors provide authoritative reuse terms.

## Required correction before the formal run

1. Train the deletion loss on the hard selected citation mask used by inference, while retaining supervised citation selection as its own loss.
2. Make visual swaps source a valid visual item of the same modality from another claim in both training and evaluation.
3. Add the promised modality/missing-media and duplicate audit to the run manifest.
4. Disable MuSciClaims by default unless its reuse license is authoritatively confirmed.
5. Rerun the same validation-only pilot. Proceed to formal only if selected-citation comprehensiveness becomes positive, the matched verdict drop remains within one point, and citation macro-F1 remains above TRACE-no-CF.

No official-test value should be generated before this replacement pilot passes.
