# Manuscript readiness assessment

## Overall assessment

The manuscript now presents a coherent and reproducible controlled development study. Its strongest contribution is diagnostic: retrieval reachability, probability calibration, set decoding, and verdict preservation are measured separately. The implementation evidence is strong enough for the bounded claims in the paper. The study is not ready for an end-to-end superiority claim or a final journal submission because external confirmation, reference support review, and author declarations remain incomplete.

## Findings

| Location | Severity | Issue | Consequence | Evidence | Repair | Recheck/status |
|---|---|---|---|---|---|---|
| Results and conclusion | Major | No fresh external or formal test confirmation | Development gains may not generalize beyond released MOCHEG validation | Test labels sealed; prior development use disclosed | Evaluate on a fresh external or formal test population | open |
| Results VI-D | Major | Verifier non-inferiority fails narrowly | Evidence-set improvement is not an established end-to-end system improvement | 95% interval [-0.01053, 0.01279] against -0.01 margin | Stronger prespecified verifier study | open and disclosed |
| Results VI-A | Major | Retrieval ceiling is low | Most selector errors cannot be repaired by decoding alone | Packet oracle 0.3560; complete-gold coverage 4.39%; 3,086 zero-reachable claims | Improve text early ranking and image retrieval; rerun full chain | open and central to paper |
| Reachability diagnostic | Major | Complete-packet subgroup is post hoc and small | The 0.5240 expected-F1 result may be unstable or dataset-specific | 79 complete-reachability validation claims | Prespecify and replicate with uncertainty | open and disclosed |
| Bibliography | Major | Human claim-support ledger is incomplete; three cited records conflict with the strict DOI/year policy | Reference readiness cannot be certified even though the expanded literature coverage and most DOI metadata are in place | Online audit in `reference_audit/audit.md`: 52 pending, 3 excluded | Inspect source passages and notices; resolve or document the RRF, calibration, and MMCV conflicts | open |
| Experimental design | Minor | End-to-end hardware, latency, and energy are not fully recorded | No compute-efficiency claim can be made | Software environment recorded; full retrieval cost absent | Record hardware and measurement boundaries if such a claim is needed | open and disclosed |
| Declarations | Major | Funding, conflicts, contributions, and venue-specific statements lack author-approved facts | Submission package is incomplete | No verified declarations supplied | Authors provide facts after venue selection | open |
| Source and PDF | Resolved | The compact revision reduced literature, comparison, and result depth too aggressively | It no longer matched the quality of the stronger reference manuscript | Restored a 55-reference technical review, ten-row comparison, full retrieval matrix, partition results, formal equations, and absolute robustness table; rebuilt as 13 balanced pages | Rebuild and inspect all pages | resolved and rechecked |

## Requirement coverage

| Requirement | Status | Evidence |
|---|---|---|
| Concrete task, gap, mechanism, tests, findings, bounded implication | present | title, abstract, Introduction, Results, Discussion |
| Method/code/configuration agreement | present | frozen configuration, candidate records, independent audits |
| Main effects with claim-aligned uncertainty | present | selector and verifier effect tables |
| Negative and inconclusive results | present | learned count failure, zero penalties, verifier safeguard failure, probability reversals |
| Useful editable figures | present | TikZ architecture and reporting figure builders |
| Clean deterministic reproduction | present | 20/20 exact core artifacts |
| External confirmation | missing | formal test sealed; external transfer proposed only |
| Reference identity and claim-support review | weak | registry metadata checked; ledger pending |
| Venue declarations and final submission rules | missing | target venue and author facts unresolved |

## What the results actually support

- Prior-aware calibration is necessary for the tested adaptive probability-based selectors.
- Expected-F1 is a precision-oriented alternative to fixed top-5 and improves mean evidence F1 by 0.00851.
- Calibrated thresholding is the strongest tested policy overall.
- Expected-F1 is most effective when retrieval already contains complete released gold; it does not improve over fixed top-5 in the partial-reachability stratum.
- Extra size and redundancy penalties have no measured benefit.
- Verdict preservation and generalization beyond this development population remain unestablished.
