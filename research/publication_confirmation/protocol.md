# Publication Confirmation Protocol

Frozen: 2026-11-09  
Baseline commit: `251b419`  
Machine-readable source: `experiments/configs/publication_confirmation.yaml`

## Objective

Test whether an early-rank retrieval repair and a prespecified
retrieval-completeness-aware decision policy improve complete-gold evidence-set
F1 over the strongest current policy, calibrated thresholding, while preserving
verdict quality. The protocol also prepares—but does not authorize—the one-shot
formal-test or external evaluation required for independent confirmation.

## Confirmatory boundary

The existing MOCHEG validation results have already been inspected. They remain
development evidence. All new outputs are isolated under
`experiments/outputs/publication_confirmation/`. The official MOCHEG test
annotations remain inaccessible. No command in this protocol downloads, lists,
or reads the test repository.

Formal-test access requires all of the following:

1. a clean protocol-freeze commit;
2. a passing development-data seal audit;
3. frozen retrieval, calibration, selector, verifier, and statistical choices;
4. an immutable development result bundle and gate report;
5. explicit authorization from the authors for one test run.

## Data roles

Claims are grouped deterministically into fitting, calibration, model-selection,
and internal-holdout roles using the existing SHA-256 claim rule. The released
validation split is run once after train-only choices are fixed. Complete
released gold remains in each metric denominator, including evidence missed by
retrieval.

## Retrieval experiment

The earlier cross-encoder improved deep recall but damaged the first ranks. The
new bounded experiment fuses its rank with the dense--BM25 RRF rank instead of
replacing that rank. Text and image fusion weights are selected independently
from `{0.25, 0.50, 0.75}` using training-role packet-oracle F1, with operating
depth recall as a guardrail. Validation labels cannot select a fusion weight.

Every ranking is written and hashed before the gold join. The primary packet
remains 40 text plus 20 image items. If weighted fusion fails the advancement
gate, at most one separately frozen image-text extension may be attempted:
either OCR or article context, but not both. The extension cannot replace or
hide an unfavorable fusion result.

## Calibration and selection experiment

The six existing probability maps are retained. Calibration is fitted on the
calibration role and selected by model-selection Brier score. Fixed budgets,
calibrated thresholding, learned count, calibrated and uncalibrated expected-F1,
and one completeness-aware switch receive identical packets and scorer outputs.

The completeness-aware switch fits an L2 logistic model on training roles to
predict whether the packet contains the full released-gold set. It then chooses
expected-F1 for predicted-complete packets and calibrated thresholding
otherwise. Its feature list and threshold grid are fixed in the YAML protocol.
Any superiority claim is against calibrated thresholding, not fixed top-5.

## Verifier and statistics

All evidence policies share one stronger verifier. Verdict preservation passes
only when the lower bound of the paired 95% interval is above `-0.01`. The
primary evidence comparison uses claim-aligned paired bootstrap intervals with
10,000 resamples and Holm correction. Reachability strata are prespecified but
secondary.

## Stop and reporting rules

- A failed retrieval gate prevents a retrieval-improvement claim.
- A selector effect below `+0.010`, a non-positive corrected lower confidence
  bound, or a negative partition prevents method-superiority language.
- A verifier interval crossing `-0.01` prevents an end-to-end benefit claim.
- Negative and contradictory outcomes remain in the result bundle and paper.
- Validation results cannot unlock the formal test automatically.
- No public release or journal submission is authorized by this protocol.

