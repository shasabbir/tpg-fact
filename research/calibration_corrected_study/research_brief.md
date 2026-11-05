# Research brief

## Objective and reader

- Topic: calibrated evidence-set selection for retrieval-conditioned multimodal fact-checking.
- Task: given an image-text claim and a fixed packet of text and image candidates, return a concise subset matching the released complete-gold evidence identifiers.
- Unit of analysis: claim; candidate-level probabilities are intermediate quantities.
- Research question: when retrieval, scorer inputs, and evaluation claims are held fixed, does calibrated expected-F1 selection improve on fixed top-k and calibrated threshold policies?
- Contribution type: controlled empirical study with calibration, reachability, robustness, and verifier safeguards.
- Reader: an ML researcher unfamiliar with MOCHEG.
- Deliverable: canonical LaTeX manuscript in `paper/`, compiled PDF, evidence-linked reporting bundle, and readiness audit.

## Materials and evidence

| Material | Location | Purpose | State |
|---|---|---|---|
| Frozen protocol | `experiments/configs/calibration_corrected_study.yaml` | Data/model revisions, roles, grids, gates | checked |
| Selector records | `experiments/outputs/calibration_corrected_study/selector/` | Candidate probabilities and claim-level set metrics | checked by independent audit |
| Verifier records | `experiments/outputs/calibration_corrected_study/verifier/` | Shared-verifier comparison | checked by independent audit |
| Robustness records | `experiments/outputs/calibration_corrected_study/robustness/` | Controlled perturbations | checked by independent audit |
| Reporting source | `research/calibration_corrected_study/final_results.json` | Manuscript tables and figures | regenerated |
| Clean reproduction | `research/calibration_corrected_study/reproduction_manifest.json` | Exact comparison of core records | 20/20 exact |
| Reference audit | `research/calibration_corrected_study/reference_audit/` | DOI/year/type and ledger status | online metadata checked; human support ledger pending |

## Constraints and decisions

- Official MOCHEG test annotations remain sealed.
- Released validation is a controlled development population, not fresh external confirmation.
- The primary metric is claim-macro complete-gold evidence-set F1.
- All policies receive the same 40-text/20-image packet.
- The reachability-stratified diagnostic is post hoc and descriptive.
- The main paper does not use project-stage language.
- Strict workflow reference policy requires DOI-bearing sources from 2020 onward; indispensable older or software provenance is retained and flagged rather than misattributed.

## Current state

- The selector, verifier, and robustness studies are implemented and measured.
- Main evidence and the new reachability diagnostic regenerate from preserved records.
- Main scientific conclusion: calibration helps expected-F1 beat fixed top-5, but calibrated thresholding is stronger overall and verifier non-inferiority is not established.
- Next scientific priority: improve packet reachability and confirm the result on a fresh external or formal test population.
- Updated: 2026-11-04.
