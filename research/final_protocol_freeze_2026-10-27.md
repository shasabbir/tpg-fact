# DACS-SCV Formal Protocol Freeze

Freeze date: 2026-10-27

## Decision

The local pilot program is complete. The only remaining scientific execution is the frozen five-seed, full-data SciVer formal run on Kaggle. The official test has not been used for model or protocol selection.

## Retained method

DACS-SCV freezes the matched graph verifier and its complete evidence ranking, then predicts only citation-set cardinality from sorted selector probabilities and normalized valid/visual candidate counts. Fixed top-one, top-two, and top-three outputs use the same frozen ranking. The evaluator rejects a changed base-checkpoint hash, feature width, claim ordering, evidence ranking, or verdict record.

## Pilot basis

The corrected one-seed development validation pilot increased citation macro-F1 from 0.4851 for fixed top three to 0.5303 while preserving the matched verifier's verdict vectors. The 10,000-sample source-paper-grouped bootstrap interval for that difference was [-0.0089, 0.0991]. A post-hoc frozen-ranking diagnostic found fixed top one at 0.5236 and fixed top two at 0.5047; DACS-SCV's difference from the best fixed cutoff was only +0.0067, with a paired interval of [-0.0077, 0.0219]. The pilot therefore supports feasibility only and motivates the stricter fixed-budget-controlled formal test.

## Local verification

- 37 unit and integration tests pass, including referenced-release-file selection, bounded HTTP 429 retry behavior, and alpha-aware image preprocessing.
- A clean smoke pipeline exports the controlled verifier baselines, fixed top-one/top-two/top-three outputs, DACS-SCV, and its two packet-count ablations.
- The exact v3 upload ZIP was rediscovered from a renamed archive in an isolated nested read-only input tree, extracted and installed into a separate writable tree, version-checked as `0.3.2`, and executed end to end. All 11 smoke variants completed and the exported archive contained a 56-file artifact inventory.
- The IEEE manuscript compiles to eight pages with 50 peer-reviewed references and no unresolved citation, reference, or overfull-box diagnostic. Every rendered page was visually checked.

## Final upload identity

- File: `dacs-scv-kaggle-v3.zip`
- Size: 98,208 bytes
- File count: 55
- SHA-256: `8B3A3B36F7803EC13E0A91ED669663D7279FAF87C34D149D5226F48F8CBE8EE8`
- Required package version: `0.3.2`

## Operational amendment (2026-10-28)

The first v1 Kaggle attempt stopped before dataset preparation because Hugging Face returned HTTP 429 from its Xet read-token endpoint for Kaggle's shared IP. No feature extraction, fitting, model selection, or official-test inference occurred, so the scientific freeze remains intact. Version 2 changes only execution reliability: it requires an enabled private `HF_TOKEN`, disables Xet, downloads only files referenced by the pinned 3,000-row release, bounds Hub concurrency and retries, and persists the data, features, checkpoints, and outputs under `/kaggle/working` so interrupted same-session runs can resume. The dataset revision, split rules, model, seeds, comparisons, metrics, and success rule are unchanged.

The second attempt stopped at the unchanged release gate before feature extraction. Local reproduction showed a false exact-perceptual overlap: three distinct `LA` figures with different raw hashes and content collapsed to zero dHash because alpha was discarded before grayscale conversion. Version 3 composites transparent images onto white before both dHash auditing and CLIP encoding. The full pinned 804/196/2,000 train/validation/test reconstruction then passes the unchanged group, exact-visual, and gold-content gates. This preprocessing correction does not alter data rows, splits, labels, model architecture, seeds, comparisons, metrics, or the success rule.

## Formal success rule

The manuscript may claim adaptive citation-set superiority only if the five-seed official-test DACS-SCV effect is positive against each prespecified graph-fixed top-one, top-two, and top-three rule, with all three source-paper-grouped bootstrap intervals reported and Holm correction applied to this comparison family. Every paired verdict probability and verdict prediction must be identical. A non-positive effect against any fixed budget is reported as a failed adaptive method; a subgroup-only effect is narrowed to the supported subgroup.

## Next action

Keep the Kaggle secret `HF_TOKEN` enabled, upload only the frozen v3 ZIP, remove every older source attachment, enable Internet and a GPU, import the included notebook, and choose **Run All** without changing the formal profile. Confirm that the first cell reports `hf_authenticated: True` and the install cell verifies package `0.3.2`. Return `dacs-scv-formal-outputs.zip` unchanged for independent audit before editing the results section.
