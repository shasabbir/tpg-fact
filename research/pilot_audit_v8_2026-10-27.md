# TRACE-SCV v8 Pilot Audit

Audit date: 2026-10-27  
Archive: `trace-scv-pilot-outputs.zip`  
Archive size: 13,292,553 bytes  
SHA-256: `a733f4727bfafab3761de85bd00e3047b8b51eacdae293827f1d5ced36bc8f3d`

## Decision

**The archive is mechanically valid but scientifically invalid. Do not use its metrics in the manuscript and do not start the formal run from v8.**

The exported data-release gate correctly failed because six exact evidence-content fingerprints involving gold evidence appeared in both the 600-row training split and the 150-row validation split. Item-level duplicate records led to a deterministic parser defect: when an annotation requested a subsection such as `4.3`, the parser encountered and returned its parent section `4` first. SciVer parent rows commonly contain a heading but no body text. The normalization step therefore produced repeated generic evidence such as `Experiments`, creating artificial cross-split duplicates and training/evaluation inputs that did not represent the annotated evidence.

This is a data-construction error, not evidence that the method passed or failed its scientific hypothesis. The prespecified mechanism decision is deferred to one corrected v9 validation run. The release gate remains unchanged.

## Mechanical integrity checks

- ZIP structure: 59 unique, safe entries; no duplicate or path-traversal members.
- Artifact inventory: all 37 declared artifacts exist and match their recorded byte counts and SHA-256 hashes.
- Code provenance: all 26 recorded source hashes match the audited v8 workspace.
- Checkpoints: all six trained variants have input fingerprints that independently recompute from the recorded feature hashes and training configurations.
- Evaluation alignment: every variant contains 150 unique validation claims with identical claim, group, gold-label, and gold-citation identities.
- Predictions: exported predictions agree with the corresponding per-claim comparison records.
- Probabilities: all nine clean/intervention distributions are finite and sum to one.
- Metrics: verdict macro-F1, citation macro-F1, selected-citation comprehensiveness, and swap sensitivity independently recompute exactly for every variant.
- Scope: one seed, development validation only; the official test contains zero evaluated rows.

## Invalid diagnostic values

These values describe the faulty v8 preprocessing and are retained only to make the audit reproducible:

| Variant | Verdict macro-F1 | Citation macro-F1 | Comprehensiveness | Swap sensitivity |
|---|---:|---:|---:|---:|
| TRACE-no-CF | 0.5528 | 0.5829 | -0.0066 | 0.0282 |
| TRACE-SCV | 0.5572 | 0.5909 | -0.0036 | 0.0322 |

The readiness gate failed selected-citation comprehensiveness and the data-release gate. Its apparent TRACE-SCV improvements over TRACE-no-CF were +0.0043 verdict macro-F1, +0.0080 citation macro-F1, +0.0030 comprehensiveness, and +0.0040 swap sensitivity. None is a usable result because the gold evidence was incorrectly constructed.

## Root-cause confirmation

Pinned release records confirm that the requested exact children contain substantive text while the incorrectly returned parents are empty. Examples include:

- paper `2409.20332v1`: requested `4.3`, returned `4`; exact child is `4.3. Synthetic Volumes for Self-Supervised Pretraining`;
- paper `2409.16745v1`: requested `2.2`, returned `2`; exact child is `Optical polarization`;
- paper `2411.02949v1`: requested `3.1`, returned `3`; exact child is `Performance measures`;
- paper `2409.07703v1`: requested `3.2.1`, returned `3`; exact child is `Data Analysis`.

## Corrective action and verification

Version 9 (`0.2.1`) now:

1. searches for an exact section identifier before considering descendants;
2. never maps a precise child request upward to a generic parent;
3. inserts a gold abstract fallback before distractors only when no annotated gold evidence is recoverable;
4. reports the number of fallback gold items in every split audit; and
5. adds regression tests for exact-child precedence and missing-annotation fallback.

All 29 automated tests pass. A fresh local reconstruction of the identical 600/150 pilot selection downloaded all 363 selected paper records and all 357 referenced image files. It produced 165/165 recoverable training visual items and 44/44 validation visual items, zero fallback gold items, zero train/validation exact content overlap, zero exact or near-perceptual visual overlap, zero group overlap, and a passing data-release gate.

## Next authorized run

Run the v9 pilot once on Kaggle. Continue to the formal five-seed matrix only if the returned archive passes artifact integrity, data release, citation/verdict, counterfactual, and visually grounded gates. If the corrected pilot still fails the causal/visual gates, retire the causal-grounding claim and pivot as specified in `plan.md`.

Prepared runner: `trace-scv-kaggle-v9.zip`, 75,422 bytes, SHA-256 `7595a825ffdf4bc8174e57fd6f3276b31687f1a05dfc0e5fa9b8709fc5c769e6`.
