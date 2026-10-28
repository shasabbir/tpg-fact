# TRACE-SCV pilot-output audit

Audit date: 2026-10-26  
Artifact: `trace-scv-pilot-outputs.zip`  
Archive size: 8,841,343 bytes  
SHA-256: `630c4e7a64a16b6e7cb7b569556b82960182c5adf4c5f56514f1dc56461a3f2b`

## Decision

The archive passes the engineering-execution check but does **not** constitute scientific validation. The recorded `scientific_results_validated: false` flag is correct. The run proves that the end-to-end training, checkpoint loading, inference, intervention evaluation, and aggregation paths execute on a Kaggle T4. It does not establish the paper's superiority hypothesis.

The formal run may proceed only with the revised export/resume safeguards recorded below. Pilot values must not be inserted into the manuscript as study results.

## Mechanical integrity

- The archive contains the aggregate summary, seven metric files, seven aligned prediction files, six distinct trained checkpoints, and six training histories.
- Every system produced exactly 300 predictions with 300 unique claim identifiers in identical order.
- All probability vectors are finite, lie in `[0,1]`, sum to one, and agree with the exported argmax label.
- No prediction contains duplicate citation identifiers. Every non-claim-only learned selector emits exactly three citations; all-pool emits a mean of 11.543 candidates.
- All six learned checkpoints have distinct SHA-256 hashes.
- The full model trained for 10 epochs after early stopping, used a Tesla T4, and recorded 58.5 MB peak allocated model-training memory. These figures exclude frozen feature extraction, which the hash pilot did not run.

## Dataset and design cautions

- The bounded v4 pilot selected the first 300 official-test rows. The resulting test slice contains 254 supported and 46 refuted claims (84.7% supported), so it is not a representative performance sample.
- The 300 claims come from 300 distinct source-paper groups. Group bootstrap therefore reduces to claim-level resampling in this pilot.
- The pilot used deterministic hash/text-caption features and downloaded no images. It is a software/resource diagnostic, not a valid test of the frozen BGE/CLIP multimodal method.
- Only one seed and 1,000 bootstrap samples were used. Seed-level sign-flip tests have no inferential value with one seed.
- The v4 ZIP omits gold labels, gold citations, source-paper groups, split manifests, feature hashes, effective configuration, and a full artifact inventory. Consequently, the metric files cannot be independently recomputed from this archive alone.

## Observed pilot metrics

| System | Verdict Macro-F1 | Citation Macro-F1 | Comprehensiveness | Sufficiency Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Majority | 0.4585 | 0.0000 | 0.0000 | 0.4585 | 0.3392 |
| Claim-only | 0.4271 | 0.0000 | 0.0000 | 0.4271 | 0.0523 |
| All-pool | 0.4702 | 0.3233 | 0.0468 | 0.4702 | 0.0081 |
| Relevance | 0.4702 | 0.5966 | -0.0193 | 0.4702 | 0.0286 |
| Flat selector | 0.5088 | 0.6968 | -0.0239 | 0.5088 | 0.0553 |
| TRACE-no-CF | 0.4420 | 0.7019 | 0.0088 | 0.4349 | 0.1087 |
| TRACE-SCV | 0.4139 | 0.7059 | 0.0389 | 0.4457 | 0.2364 |

On this bounded hash-feature slice, TRACE-SCV loses 9.49 absolute verdict-F1 points versus the flat selector while gaining only 0.91 citation-F1 points. The paired group-bootstrap interval for the verdict difference is `[-0.1455,-0.0402]`; the citation interval is `[-0.0002,0.0187]`. Relative to TRACE-no-CF, the verdict difference is -2.81 points and the citation difference is +0.40 points, with both intervals crossing zero. The prespecified superiority rule is therefore not met by this pilot.

The counterfactual direction is nevertheless non-degenerate: TRACE-SCV has positive cited-evidence deletion comprehensiveness (0.0389), whereas flat and relevance selectors are negative, and it is more responsive to cross-modal swaps. These are exploratory engineering observations only.

## Corrective implementation changes after audit

1. Every evaluation now exports `comparison_records.jsonl` with claim ID, source-paper group, gold label, predicted label, gold citations, and predicted citations.
2. Each output tree now contains `audit/run_manifest.json`, the executable study configuration, source/processed/feature/code hashes, runtime metadata, and `audit/artifact_inventory.json`.
3. Kaggle preflight output is persisted inside the exported audit directory.
4. Checkpoints are cryptographically bound to the exact train/validation feature stores and effective training configuration. A mismatched checkpoint now causes a hard failure instead of silent reuse.
5. Bounded pilots now use deterministic SHA-256 sampling rather than release-row prefixes.
6. Development records whose source-paper group appears in the official test set are excluded before the paper-grouped train/validation split. All train/validation/test group intersections are recorded and must be zero.
7. The full local test suite passes after these changes.

The corrected Kaggle upload is `trace-scv-kaggle-v5.zip` (62,735 bytes; SHA-256 `dc407e98c0fbb5d2199dc7163e54076347976bf7a7c59f535c90967d5f158490`). Its notebook was executed from a clean extraction in local smoke mode. The resulting export contained the preflight record, run manifest, configuration, comparison records, artifact inventory, checkpoints, predictions, and metrics.

## Scientific status

The pilot validates execution, not H1--H4. The formal BGE/CLIP, full-data, five-seed matrix remains the first admissible scientific result. Its exported archive must pass metric recomputation, group-isolation, completeness, finite-value, seed-matrix, and claim-evidence audits before any manuscript result is written.
