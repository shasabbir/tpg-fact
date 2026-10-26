# Living Study Plan

## Document control

- **Study:** Decision-Time Provenance for Multimodal Fact-Checking: Independent Evidence, Temporal Validity, and Calibrated Deferral
- **Short name:** DTP-Fact / US-TPMFC
- **Primary venue:** IEEE Transactions on Computational Social Systems (TCSS)
- **Principal author:** Sabbir Hossain
- **Institution:** Department of Computer Science, American International University-Bangladesh, Dhaka, Bangladesh
- **Email:** 26-94030-1@student.aiub.edu
- **Plan status:** Active; the final-run software path is implemented and mechanically checked, while M2 manual screening/evidence audit remains the scientific gate
- **Last updated:** 2026-10-24

This file is the single source of truth for the study. It must be read before substantive data, model, evaluation, or manuscript work. After every material change, update the applicable checklist, decision/deviation log, evidence status, and next actions here. A change to the method invalidates dependent results until the affected experiments are rerun.

Use these evidence-state labels consistently:

- **Proposed:** specified here but not implemented.
- **Implemented:** present in code or manuscript and mechanically checked.
- **Measured:** produced by a preserved real-data experiment artifact.
- **Checked:** independently reconciled against the originating data, configuration, code, and output.

Synthetic smoke-test values are software diagnostics only. They are never scientific results.

## 1. Research objective

Create a full IEEE TCSS-style study titled:

> **Decision-Time Provenance for Multimodal Fact-Checking: Independent Evidence, Temporal Validity, and Calibrated Deferral**

The system will classify US-focused image-text claims as:

- `supported`
- `refuted`
- `insufficient_evidence`

The central research question is:

> Does decision-time-valid, independently corroborated evidence improve future-event multimodal fact-checking and selective risk compared with matched document-level RAG?

The central test compares the full temporal-provenance system with a matched vanilla multimodal RAG system under a future-event, event-disjoint evaluation.

### Planned contributions

1. A decision-time evidence protocol that requires a documented pre-cutoff observation and audits timestamp confidence.
2. Source-dependence-aware corroboration that bounds the contribution of copied and syndicated documents without treating dependence as credibility.
3. A sufficiency-first three-way detector with calibrated selective deferral, where sufficiency supervision is separately audited.
4. Counterfactual leakage and duplication interventions under chronological, event-disjoint evaluation.

### Scope boundaries

- The study is US-specific and will not claim global generalization.
- “Provenance” means evidence origin and source dependence, not only human-versus-AI authorship.
- Source dependence is distinct from source credibility; credibility is not a primary-model weight.
- Professional fact-checker verdicts are inherited labels; the project does not claim new human annotation.
- Automatically extracted evidence links are silver evidence.
- `claim_time` is the earliest observed public appearance of the claim. `decision_cutoff` is the fact-check publication time or a documented equivalent.
- The 2026 test period is a partial-year snapshot ending at the first successful frozen candidate collection, `2026-10-24T07:53:33.754563Z`, and must be reported separately from 2025; the earlier planned date of 2026-10-21 was not a completed data snapshot.
- Q1 publication is an objective, not a guaranteed outcome.
- No result may be described as measured until its raw artifact and configuration are preserved.

## 2. Project structure and formats

All project work belongs under `D:\1.msc\ET2`. The `paper` directory is reserved for the manuscript and files directly supporting its compilation.

```text
ET2/
├── plan.md
├── README.md
├── research/
│   ├── literature_matrix.csv
│   ├── reference_ledger.csv
│   └── claim_evidence_register.csv
├── experiments/
│   ├── src/
│   │   ├── data/
│   │   ├── retrieval/
│   │   ├── provenance/
│   │   ├── models/
│   │   ├── training/
│   │   └── evaluation/
│   ├── configs/
│   ├── notebooks/
│   │   ├── 01_data_audit.ipynb
│   │   ├── 02_retrieval_analysis.ipynb
│   │   ├── 03_error_analysis.ipynb
│   │   └── 04_tables_and_figures.ipynb
│   ├── tests/
│   ├── scripts/
│   ├── outputs/
│   ├── pyproject.toml
│   └── README.md
└── paper/
    ├── paper.tex
    ├── supplement.tex
    ├── references.bib
    ├── sections/
    │   ├── 01_introduction.tex
    │   ├── 02_related_work.tex
    │   ├── 03_dataset.tex
    │   ├── 04_method.tex
    │   ├── 05_experiments.tex
    │   ├── 06_results.tex
    │   ├── 07_discussion.tex
    │   └── 08_conclusion.tex
    ├── figures/
    ├── tables/
    └── output/
        └── pdf/
```

### LaTeX policy

- Use `IEEEtran` in two-column journal mode.
- Use BibTeX with the `IEEEtran` bibliography style.
- Keep prose in section files and control the manuscript from `paper.tex`.
- Generate tables and figures from experiment outputs; do not manually copy result values.
- Compile and visually inspect both the paper and supplement after material changes.
- Target approximately ten main-paper pages, subject to current IEEE TCSS author guidance.
- Keep proposed, implemented, and measured claims visibly distinct while experiments are incomplete.

### Python and notebook policy

Use both `.py` and `.ipynb`:

- `.py` files contain all reusable data processing, retrieval, graph construction, training, metrics, and plotting functions.
- Notebooks import those modules and are limited to exploration, auditing, error analysis, and presentation.
- No core model or metric may exist only inside a notebook.
- Training and five-seed evaluations must run through command-line scripts.
- Every notebook must execute from top to bottom in a clean environment.
- Configuration files hold paths, model revisions, seeds, split identities, and hyperparameters.
- Notebook-only implementation is excluded because it makes multi-GPU execution, repeated experiments, testing, and reproducibility fragile.

## 3. Data and system design

### 3.1 US-TPMFC benchmark

Collect English ClaimReview cases concerning US events from professional fact-checkers. Cover 2020-2026 politics, disasters, health, technology, conflict, public safety, and major social events.

Retain only image-text claims with:

- an identifiable earliest timestamp;
- a recoverable image or archive;
- an unambiguous professional rating;
- sufficient metadata for temporal filtering.

Map only explicit ratings:

- true/correct → `supported`;
- false/fake/miscaptioned → `refuted`;
- unproven/unsubstantiated/no evidence → `insufficient_evidence`.

Exclude mixed, partially true, satire, opinion, and ambiguous ratings.

Use open sources only:

- ClaimReview for claim and verdict metadata;
- GDELT for evidence discovery;
- Common Crawl for archived page recovery and timestamps.

Exclude fact-check articles and post-cutoff documents from the evidence pool.

### 3.2 Temporal and group splits

- Train: 2020-2023.
- Validation: 2024.
- Test: 2025-2026.
- Keep the same event, near-duplicate claim, and perceptually identical image in one partition.
- Add rolling-origin evaluation as a secondary analysis.
- Fit preprocessing, calibration, thresholds, and model selection using training/validation data only.

The benchmark contribution proceeds only if the filtered corpus contains at least:

- 2,000 cases;
- 300 cases per label;
- 50 final-test events;
- two independent evidence clusters for at least 70% of retained claims.

If these thresholds fail, train on existing datasets and use the new corpus only as a temporal/provenance stress test. Record this as a scope change, not as the original benchmark contribution.

### 3.3 Public record types

Claim record:

```text
claim_id, claim_text, claim_image_url, image_sha256, image_phash,
claim_time, decision_cutoff, fact_check_url, source_rating,
normalized_label, event_id, evidence_ids, cutoff_basis, cutoff_confidence,
sufficiency_target, sufficiency_target_source
```

Evidence record:

```text
evidence_id, url, domain, title, published_at, archived_at,
text_sha256, image_phash, media_type, source_cluster_id,
outgoing_links, retrieval_query, timestamp_confidence, timestamp_source,
dependency_signals
```

Prediction record:

```text
claim_id, predicted_label, class_probabilities,
sufficient_evidence_score, selected_evidence_ids,
source_cluster_ids, decision_cutoff
```

Release code, URL/hash metadata, graph annotations, retrieval logs, and split manifests without redistributing copyrighted articles or images.

For the primary analysis, evidence must have both a publication time at or before the decision cutoff and an independent archive or crawl observation at or before the cutoff. Timestamp confidence is recorded as `confirmed`, `observed`, or `uncertain`. Uncertain items are excluded from the primary analysis and admitted only in a named sensitivity analysis. A post-cutoff crawl does not prove pre-cutoff availability.

Site-specific rating dictionaries are frozen and audited rather than inferred with one global keyword rule. The sufficiency target is not derived automatically from the professional verdict. A manually audited subset supplies gold sufficiency labels; any automatically derived labels are reported as silver and evaluated separately.

### 3.4 Retrieval and provenance model

1. Extract entities, dates, OCR text, and image captions.
2. Retrieve text using BM25 and BGE-M3.
3. Retrieve images using SigLIP2 and perceptual hashes.
4. Fuse rankings with reciprocal-rank fusion.
5. Rerank 100 candidates and retain the top 20 time-valid items.
6. Construct a heterogeneous temporal-provenance graph.
7. Cluster copied and syndicated evidence using exact hashes, MinHash, semantic similarity, image pHash, hyperlinks, and publication order; validate clusters on an audited sample.
8. Encode typed claim, document, image, publisher, and cluster relations with a relation-aware graph encoder.
9. Aggregate independent source clusters rather than raw document counts, with each cluster contributing bounded total mass.
10. Predict evidence sufficiency first and veracity second.
11. Apply validation-fitted temperature scaling and an abstention threshold targeting 80% coverage.

Qwen2.5-VL-7B-Instruct will serve as an open multimodal comparison and optional query generator, not as an unverified label source. Record the exact model revision, prompt, decoding configuration, and execution date when used. The graph architecture is an implementation choice, not the principal novelty; if a simpler cluster-aware aggregator matches it, report that result and narrow the method claim.

## 4. Experiment and testing plan

### 4.1 Baselines

1. Majority-class baseline.
2. Text-only encoder.
3. Image-text fusion without retrieval.
4. Uncertainty-aware image-text fusion, following MDF-FND where reproducible.
5. Event-aware graph model without external retrieval, following Event-Radar where reproducible.
6. Multimodal LLM zero-shot and few-shot.
7. Vanilla multimodal RAG.
8. Document-level evidence summarization or aggregation.
9. Original-context prediction, following COVE/5Pils where compatible.
10. Time-filtered RAG without provenance.
11. Provenance-aware RAG without temporal filtering.
12. RED-DOT where its implementation is compatible.
13. Full decision-time provenance model.

All controlled baselines receive the same evidence corpus, splits, metrics, and 20-trial validation tuning budget. Any unavoidable difference in pretraining, information access, or implementation must be disclosed.

### 4.2 Metrics

Primary:

- Macro-F1 on the 2025-2026 event-disjoint test set.

Secondary:

- Per-class F1 and balanced accuracy.
- Macro one-vs-rest AUROC.
- Brier score, negative log-likelihood, and ECE.
- AURC and risk at 80% coverage.
- Retrieval Recall@10 and nDCG@10.
- Runtime, GPU memory, and retrieved-document count.

### 4.3 Ablations

Remove individually:

- temporal filtering;
- source-dependency clustering;
- temporal graph features;
- graph encoder, replaced by a parameter-matched cluster-aware pooling model;
- text evidence;
- visual evidence;
- evidence-sufficiency gate;
- evidence-dropout training.

### 4.4 Robustness scenarios

- Inject 1, 5, and 10 syndicated copies of one source.
- Add post-cutoff documents containing the verdict.
- Remove the earliest source.
- Introduce conflicting dependent and independent evidence.
- Swap captions and reuse historical images.
- Evaluate by topic and chronological window.
- Report 2025 and partial-2026 results separately and jointly.

### 4.5 Reproducibility and statistical tests

- Run five seeds only after configuration selection.
- Use 10,000 paired event-level bootstrap samples for 95% confidence intervals.
- Apply Holm correction to primary comparison families.
- Unit-test timestamp filtering, split isolation, duplicate clustering, label mapping, metric aggregation, and abstention.
- Integration-test one complete miniature run from raw records to exported predictions.
- Execute notebooks automatically as part of the final reproducibility check.
- Generate every manuscript table and figure from saved result artifacts.
- Preserve raw predictions, retrieval traces, run logs, failures, exclusions, and configuration identities.

Drop any claim whose corresponding test fails. If the graph does not improve detection, retain the benchmark and leakage analysis only if provenance processing measurably improves duplicate robustness or selective risk.

## 5. Claim-test map

| Intended claim | Required comparison/control | Measurement | Independent unit and uncertainty | Falsifier / consequence |
|---|---|---|---|---|
| Decision-time filtering prevents future-information leakage and improves valid future-event classification | Full model vs. matched vanilla RAG and provenance-only RAG | Test macro-F1; post-cutoff verdict intervention | Event-level paired bootstrap, five seeds | No improvement outside leaked setting → retain leakage audit only and drop performance claim |
| Source-dependency modeling prevents duplicate evidence inflation | Full model vs. no-clustering ablation under 1/5/10 copy injection | Macro-F1, confidence shift, Brier, selective risk | Event-level paired comparisons | Similar degradation or no robustness benefit → drop/narrow provenance claim |
| Sufficiency-first prediction improves handling of unavailable evidence | Full model vs. flat three-way classifier and no-gate ablation on the audited sufficiency subset | Insufficient-evidence F1, macro-F1, Brier | Event-level paired bootstrap | No benefit or worse supported/refuted performance without selective-risk gain → remove claim |
| Calibrated abstention reduces operational risk | Temperature-scaled threshold vs. uncalibrated confidence at matched coverage | AURC and risk at 80% coverage | Event-level bootstrap; validation-selected threshold | Selective risk does not improve at matched coverage → remove calibrated-abstention claim |
| Multimodal evidence is necessary | Full model vs. text-only and visual-only evidence ablations | Macro-F1 and topic/window breakdown | Event-level paired bootstrap | One modality matches full model within uncertainty → narrow multimodal necessity claim |
| US-TPMFC supports benchmark-level conclusions | Corpus passes all preregistered thresholds and audit | Counts, class balance, event counts, evidence-cluster coverage | Dataset records grouped by event | Any threshold fails → release/use only as stress test |

## 6. Paper workflow and manuscript structure

### Main sections

1. Introduction
2. Related Work
3. Dataset and Task
4. Methodology
5. Experimental Setup
6. Results
7. Discussion
8. Conclusion

Keep robustness, calibration, error analysis, ethical considerations, and limitations as conventional subsections within Results or Discussion. Do not invent decorative section names.

### Reporting discipline

- Keep one contribution list.
- Explain the task, temporal cutoff, source dependence, sufficiency, calibration, and abstention before relying on those terms.
- Keep inherited labels and silver evidence explicit.
- Report negative and inconclusive results relevant to the claims.
- Reconcile every manuscript value with raw outputs before retaining it.
- Do not compare rerun controlled results with literature-reported numbers as if they used the same protocol.
- Verify TCSS scope, author instructions, page policy, and current quartile from an identified ranking database immediately before submission.

## 7. Milestones and gates

Status markers: `[x]` complete, `[~]` in progress/partial, `[ ]` not started, `[!]` blocked or failed.

### M1 — Project and protocol scaffold

- [x] Create LaTeX and experiment directory structure.
- [x] Create the IEEEtran manuscript and supplementary protocol.
- [x] Record author information.
- [x] Create reusable Python modules and notebook shells.
- [x] Create configuration, literature, reference, and claim-evidence registers.
- [x] Add synthetic end-to-end smoke test.
- [x] Add automated tests and clean notebook execution.
- [x] Add five-seed command orchestration and prediction evaluation.
- [x] Add editable architecture and temporal-split figures plus a claim-to-test matrix.
- [x] Document environment setup, smoke checks, pilot ingestion, seeded runs, evaluation, notebooks, and LaTeX compilation.
- [x] Compile and visually inspect the current paper and supplement.
- [x] Audit all 17 supplied reference PDFs and record venue-priority caveats.
- [x] Revise the novelty position around decision-time validity and independent corroboration.
- [x] Implement timestamp-confidence checks, cluster-normalized weights, feasibility auditing, deterministic evidence interventions, and optional dense retrieval adapters.

**Gate:** Passed for the software/protocol scaffold only. This gate does not establish scientific readiness.

### M2 — 200-claim feasibility pilot

- [x] The initial professional fact-checker allowlist is frozen in `configs/base.yaml`, and the first authenticated candidate snapshot is frozen at `2026-10-24T07:53:33.754563Z`.
- [x] Implement paginated API collection, untouched raw-response preservation, candidate-review validation, verified image download/hashing, GDELT candidate discovery, Common Crawl observation checking, and a pilot-specific assessment.
- [x] Collect and validate a broad screening pool: 32,898 pending candidates, 440 raw API pages, and no review-schema errors.
- [x] Generate a deterministic 1,200-row screening queue balanced at 400 candidates per suggested label; preserve its input/output hashes and manifest.
- [~] Screen the queue until at least 200 eligible image--text claims are retained.
- [ ] Run explicit-label mapping and exclusion audit.
- [ ] Verify timestamps, image recoverability, and archive availability.
- [ ] Assign `confirmed`, `observed`, or `uncertain` timestamp confidence with supporting provenance.
- [ ] Create and audit gold evidence-sufficiency labels; do not infer them only from verdict strings.
- [ ] Audit 200 records manually for extraction quality.
- [ ] Estimate label balance, event counts, independent-source coverage, runtime, storage, and failure rates.
- [ ] Decide whether the 2,000-case benchmark thresholds are feasible.

**Gate:** Do not scale collection or train a main model until feasibility, access, and temporal metadata quality are documented.

### M3 — Corpus and split freeze

- [ ] Complete eligible corpus collection.
- [ ] Freeze inclusion/exclusion rules.
- [ ] Freeze record IDs, hashes, event groups, and decision cutoffs.
- [ ] Freeze train/validation/test manifests.
- [ ] Pass event, near-duplicate claim, and image-overlap tests.
- [ ] Report original and post-filter counts.
- [ ] Decide benchmark contribution vs. stress-test fallback using preregistered thresholds.

### M4 — Retrieval and provenance freeze

- [ ] Freeze evidence corpus and archive snapshot.
- [ ] Exclude fact-check domains and post-cutoff evidence.
- [ ] Freeze BM25, BGE-M3, SigLIP2, pHash, fusion, reranking, and top-k settings.
- [ ] Validate dependency clusters on an audited sample.
- [ ] Report pairwise precision, recall, and F1 for dependency clustering.
- [ ] Freeze graph schema, temporal features, and unknown-origin treatment.
- [ ] Measure retrieval Recall@10/nDCG@10 and cluster-quality diagnostics.

### M5 — End-to-end model pilot

- [~] BGE-M3, reranker, and SigLIP2 adapters are implemented; execute and profile them on the real pilot.
- [x] Implement batched heterogeneous-graph and matched feature-model training, validation selection, checkpointing, calibration, prediction export, and runtime metadata.
- [~] Verify objective, gradients, missing-evidence cases, saving/loading, inference, and metrics on a tiny CPU diagnostic; repeat on the real pilot.
- [ ] Measure pilot runtime and peak GPU memory.
- [x] Mechanically verify temperature fitting and 80% coverage threshold selection on validation only; measure their behavior on the real pilot.

### M6 — Baseline tuning and freeze

- [x] Implement the majority baseline and the controlled text, image--text, vanilla-RAG, time-filtered-RAG, provenance-pooling, no-graph, and graph variants defined in the frozen configuration.
- [ ] Give each baseline the same 20-trial validation budget.
- [ ] Record model revisions, prompts, preprocessing, hardware, and deviations.
- [ ] Freeze configuration-selection rules before final-test evaluation.

### M7 — Main experiments

- [ ] Run five seeds for the full model and controlled baselines.
- [ ] Run all ablations.
- [ ] Run temporal-leakage and duplicated-source stress tests.
- [ ] Run topic and chronological-window analyses.
- [ ] Generate event-level confidence intervals and Holm-adjusted comparison results.
- [ ] Preserve raw predictions, checkpoints, logs, and resource measurements.

### M8 — Evidence and manuscript audit

- [ ] Update the claim-evidence register from actual outputs.
- [ ] Regenerate every table and figure from preserved artifacts.
- [ ] Audit references, cited versions, claim support, and correction/retraction notices.
- [ ] Reconcile abstract, results, discussion, supplement, tables, and raw outputs.
- [ ] Remove or narrow claims that fail their tests.
- [ ] Complete ethical considerations, limitations, and release documentation.

### M9 — Submission package

- [ ] Expand the manuscript to the evidence-supported journal version.
- [ ] Recheck current TCSS author instructions and page policy.
- [ ] Verify current JCR or Scopus quartile with database/category/year/date recorded.
- [ ] Build from a clean environment.
- [ ] Execute all notebooks and tests from a clean environment.
- [ ] Compile and visually inspect paper and supplement.
- [ ] Verify archive contents, licenses, links, hashes, and README commands.
- [ ] Prepare—but do not submit without explicit authorization—the final package.

## 8. Current implementation and evidence status

### Implemented and mechanically checked

- IEEEtran protocol manuscript and supplement.
- Record schemas for claims, evidence, and predictions.
- Explicit rating normalization and ambiguous-label rejection.
- Conservative decision-time filtering with timestamp-confidence tiers and an uncertain-date sensitivity mode.
- Event/split leakage checks.
- BM25 ranking and reciprocal-rank fusion scaffold.
- Duplicate/source clustering, explicit dependence signals, cluster-normalized weights, and provenance graph construction.
- Deterministic duplicate, post-cutoff verdict, earliest-source, dependent/independent conflict, caption-swap, and historical-image-reuse interventions with frozen-checkpoint inference, per-seed predictions, change-rate summaries, and generated manuscript tables.
- Configured BGE-M3, BGE reranker, and SigLIP2 adapters with pinned-revision fields.
- Frozen retrieval runner that ranks the shared corpus with BM25, BGE-M3, and available SigLIP2 features, applies reciprocal-rank fusion and the pinned BGE reranker, filters by decision time, retains the top 20, and reports split-specific Recall@10/nDCG@10.
- Exact public Hugging Face commit revisions frozen on 2026-10-24 for BGE-M3, BGE reranker v2-m3, SigLIP2 SO400M, and Qwen2.5-VL-7B-Instruct.
- Feasibility-gate audit command using the preregistered corpus thresholds.
- Hierarchical smoke classifier and trainable graph-model components.
- Temperature scaling and abstention-threshold utilities.
- Classification, calibration, selective-risk, retrieval, AUROC, bootstrap, and Holm utilities.
- ClaimReview ingestion and open-archive discovery entry points.
- Paginated Google Fact Check API collection with API-key redaction, raw-response preservation, deterministic candidate IDs, deduplication, and year filtering.
- First frozen ClaimReview candidate snapshot: 32,898 deduplicated 2020--2026 rows, 440 preserved raw API pages, seven configured publisher filters, and a key-free collection manifest timestamped `2026-10-24T07:53:33.754563Z`.
- UTF-8 candidate-audit CSV generation and strict keep/reject validation.
- Private target-image download with exact SHA-256 and perceptual hashing.
- Batch GDELT evidence-candidate collection with raw response and failure manifests.
- Batch Common Crawl pre-cutoff observation checks with queried-index audit trails.
- A pilot-specific assessment that separates 200-case completion and projected collection burden from the final 2,000-case benchmark gate.
- A machine-readable full-run preflight that blocks training until API access, pilot data/evidence, the pilot gate, model revisions, a real trainer, PyTorch, and CUDA are present.
- Frozen-dataset writer with record, evidence-link, split-completeness, event-leakage, and image-overlap checks plus SHA-256 manifests.
- Frozen BGE-M3/SigLIP2 feature-store builder with pinned revisions and input/output hashes.
- Batched heterogeneous-graph trainer and parameter-matched feature baselines with class weighting, evidence dropout, AdamW, cosine scheduling, early stopping, checkpoint hashes, runtime metadata, and prediction export.
- Validation-only deterministic 20-trial selection, a five-seed baseline/ablation matrix, majority baseline, per-seed evaluation, and resume-safe orchestration.
- Paired event-level bootstrap comparisons, two-sided bootstrap p-values, Holm adjustment, and checked LaTeX table generation from preserved outputs.
- Deterministic 1,200-row screening queue with 400 candidates per suggested label, 652 train-period, 233 validation-period, and 315 test-period candidates.
- Identity-checked queue-to-master review merger that preserves both source CSVs and records SHA-256 hashes.
- Four analysis/presentation notebooks.
- Synthetic end-to-end output generation.
- Current test suite, Python compilation, notebook execution, LaTeX compilation, and PDF visual inspection.
- Editable TikZ diagrams for the system pipeline, temporal split, and evidence interventions, with a manuscript claim-to-test table.
- Verified command guide for environment setup, diagnostic checks, pilot ingestion, seeded execution, evaluation, notebooks, and manuscript builds.

### Measured scientific evidence

- None yet.

### Current diagnostic evidence

- Synthetic six-record smoke run passes the software path.
- These values are tagged `synthetic_software_diagnostic_only` and must not support scientific claims.
- All 40 unit and integration tests pass in the project environment.
- The complete CPU-side smoke, manifest, evaluation, intervention, test, and four-notebook execution sequence passed again on 2026-10-24. Its numerical values remain synthetic diagnostics only.
- The synthetic feasibility audit correctly returns `stress_test_fallback`; its evidence records remain deliberately unassigned to audited real source clusters.
- The intervention builder produced 30 deterministic smoke manifests across six claims.
- All four notebooks execute from top to bottom, and the final six-page paper plus two-page supplement compile without LaTeX errors or unresolved references.
- The key-free frozen collection manifest is accepted by the preflight as `frozen_snapshot`; the full-run readiness artifact has API access removed from the blocker list.
- The candidate snapshot contains 32,898 deduplicated rows. Before manual screening, 16,809 have a conservative automatically suggested three-way label (14,768 refuted, 1,604 supported, and 437 insufficient-evidence); 16,089 remain deliberately unmapped.
- The 1,200-row acquisition queue passes strict review-schema validation with all rows still `pending`. Its SHA-256 is `b30f3b1835bca8f3c3f04a8df5ecbd761b1e6af8bef021a5e3cc72d6463acfc7`; the source review CSV SHA-256 is `240e6081303c6899b80fa083f4c566c0e40f168611611129e4cd25f5058c0ee5`.
- A tiny CPU train/checkpoint/calibration/prediction test passes. It checks mechanics only and supplies no scientific result.

### Unresolved before scientific claims

- ClaimReview candidate collection has been executed, but candidate-level US relevance, multimodality, target-image identity, explicit label, claim time, decision cutoff, and event membership have not yet been audited. GDELT and Common Crawl evidence collection therefore has not started.
- PolitiFact, AFP, and Snopes each reached the configured 100-page collection cap. The snapshot is sufficient for the 200-case screening pilot but is not a complete census; the final benchmark collector must use a higher cap, checkpointed pagination, or nonoverlapping dated queries and document the resulting coverage.
- No 200-claim feasibility audit exists.
- No frozen real-data split manifest exists.
- Dense BGE-M3, reranker, and SigLIP2 adapters are implemented but not executed on real data.
- Full CUDA graph training is implemented but not measured on real data; the local host has no CUDA runtime.
- Controlled baselines, ablations, and all implemented robustness evaluations have not been executed on the real corpus. The deterministic opposite-label donor policy for conflict interventions must be audited on the pilot and replaced with an audited donor manifest if it does not create semantically valid conflicts.
- All 17 supplied PDFs have a detailed task/method/evaluation/limitation audit. Correction, withdrawal, and retraction notice review remains pending.
- A dated venue-priority audit exists, but quartiles and author guidance must be rechecked near submission.

## 9. Resource and reproducibility assumptions

- The local host audit on 2026-10-24 found no NVIDIA runtime/GPU, 15.2 GB total RAM, and 150.2 GB free on drive D. A CPU PyTorch build is installed in the project environment for unit and miniature integration tests, but Kaggle or another CUDA environment is required after the data gate for the scientific run.
- Use open metadata and archives only; access does not automatically grant redistribution rights.
- Preserve exact model and tokenizer revisions.
- Preserve Python, framework, CUDA/driver, GPU, precision, batch-size, and runtime information.
- Use the fixed seed set `[11, 23, 37, 53, 71]` after model selection unless a documented deviation is approved.
- Use validation data for configuration and threshold selection; do not tune on final test data.
- A single-author audit of 200 records checks extraction quality but does not establish inter-annotator reliability.

## 10. Standard execution checks

From `experiments` with the project environment active:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_smoke_test.py --config configs/base.yaml
python scripts/validate_manifests.py --claims outputs/smoke/claims.jsonl
python scripts/evaluate_predictions.py --claims outputs/smoke/claims.jsonl --predictions outputs/smoke/predictions.jsonl --bootstrap-replicates 10000 --output outputs/smoke/evaluation.json
python scripts/audit_feasibility.py --claims outputs/smoke/claims.jsonl --evidence outputs/smoke/evidence.jsonl --output outputs/smoke/feasibility.json
python scripts/build_intervention_sets.py --claims outputs/smoke/claims.jsonl --evidence outputs/smoke/evidence.jsonl --output-directory outputs/smoke/interventions
python -m pytest -q
python scripts/execute_notebooks.py
```

After M2--M4 pass, freeze the audited manifests with `freeze_dataset.py`, then use `run_full_study.py --resume` on CUDA. The orchestrator builds pinned feature archives, performs validation-only selection, runs all configured five-seed baselines and ablations, evaluates the test split, computes paired event-level comparisons, and exports checked LaTeX tables. Exact Windows and Kaggle commands are maintained in `experiments/README.md`.

## 11. Decision and deviation log

| Date | Type | Decision or deviation | Reason | Affected artifacts/results | Required follow-up |
|---|---|---|---|---|---|
| 2026-10-15 | Decision | Use IEEE TCSS as the primary target venue. | Scope fit with computational social systems and multimodal misinformation. | Manuscript format and positioning. | Reverify author rules and quartile before submission. |
| 2026-10-15 | Decision | Restrict the new corpus to US-focused English cases. | Improves temporal/event scope coherence and matches the selected study design. | Dataset and generalization claims. | State geographic/linguistic limits in all final claims. |
| 2026-10-15 | Decision | Use inherited professional verdicts and silver evidence links. | No new multi-annotator labeling study is planned. | Label validity and evidence-quality claims. | Audit extraction quality; do not claim new human annotation or IAA. |
| 2026-10-15 | Decision | Use Python modules for core logic and notebooks only for analysis/presentation. | Required for testing, repeated runs, and multi-GPU reproducibility. | Entire experiment package. | Keep notebook-only scientific logic prohibited. |
| 2026-10-15 | Decision | Treat current manuscript as a preregistered protocol/pre-results draft. | Real-data experiments are not complete. | Abstract, results, discussion, PDF. | Replace status boxes and placeholders only after checked evidence exists. |
| 2026-10-15 | Implementation | Added project scaffold, tests, notebook execution, synthetic diagnostic, five-seed runner, and prediction evaluator. | Establish an executable base before data collection. | Repository and current PDFs. | Begin M2 feasibility pilot. |
| 2026-10-15 | Organization | Reserved `paper/` for LaTeX sources and manuscript-supporting assets; moved experiments, research records, the living plan, and the repository README to the study root. | Separate the manuscript build from the wider research workspace. | Paths, generated-table destination, execution commands, and audit locations. | Keep future non-manuscript artifacts outside `paper/`. |
| 2026-10-15 | Manuscript style | Removed draft-status boxes and synthetic diagnostic results from the paper and supplement; revised the prose with the Humanizer skill while preserving the protocol, citations, notation, and evidence limits. | Keep implementation diagnostics out of the scholarly narrative and improve readability. | Main manuscript, supplement, smoke-test generator, and manuscript table assets. | Add results only from preserved real-data experiment artifacts. |
| 2026-10-15 | Manuscript design | Added editable vector diagrams for the end-to-end system and temporal split, plus a claim-to-test matrix; converted the author block to IEEE journal style. | Make the study logic and leakage controls readable at a glance without adding unmeasured results. | Main manuscript, figure sources, table source, compiled PDF, and Overleaf package. | Replace or extend only with evidence-derived plots after real experiments. |
| 2026-10-15 | Reproducibility | Expanded the experiment guide with tested setup, diagnostic, pilot ingestion, seeded-run, evaluation, notebook, and LaTeX commands. | Give an executable path while keeping unimplemented real-training components explicit. | `experiments/README.md` and repository quick start. | Add the frozen real trainer command after M5 implementation. |
| 2026-10-17 | Manuscript clarity | Rewrote the abstract to open with a concrete false-context example, state the temporal, provenance, and insufficiency problems explicitly, and map them directly to the proposed solution and evaluation. Removed first-person plural wording throughout the manuscript in favor of ``this study'' or direct constructions. | Make the problem--solution logic understandable without requiring the reader to infer it from method details and maintain the requested academic voice. | Main manuscript and compiled deliverables. | Add measured findings only after the controlled real-data experiments are complete. |
| 2026-10-17 | Manuscript clarity | Expanded the introduction with background on false-context misinformation, decision-time leakage, source dependence, evidence sufficiency, and abstention; sharpened the literature gap and ended with one three-part contribution list. | Give readers enough context to understand the research problem before the method and state the paper's contributions without implying unmeasured results. | Introduction, main manuscript, and compiled deliverables. | Recheck contribution wording after the feasibility gate and completed experiments. |
| 2026-10-17 | Manuscript style | Applied a dedicated Humanizer review to the introduction, replacing vague or formulaic transitions with direct technical statements while preserving its claims, citations, and contribution structure. | Reduce detectable AI-writing patterns without weakening the academic argument or adding unsupported content. | Introduction and compiled deliverables. | Apply the same targeted review after later substantive prose revisions. |
| 2026-10-17 | Packaging | Reproduced the reported Overleaf failure check from the exact v4 archive, confirmed that its sources build after clean extraction, and created a main-manuscript-only v5 archive with a single top-level `main.tex` entry point. | Remove root-document ambiguity caused by packaging the main paper and supplement as separate compilable documents in one upload. | Overleaf upload package; manuscript source content is unchanged. | Upload v5 as a new Overleaf project or set `main.tex` as the main document and select pdfLaTeX. |
| 2026-10-17 | Manuscript correction | Repaired only meaning-changing grammar and phrase-order errors in the author's manually revised abstract. | Restore the intended method, split, and evaluation statements while preserving the author's wording and avoiding a broad stylistic rewrite. | Abstract and compiled deliverables. | Preserve this author-edited voice in later revisions; change wording only for accuracy or clarity. |
| 2026-10-17 | Manuscript scope | Cleared the body text under Evaluation Questions, Discussion, Ethics, and Limitations, and Conclusion while retaining their section headings. | The author requested these flagged passages remain blank until they are rewritten. | Sections 6--8 and compiled deliverables. | Refill only when the author supplies or requests replacement prose. |
| 2026-10-17 | Author metadata | Moved the department, institution, location, and email from the IEEE footnote into the visible author block beneath the title. | Ensure the full supplied affiliation is visible with the author's name on the first page. | Main manuscript title block and compiled deliverables. | Revert to the journal footnote form only if the target venue requires it at submission. |
| 2026-10-21 | Literature audit | Reviewed all 17 supplied papers at the task, method, evaluation, ablation, and limitation levels; recorded a dated venue-priority audit. | Ground the study in the closest recent A*/Q1 work rather than broad topical similarity. | Literature matrix, reference ledger, reference-paper dossier, venue audit, related work, and baseline plan. | Recheck notices and venue status before submission. |
| 2026-10-21 | Novelty refinement | Defined the central contribution as decision-time evidence validity plus independent corroboration; treated the graph encoder as an implementation choice. | Prior work already covers multimodal RAG, temporal reasoning, graph learning, relevance reranking, and context prediction separately. | Title, research question, contributions, method, ablations, and falsification rules. | Validate the joint mechanism with matched baselines and counterfactual interventions. |
| 2026-10-21 | Temporal protocol | Required a pre-cutoff archive/crawl observation for primary evidence and introduced timestamp-confidence tiers. | A publication field alone does not prove historical availability, and a post-cutoff archive can leak future information. | Schemas, temporal filter, tests, manuscript, and supplement. | Audit timestamp confidence on the 200-case pilot. |
| 2026-10-21 | Provenance protocol | Separated source dependence from credibility and bounded each dependency cluster's total aggregation mass. | Conflicting-evidence research shows credibility filtering can suppress useful counter-evidence, while syndication can inflate document counts. | Clustering, graph attributes, interventions, baselines, and manuscript. | Manually validate cluster quality and run duplicate interventions. |
| 2026-10-21 | Execution format | Retained a modular Python package and notebooks for analysis only; did not convert the study to one Kaggle notebook. | Collection, retrieval, graph construction, multi-seed training, tests, and artifact generation require reusable command-line modules. | Experiment layout and running instructions. | Kaggle may execute the same commands in a GPU session after the pilot data are mounted. |
| 2026-10-21 | Manuscript revision | Renamed the study, retained conventional top-level sections, added a mechanism-focused intervention diagram, strengthened the related-work comparison, and stated the pre-results evidence boundary explicitly. | Align the manuscript with the audited novelty and avoid unsupported results or decorative structure. | Main paper, supplement, bibliography, figures, table, and compiled PDFs. | Replace the Results status text only after frozen real-data artifacts exist. |
| 2026-10-21 | Verification | Passed 18 tests, the synthetic smoke path, feasibility and intervention commands, all four notebooks, LaTeX/BibTeX builds, log checks, and page-by-page PDF inspection. | Confirm the revised scaffold is mechanically consistent before the real-data pilot. | Experiment outputs and compiled paper/supplement. | Repeat from a clean frozen environment at M5 and M9. |
| 2026-10-23 | Data-intake implementation | Added authenticated paginated Fact Check API collection, raw-response manifests, candidate-review validation, verified image hashing, batch evidence discovery, Common Crawl observation checks, and a pilot-specific assessment. | Make the 200-case pilot executable without asking the author to download or hand-code records. | Experiment modules, scripts, tests, configuration, ignore rules, and running guide. | Store the API key locally and execute the first frozen collection. |
| 2026-10-23 | Snapshot correction | Replaced the unexecuted 2026-10-21 snapshot claim with a requirement to record the first successful collection timestamp. | No real collection occurred on 2026-10-21, so calling that date a frozen data snapshot would be inaccurate. | Plan now; dataset section and manuscript dates after real collection. | Propagate the actual timestamp only after the raw collection manifest exists. |
| 2026-10-23 | Data-record separation | Reserved `research/claim_evidence_register.csv` for manuscript-level scientific claims; pilot cases and evidence belong under `experiments/data/`. | Mixing paper-claim evidence status with dataset annotations would corrupt both audit trails. | Plan, experiment guide, and future pilot artifacts. | Generate pilot review and processed manifests through the collection scripts. |
| 2026-10-24 | Credential preflight | Detected the ignored local API key and reached its Google Cloud project, but Google returned HTTP 403 because the Fact Check Tools API is disabled. The preflight output redacts the key. | Verify live access before creating a supposedly frozen raw snapshot. | No real candidate data were accepted; API checker and error reporting were hardened. | Project owner must enable Fact Check Tools API and allow propagation, then rerun the preflight. |
| 2026-10-24 | Resource audit | The local host has no NVIDIA GPU and the project environment does not currently include PyTorch; collection and CPU checks remain viable. | Do not launch an unmeasured full training workflow on unsuitable hardware. | Execution plan and Kaggle requirement. | Complete M2 locally, then package frozen data and run GPU stages on Kaggle or equivalent. |
| 2026-10-24 | Model identity freeze | Pinned the official Hugging Face commit revisions for BGE-M3, BGE reranker v2-m3, SigLIP2 SO400M, and Qwen2.5-VL-7B-Instruct. | Prevent silent upstream model changes between pilot and final runs. | `configs/base.yaml`, adapters, and full-run readiness audit. | Recheck availability without changing revisions before the frozen GPU run. |
| 2026-10-24 | Full-run guard | Added and executed `preflight_full_run.py`; the first run blocked on API access, absent real pilot artifacts, absent real trainer, and absent local PyTorch/CUDA. | Make unresolved prerequisites machine-visible and prevent synthetic or incomplete work from being presented as a full run. | `outputs/preflight/full_run_readiness.json`, tests, and run instructions. | Rerun at every milestone; implement the trainer only after M2 passes. |
| 2026-10-24 | API restriction check | After service enablement, the live preflight changed to HTTP 403 stating that `FactCheckedClaimSearch` requests are blocked. | Distinguish service activation from API-key application/method authorization. | No raw snapshot was created and no key value was logged. | Adjust the key to permit local/server requests and restrict it to Fact Check Tools API, wait for propagation, then rerun preflight. |
| 2026-10-24 | Candidate snapshot | Confirmed the corrected key, collected 32,898 deduplicated 2020--2026 ClaimReview candidates from seven configured publisher filters, and preserved 440 raw API pages plus the review CSV and manifest at `2026-10-24T07:53:33.754563Z`. | Establish a reproducible source pool for the 200-case feasibility pilot without exposing the credential. | `experiments/data/private/pilot_claims_2026-10-24/`; full-run preflight now reports API status `ok`. | Build and audit the stratified review queue; treat the three 100-page sources as capped rather than exhaustive. |
| 2026-10-24 | Screening preparation | Built and validated a deterministic 1,200-row acquisition queue balanced at 400 candidates per suggested label, with preserved source and queue hashes. | Reduce the 32,898-row pool to a reproducible audit workload while retaining label and temporal coverage. | `screening_queue.csv`, `screening_queue_manifest.json`, M2 status, and run guide. | Audit rows in order until 200 eligible cases remain; automated label suggestions remain non-gold. |
| 2026-10-24 | Final-run implementation | Added frozen feature construction; fused BM25/BGE-M3/SigLIP2 retrieval with pinned BGE reranking; graph and matched feature trainers; validation-only tuning; five-seed baseline/ablation execution; paired event bootstrap with Holm adjustment; checked table export; and a resume-safe full-study orchestrator. | Make the planned CUDA experiment executable without oracle evidence links, notebook-only logic, or test-set tuning. | Retrieval/training modules, scripts, configuration, tests, run guide, preflight, and conditional manuscript tables. | Execute on the audited pilot first, freeze any protocol corrections, then run the unchanged final matrix on CUDA. |
| 2026-10-24 | Robustness validity gate | Required an audited semantic-conflict donor manifest for the final run and made manuscript-table export reject automatic opposite-label pairing or diagnostic zero-vector verdict injections. | An opposite inherited label does not establish that a donor document semantically contradicts a particular claim, and a zero vector is not a valid text intervention. | Full-run preflight, intervention evaluator, table exporter, tests, run guide, manuscript, and supplement. | Audit every held-out claim--donor pair before the CUDA run and preserve the manifest with the frozen study artifacts. |
| 2026-10-24 | Manuscript synchronization | Aligned the method and experiment prose with the implemented graph, objectives, baselines, and selection procedure; connected results tables to generated artifacts and recorded the real candidate snapshot without presenting it as a benchmark. | Keep the pre-results manuscript accurate and prevent synthetic or screening values from becoming scientific results. | Main paper, supplement, figures, compiled PDFs, and plan. | Populate Results and revise Discussion/Conclusion only after checked real-data artifacts exist. |
| 2026-10-24 | Release-readiness verification | Passed 40 automated tests, executed all four notebooks from top to bottom, rebuilt the six-page main paper and two-page supplement, checked logs for fatal/undefined/overfull errors, and visually inspected the changed method, experiment, results, reference, and supplement pages. | Establish that the CPU-side implementation and pre-results deliverables are mechanically coherent before human audit and CUDA execution. | Executed notebooks, test record, preflight report, rendered QA pages, and final PDFs. | Repeat the same verification from the frozen CUDA environment after the pilot gate passes; the Codex built-in preview compiler remains unavailable on this Windows host because its restricted-token sandbox cannot enforce the configured writable roots, so the successful local `latexmk` build is the current compiler evidence. |
| 2026-10-24 | Overleaf packaging | Rebuilt the main-manuscript-only package as `overleaf-main-2026-10-24-v9.zip`, including the new intervention figure, and compiled it successfully after clean extraction. | Keep the upload bundle synchronized with the verified local sources and avoid multiple-root ambiguity. | Overleaf ZIP and clean-extraction build record. | Upload the v9 archive with `main.tex` as the root; rebuild the package after generated result tables are added. |

## 12. Immediate next actions

1. Open `experiments/data/private/pilot_claims_2026-10-24/screening_queue.csv` and audit rows in their existing order until 200 valid image--text claims remain. Preserve every keep/reject decision and rejection reason; verify US relevance, target image, rating, claim time, decision cutoff, normalized label, and event identifier.
2. Copy the completed decisions back to the full `candidate_review.csv` or use an explicitly documented queue-to-master merge, validate it, fetch and hash only verified target images, and finalize the 200-case claim manifest.
3. Collect GDELT evidence candidates for retained claims, confirm publication times and pre-cutoff Common Crawl observations, review relevance and dependence clusters, and finalize the pilot evidence and linked-claim manifests.
4. Create and audit the semantic-conflict donor manifest for the held-out claims; every row must map a claim to genuinely opposing evidence rather than relying only on an opposite inherited label.
5. Execute `audit_pilot.py`. If the feasibility gate passes, run the implemented feature/training pipeline on the 200-case pilot as a CUDA systems check; record runtime, memory, failures, calibration behavior, and any protocol change before scaling.
6. Build and audit the full corpus, run `freeze_dataset.py`, transfer the frozen dataset and verified private image manifests to Kaggle or another CUDA host, then execute the exact resumable `run_full_study.py` command in `experiments/README.md`.
7. Import only the generated checked tables into the manuscript, reconcile every reported value with raw artifacts, and finish Results, Discussion, Limitations, and Conclusion. Until then the paper remains a complete pre-results construction, not a final empirical paper.

## 13. Update protocol

For every substantive study session:

1. Read this file and inspect the latest relevant outputs/configurations.
2. Identify the active milestone and its gate.
3. Make only changes within the recorded scope or add a dated deviation first.
4. Run proportionate tests and preserve outputs.
5. Update the status checklist, current evidence state, decision/deviation log, and immediate next actions.
6. Update manuscript claims only when the corresponding claim-test row has adequate measured evidence.
7. Recompile and visually inspect the PDFs after material LaTeX changes.

This plan remains active until the final package is audited or the study is explicitly paused or abandoned.
