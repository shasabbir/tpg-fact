# Living Study Plan

## Document control

- **Study:** Temporal and Evidence-Source Provenance-Aware Multimodal Misinformation Detection with Calibrated Abstention
- **Short name:** TPG-Fact / US-TPMFC
- **Primary venue:** IEEE Transactions on Computational Social Systems (TCSS)
- **Principal author:** Sabbir Hossain
- **Institution:** Department of Computer Science, American International University-Bangladesh, Dhaka, Bangladesh
- **Email:** 26-94030-1@student.aiub.edu
- **Plan status:** Active; protocol and software scaffold implemented, real-data feasibility pilot not yet executed
- **Last updated:** 2026-10-17

This file is the single source of truth for the study. It must be read before substantive data, model, evaluation, or manuscript work. After every material change, update the applicable checklist, decision/deviation log, evidence status, and next actions here. A change to the method invalidates dependent results until the affected experiments are rerun.

Use these evidence-state labels consistently:

- **Proposed:** specified here but not implemented.
- **Implemented:** present in code or manuscript and mechanically checked.
- **Measured:** produced by a preserved real-data experiment artifact.
- **Checked:** independently reconciled against the originating data, configuration, code, and output.

Synthetic smoke-test values are software diagnostics only. They are never scientific results.

## 1. Research objective

Create a full IEEE TCSS-style study titled:

> **Temporal and Evidence-Source Provenance-Aware Multimodal Misinformation Detection with Calibrated Abstention**

The system will classify US-focused image-text claims as:

- `supported`
- `refuted`
- `insufficient_evidence`

The central research question is:

> Does reasoning over time-valid, independently sourced evidence improve future-event multimodal misinformation detection and reduce overconfident decisions compared with ordinary multimodal RAG?

The central test compares the full temporal-provenance system with a matched vanilla multimodal RAG system under a future-event, event-disjoint evaluation.

### Planned contributions

1. A temporally partitioned US multimodal fact-checking benchmark.
2. A provenance graph representing evidence duplication, citation, source, image reuse, and publication time.
3. A hierarchical sufficiency-and-veracity detector.
4. Temporal-leakage, duplicated-source, calibration, abstention, and future-event evaluations.

### Scope boundaries

- The study is US-specific and will not claim global generalization.
- “Provenance” means evidence origin and source dependence, not only human-versus-AI authorship.
- Professional fact-checker verdicts are inherited labels; the project does not claim new human annotation.
- Automatically extracted evidence links are silver evidence.
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
normalized_label, event_id, evidence_ids
```

Evidence record:

```text
evidence_id, url, domain, title, published_at, archived_at,
text_sha256, image_phash, media_type, source_cluster_id,
outgoing_links, retrieval_query
```

Prediction record:

```text
claim_id, predicted_label, class_probabilities,
sufficient_evidence_score, selected_evidence_ids,
source_cluster_ids, decision_cutoff
```

Release code, URL/hash metadata, graph annotations, retrieval logs, and split manifests without redistributing copyrighted articles or images.

### 3.4 Retrieval and provenance model

1. Extract entities, dates, OCR text, and image captions.
2. Retrieve text using BM25 and BGE-M3.
3. Retrieve images using SigLIP2 and perceptual hashes.
4. Fuse rankings with reciprocal-rank fusion.
5. Rerank 100 candidates and retain the top 20 time-valid items.
6. Construct a heterogeneous temporal-provenance graph.
7. Cluster copied and syndicated evidence using MinHash, semantic similarity, image pHash, hyperlinks, domains, and publication order.
8. Encode the graph with a heterogeneous graph transformer.
9. Aggregate independent source clusters rather than raw document counts.
10. Predict evidence sufficiency first and veracity second.
11. Apply validation-fitted temperature scaling and an abstention threshold targeting 80% coverage.

Qwen2.5-VL-7B-Instruct will serve as an open multimodal comparison and optional query generator, not as an unverified label source. Record the exact model revision, prompt, decoding configuration, and execution date when used.

## 4. Experiment and testing plan

### 4.1 Baselines

1. Majority-class baseline.
2. Text-only encoder.
3. Image-text fusion without retrieval.
4. Multimodal LLM zero-shot and few-shot.
5. Vanilla multimodal RAG.
6. Time-filtered RAG without provenance.
7. Provenance-aware RAG without temporal filtering.
8. RED-DOT where its implementation is compatible.
9. Full temporal-provenance model.

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
- graph transformer;
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
| Time-valid retrieval improves future-event classification | Full model vs. matched vanilla RAG and provenance-only RAG | Test macro-F1; leakage stress test | Event-level paired bootstrap, five seeds | No improvement outside leaked setting → drop/narrow temporal-benefit claim |
| Source-dependency modeling prevents duplicate evidence inflation | Full model vs. no-clustering ablation under 1/5/10 copy injection | Macro-F1, confidence shift, Brier, selective risk | Event-level paired comparisons | Similar degradation or no robustness benefit → drop/narrow provenance claim |
| Sufficiency-first prediction improves handling of unavailable evidence | Full model vs. flat three-way classifier and no-gate ablation | Insufficient-evidence F1, macro-F1, Brier | Event-level paired bootstrap | No benefit or worse supported/refuted performance without selective-risk gain → remove claim |
| Calibrated abstention reduces operational risk | Temperature-scaled threshold vs. uncalibrated confidence at matched coverage | AURC and risk at 80% coverage | Event-level bootstrap; validation-selected threshold | Selective risk does not improve at matched coverage → remove calibrated-abstention claim |
| Multimodal evidence is necessary | Full model vs. text-only and visual-only evidence ablations | Macro-F1 and topic/window breakdown | Event-level paired bootstrap | One modality matches full model within uncertainty → narrow multimodal necessity claim |
| US-TPMFC supports benchmark-level conclusions | Corpus passes all preregistered thresholds and audit | Counts, class balance, event counts, evidence-cluster coverage | Dataset records grouped by event | Any threshold fails → release/use only as stress test |

## 6. Paper workflow and manuscript structure

### Main sections

1. Introduction
2. Related Work
3. Task and Benchmark Construction
4. Temporal-Provenance Method
5. Experimental Setup
6. Results
7. Robustness, Calibration, and Error Analysis
8. Ethical Considerations and Limitations
9. Conclusion

The current source groups discussion, ethics, and limitations in one section file and contains eight included section files. Split or rename files only when useful; preserve the nine-part conceptual structure in the final manuscript.

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

**Gate:** Passed for the software/protocol scaffold only. This gate does not establish scientific readiness.

### M2 — 200-claim feasibility pilot

- [ ] Freeze eligible fact-checker sources and collection snapshot date.
- [ ] Collect at least 200 candidate image-text claims.
- [ ] Run explicit-label mapping and exclusion audit.
- [ ] Verify timestamps, image recoverability, and archive availability.
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
- [ ] Freeze graph schema, temporal features, and unknown-origin treatment.
- [ ] Measure retrieval Recall@10/nDCG@10 and cluster-quality diagnostics.

### M5 — End-to-end model pilot

- [ ] Implement real BGE-M3/SigLIP2 feature extraction.
- [ ] Implement batched heterogeneous-graph training and checkpointing.
- [ ] Verify objective, gradients, missing-evidence cases, saving/loading, inference, and metrics.
- [ ] Measure pilot runtime and peak GPU memory.
- [ ] Verify temperature fitting and 80% coverage threshold selection on validation only.

### M6 — Baseline tuning and freeze

- [ ] Implement all feasible controlled baselines.
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
- Temporal evidence filtering.
- Event/split leakage checks.
- BM25 ranking and reciprocal-rank fusion scaffold.
- Duplicate/source clustering and provenance graph construction.
- Hierarchical smoke classifier and trainable graph-model components.
- Temperature scaling and abstention-threshold utilities.
- Classification, calibration, selective-risk, retrieval, AUROC, bootstrap, and Holm utilities.
- ClaimReview ingestion and open-archive discovery entry points.
- Prediction evaluator and five-seed command runner.
- Four analysis/presentation notebooks.
- Synthetic end-to-end output generation.
- Current test suite, Python compilation, notebook execution, LaTeX compilation, and PDF visual inspection.
- Editable TikZ diagrams for the system pipeline and temporal split, with a manuscript claim-to-test table.
- Verified command guide for environment setup, diagnostic checks, pilot ingestion, seeded execution, evaluation, notebooks, and manuscript builds.

### Measured scientific evidence

- None yet.

### Current diagnostic evidence

- Synthetic six-record smoke run passes the software path.
- These values are tagged `synthetic_software_diagnostic_only` and must not support scientific claims.

### Unresolved before scientific claims

- Real ClaimReview/GDELT/Common Crawl collection has not been executed.
- No 200-claim feasibility audit exists.
- No frozen real-data split manifest exists.
- Dense BGE-M3 and SigLIP2 retrieval is configured but not executed.
- Full batched/multi-GPU graph training is not implemented or measured.
- Controlled baselines, real ablations, and robustness studies are not executed.
- Reference identity/support entries exist, but full author/venue reconciliation and correction/withdrawal/retraction notice review remain pending.
- Venue quartile is unverified and must be checked near submission.

## 9. Resource and reproducibility assumptions

- Multi-GPU resources are expected but not yet inventoried.
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
python -m unittest discover -s tests -v
python scripts/execute_notebooks.py
```

For final real-data evaluation, pass five separate prediction files to `evaluate_predictions.py` and require `--expected-seeds 5`. Use `run_seed_matrix.py` to invoke the frozen training command once per configured seed.

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

## 12. Immediate next actions

1. Define the exact professional fact-checker source allowlist and collection snapshot date.
2. Collect a 200-claim feasibility sample without altering final split rules after inspecting results.
3. Populate `research/claim_evidence_register.csv` with pilot claim IDs, timestamps, archive status, and evidence-cluster audit fields.
4. Run the label, timestamp, image-recovery, event-group, and evidence-independence audit.
5. Measure pilot collection yield and resource cost, then decide whether the full benchmark thresholds are feasible.

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
