# Q1-Primary / Q2-Fallback Expansion Plan

Last updated: 2026-10-29

Status: **proposed expansion only**. Nothing in this document is a completed experiment or a result.

Quality objective: develop a study strong enough to target a Q1 venue, with a Q2 venue as the fallback. Quartile placement and acceptance are goals, not promises, and must be checked against the chosen venue, category, and year before submission.

## 1. Stable baseline and non-negotiable boundary

The completed CEFES/MOCHEG negative development study remains the stable, showable baseline:

- baseline commit: `c411ada` (`docs: report frozen negative MOCHEG development result`);
- frozen full-corpus feature SHA-256: `68889ae34ddbe55576725583dfd0312c71e061a2a579ff624c0885c1d590731f`;
- frozen development archive SHA-256: `b7961df904fcce77e9a596b119aba342e31d51c669ce82d8a306b9465c21d00b`;
- full-corpus mean CEFES-minus-fixed evidence-F1: `-0.04185`;
- all three seed effects are negative;
- `formal_test_unlocked=false` and no formal-test annotation was accessed.

The active negative-study manuscript remains in `paper/`. Phase-2 work must not overwrite its artifacts, silently change its protocol, remove unfavorable results, or reinterpret its failed development gate. New outputs will use a separate namespace, initially `experiments/outputs/q1_phase2/`.

The old CPU hash run remains software validation only. It is never scientific evidence.

## 2. Proposed study identity

### 2.1 Working title

**When Does Adaptive Evidence Selection Work? Retrieval-Conditioned Evaluation for Multimodal Fact-Checking**

### 2.2 Central problem

Evidence selectors are often evaluated after candidate construction has already determined which gold evidence is reachable. A selector can appear effective on a released judgment-candidate packet yet fail after label-blind retrieval from the full corpus. Fixed top-$k$, probability thresholds, learned counts, and expected-utility decoders may therefore be compared under retrieval conditions that favor different behaviors.

### 2.3 Proposed solution

Build a retrieval-conditioned evaluation framework that:

1. compares adaptive and fixed selectors on identical candidate packets;
2. measures retrieval coverage before interpreting selector performance;
3. evaluates selectors across multiple declared retrieval configurations and controlled retrieval-quality strata;
4. separates ranking failure, probability-calibration failure, set-decoding failure, and downstream-verification effects; and
5. tests whether conclusions replicate on a second independently licensed multimodal evidence benchmark.

CEFES remains one investigated selector. The broader contribution is the framework and the resulting evidence about when adaptive selection succeeds or fails, not a requirement that CEFES win.

## 3. Research questions

### RQ1 — Retrieval-conditioned selector validity

How do candidate construction, gold-evidence ceiling coverage, and ranking quality change the measured advantage of adaptive evidence selection over the strongest permitted fixed budget?

### RQ2 — Selector mechanism

Under matched candidates and scorers, when do exact expected-F1 decoding, a calibrated threshold, count prediction, and fixed top-$k$ succeed or fail?

### RQ3 — Failure decomposition

How much of the end-to-end error is attributable to retrieval misses, within-packet ranking, probability calibration, dependence among evidence items, and the final set decision?

### RQ4 — Verification and generalization

Do selector conclusions preserve verdict quality and replicate across retriever families, random seeds, evidence modalities, evidence-set sizes, and a second benchmark?

## 4. Falsifiable hypotheses

- **H1:** The relative ordering of selectors changes as complete-gold retrieval coverage changes.
- **H2:** No adaptive selector has a defensible end-to-end advantage when most gold evidence is absent from its candidate packet.
- **H3:** Exact expected-F1 decoding is most competitive when item probabilities are calibrated and ranking error is moderate; its advantage weakens under severe retrieval error or strongly dependent evidence items.
- **H4:** Conclusions observed only on judgment candidates will not be treated as end-to-end evidence unless they replicate under label-blind full-corpus retrieval.
- **H5:** Any claimed selector benefit must preserve verdict macro-F1 within a prespecified non-inferiority margin.

These hypotheses may be supported, rejected, or qualified. A negative result remains publishable evidence if the comparisons are complete, matched, and independently reproducible.

## 5. Contribution target

A Q1-level submission should aim to provide all of the following rather than only another CEFES run:

1. a precise retrieval-conditioned formulation of multimodal evidence selection;
2. a matched benchmark protocol covering fixed, threshold, count, and utility-based selectors;
3. a multi-retriever study that reveals how candidate quality changes conclusions;
4. a diagnostic decomposition with oracle ceilings and controlled degradation;
5. replication on a second appropriate benchmark;
6. rigorous paired uncertainty, multiplicity control, calibration analysis, and verifier safeguards; and
7. released code, manifests, predictions, hashes, failure logs, and frozen protocols.

If the second benchmark or multi-retriever evidence cannot be completed credibly, the work should be framed for a strong Q2 target rather than overstated as a Q1-ready general result.

## 6. Phase 0 — synchronize the existing study

Before new scientific runs, finish the remaining analyses and manuscript corrections using saved artifacts wherever possible.

### 6.1 Protocol wording and claim audit

- Replace “validation-selected” or “development-selected” for the fixed budget where the implementation actually selects it on the permitted training calibration fold.
- Remove or qualify any claim that the official MOCHEG baseline was reproduced unless a linked artifact establishes that reproduction.
- Distinguish the authoritative archive recorded as a recovery source from data actually downloaded and audited.
- Distinguish exact byte-duplicate auditing from perceptual duplicate or source/event auditing.
- Remove stale “pending” or future-tense language from completed results.

### 6.2 Missing saved-output analyses

- Compute the prespecified seed-aligned aggregate paired bootstrap interval.
- Apply and report the declared Holm correction for the selector family.
- Compute the paired verifier non-inferiority interval against the `-0.01` margin rather than relying only on a mean point difference.
- Generate reliability diagrams from the saved probabilities and report them with Brier score, NLL, and ECE.
- Complete the perceptual duplicate and available cross-split source/event audits, or explicitly narrow the manuscript claim when the release lacks sufficient metadata.

No new model training is needed for analyses that can be recomputed from frozen records. Every derived output must link back to the frozen candidate and prediction identities.

## 7. Phase 1 — retrieval and baseline reproduction audit

### 7.1 Official baseline

Reproduce the compatible original MOCHEG retrieval/verification baseline from its official code and documented settings. Before execution, record:

- exact repository revision and environment;
- any unavailable dependency or data component;
- faithful settings versus necessary compatibility changes;
- candidate identities and evaluation mapping; and
- which published numbers are directly comparable.

A partial or incompatible reproduction must be labeled as such and cannot be presented as the official baseline.

### 7.2 Declared retrieval matrix

Freeze a small, scientifically motivated matrix before looking at selector outcomes. It should include:

- the existing pinned BGE-M3/SigLIP2 configuration;
- a reproducible sparse lexical text retriever;
- a strong dense text retriever selected from current primary documentation;
- a strong frozen vision-language image retriever; and
- one declared hybrid/fusion configuration if its tuning access is matched.

Encoder identities, revisions, preprocessing, packet widths, fusion rules, and tuning budgets must be fixed in a Phase-1 protocol file. Retriever selection must be justified by retrieval quality and reproducibility, not by whether CEFES wins.

### 7.3 Retrieval endpoints

For every configuration and modality, retain:

- Recall@5/10/20/50 and complete-gold claim coverage;
- MAP and nDCG under the official identifier mapping;
- candidate-packet gold ceiling and oracle evidence-set F1;
- runtime, peak memory, corpus/index size, and hardware;
- exact candidate rankings before relevance joins; and
- missing, duplicate, and unreadable media records.

## 8. Phase 2 — matched selector matrix

Every selector must receive identical candidate identities, ranking features, training access, and configuration budget within a retriever condition.

Required selectors:

1. fixed top-$k$ over a predeclared grid;
2. calibrated global threshold;
3. learned count classification;
4. CEFES exact expected-F1 decoding;
5. uncalibrated expected-F1 ablation; and
6. oracle diagnostics that are clearly excluded from deployable comparisons.

The strongest permitted fixed comparator must be retained even when it yields an unfavorable result. Selector hyperparameters may not be chosen on formal-test labels.

Primary outcome: claim-level evidence-set F1 under complete gold sets, where missed retrieval evidence remains a false negative.

Secondary outcomes: precision, recall, exact match, Jaccard, selected count, count MAE, Brier score, NLL, ECE, verdict macro-F1, class-wise F1, runtime, and memory.

## 9. Phase 3 — failure decomposition and robustness

### 9.1 Retrieval-quality strata

Analyze claims by complete-gold coverage, modality requirement, gold-set size, verdict, duplicate cluster, and source/event group when metadata permit.

### 9.2 Controlled degradation

Starting from frozen candidate packets, apply predeclared perturbations without retraining after seeing outcomes:

- remove a controlled fraction of gold candidates;
- insert irrelevant high-ranked candidates;
- duplicate relevant or irrelevant evidence;
- vary packet width;
- remove text or image evidence; and
- perturb probability calibration while preserving rank.

These interventions should locate the transition at which selector conclusions change. They are diagnostic experiments, not substitutes for label-blind retrieval.

### 9.3 Decoder assumptions

Measure the effect of correlated and duplicated evidence on the independent-Bernoulli approximation. Compare exact expected-F1 with threshold and count controls under identical probabilities. If a dependence-aware extension is proposed, it must receive its own frozen development protocol and ablation.

## 10. Phase 4 — second-benchmark gate

Do not choose the second benchmark merely because it produces a favorable result. Select it through a documented access audit requiring:

- explicit research-compatible license;
- identifiable text and image evidence;
- released gold evidence identifiers;
- a corpus or defensible candidate-construction procedure;
- enough training/development data for calibration without test leakage;
- stable download and immutable versioning; and
- a task sufficiently aligned for the same evidence-set metrics.

Candidate datasets must be audited before selection. MuSciClaims remains excluded unless authoritative license metadata and usable development supervision become available. AIChartClaim/ChartCheck or another candidate may be considered only after verifying its license, splits, evidence identity, and task compatibility.

If no second dataset passes this gate, report the limitation and target the strongest defensible single-benchmark Q2 study rather than claiming broad generalization.

## 11. Statistical protocol

- Use at least five frozen seeds for the expanded development matrix when computationally feasible; declare them before execution.
- Preserve claim-level paired records for every method comparison.
- Average seed-aligned claim effects before the main paired interval, with seed dispersion reported separately.
- Use source/event/fact-check-article bootstrap units when reliable group metadata exist; otherwise use claim bootstrap and state the limitation.
- Use 10,000 or more paired bootstrap resamples for primary intervals.
- Apply Holm correction across the prespecified primary selector comparisons.
- Report effect sizes and intervals, not only $p$-values.
- Evaluate verdict preservation with a paired non-inferiority interval and margin `-0.01`.
- Predeclare missing-data handling, exclusion rules, and the number of permitted configuration trials.

## 12. Test-sealing and protocol freeze

The current frozen study does not unlock the MOCHEG formal test. The formal annotations remain unavailable.

Any expanded study is a new protocol and must first freeze:

- code commit and environment lock;
- dataset and corpus revisions;
- retriever and encoder revisions;
- candidate widths and fusion rules;
- selector/scorer settings;
- seeds and statistical tests;
- exclusion rules;
- artifact manifest schema; and
- a numeric development gate.

Only a separately reviewed Phase-2/Phase-3 development record may propose formal-test access. Passing one selector comparison alone is insufficient. Until that review is recorded, `formal_test_unlocked` remains false. Formal-test annotations must never be downloaded for exploratory tuning.

## 13. Artifact and implementation policy

New work must be isolated from the frozen study:

```text
experiments/configs/q1_phase2/
experiments/outputs/q1_phase2/
research/q1_phase2/
paper_q1/                 # create only when enough checked evidence exists
```

Do not create `paper_q1/` merely to hold speculative claims. Until the evidence base justifies a new manuscript, keep the completed negative paper in `paper/` and describe Phase-2 work only in this plan and research records.

Every run must preserve:

- resolved configuration and Git commit;
- environment and hardware manifest;
- corpus, media, model, and feature hashes;
- label-blind candidate rankings;
- joined candidate/gold records;
- per-seed probabilities and selected identifiers;
- calibration metrics and reliability data;
- bootstrap samples and multiplicity-adjusted decisions;
- runtime and memory records;
- failures, repairs, and regression tests; and
- an independently recomputable freeze manifest.

Recoverable implementation errors must be fixed with a regression test and the unchanged declared experiment rerun. Failed runs and protocol deviations remain visible.

## 14. Target manuscript structure

If Phase 2 produces sufficient checked evidence, the expanded manuscript should use this structure:

1. **Introduction:** fact-checking background, retrieval-conditioned problem, gap, framework, research questions, and contributions.
2. **Related Work:** multimodal fact-checking, retrieval evaluation, adaptive set prediction, calibration, and faithful attribution.
3. **Problem Formulation:** corpus retrieval, candidate packets, evidence selection, ceilings, and matched verifier.
4. **Retrieval-Conditioned Framework:** retrievers, selectors, decomposition, and protocol controls.
5. **Datasets and Integrity Audits:** MOCHEG plus the independently qualified second benchmark.
6. **Experimental Protocol:** baselines, fairness, metrics, uncertainty, multiplicity, and test seal.
7. **Results:** retrieval matrix, selector matrix, verifier preservation, and cross-benchmark replication.
8. **Failure Analysis:** ceilings, calibration, dependence, modality, perturbations, and qualitative errors.
9. **Discussion, Limitations, and Ethics.**
10. **Conclusion.**

The abstract must explicitly state the problem, proposed framework or method, experimental scope, principal checked result, and limitation. It must not promise superiority or external generalization before those claims are supported.

## 15. Milestones

### M0 — stable baseline

- [x] Frozen negative development experiment.
- [x] Independent audits and clean reproduction.
- [x] Formal test sealed.
- [x] Stable manuscript preserved in Git.

### M1 — synchronization completion

- [ ] Correct protocol wording and stale manuscript claims.
- [ ] Aggregate bootstrap and Holm analysis.
- [ ] Paired verifier non-inferiority interval.
- [ ] Reliability diagrams.
- [ ] Perceptual/cross-split audits or explicit scope reduction.

### M2 — Phase-2 protocol freeze

- [ ] Official-baseline compatibility audit.
- [ ] Retriever matrix selected without outcome shopping.
- [ ] Second-benchmark access audit.
- [ ] Trial budget, seeds, metrics, and gates frozen.
- [ ] New configuration and output namespaces implemented and tested.

### M3 — MOCHEG expanded development study

- [ ] Build and audit all declared retrieval artifacts.
- [ ] Run the matched selector matrix.
- [ ] Run decomposition and robustness analyses.
- [ ] Independently recompute primary tables.

### M4 — external replication

- [ ] Run the unchanged compatible protocol on the qualified second benchmark.
- [ ] Explain any dataset-specific adaptations before viewing outcomes.
- [ ] Freeze and independently audit the replication record.

### M5 — venue decision

- [ ] Q1 submission only if the work supports a broad, replicated contribution with complete controls.
- [ ] Q2 fallback if the study is rigorous but remains single-benchmark or has narrower generality.
- [ ] Retain an honest negative or mixed conclusion if that is what the evidence establishes.

## 16. Immediate next actions

1. Finish Phase 0 using only frozen artifacts where possible.
2. Write a short official-MOCHEG-baseline compatibility audit before running it.
3. Audit candidate second benchmarks for license, splits, evidence identity, and corpus access.
4. Draft the Phase-2 protocol and configuration namespace.
5. Review the protocol for outcome-independent choices before any new scientific run.

No action in this plan authorizes formal-test access, overwriting frozen artifacts, weakening the comparator, or suppressing negative findings.
