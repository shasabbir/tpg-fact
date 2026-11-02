# Q1-Primary / Q2-Fallback Expansion Plan

Last updated: 2026-10-29

Status: **proposed expansion only**. Nothing in this document is a completed experiment or a result.

Quality objective: develop a study strong enough to target a Q1 venue, with a Q2 venue as the fallback. Quartile placement and acceptance are goals, not promises, and must be checked against the chosen venue, category, and year before submission.

## Executive strategy

The expansion has two linked but distinct goals:

1. **Primary scientific goal:** establish when conclusions about adaptive evidence selection are valid under changing retrieval quality. This claim can remain valuable even if no adaptive selector wins everywhere.
2. **Secondary method goal:** develop a stronger retrieval-conditioned selector that improves evidence-set F1 without obtaining its gain by returning very large evidence sets.

Work proceeds in the strict order **retrieval -> evidence scoring and calibration -> set decoding -> verifier -> external replication**. A later stage does not start merely because an earlier component can be implemented. It starts only after the earlier component passes its declared audit and quality gate. This prevents spending the main budget on increasingly complicated decoders while most gold evidence is still absent from the candidate packet.

The intended phase flow is:

```text
Frozen negative study
        |
        v
Phase 0: synchronize claims and saved-output statistics
        |
        v
Phase 1: improve and audit label-blind retrieval
        |  retrieval gate fails -> publish/evaluate retrieval-conditioned framework only
        v
Phase 2A: improve item scoring and calibration
        |  scorer gate fails -> retain simple matched scorers
        v
Phase 2B: compare predeclared set decoders
        |  selector gate fails -> no superiority claim and no formal-test access
        v
Phase 3: robustness and controlled degradation
        |
        v
Phase 4: second-benchmark replication and formal-test decision
```

### Scope control

The study will not simultaneously invent a new retriever, a large multimodal verifier, a generative explanation system, and a set decoder. The required contribution is retrieval-conditioned evidence selection. Explanation generation remains out of scope, and the shared verifier remains a safeguard/diagnostic rather than the primary optimization target.

### Fresh-data policy

The MOCHEG development outcomes have already been inspected in the completed study. They must therefore be described as reused development evidence, not as fresh confirmation. New method choices will be made using new claim-grouped partitions of the released training split: fitting, calibration, model-selection, and an internal locked holdout. Once the Phase-2 protocol is frozen, the official MOCHEG development split may be run once as a locked benchmark check. A fresh confirmatory claim requires either the still-sealed MOCHEG test under an approved gate or an independently qualified second benchmark with an untouched evaluation split.

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

### 2.4 Concrete Phase-2 method candidate

The proposed improved system is provisionally called **retrieval-conditioned CEFES (RC-CEFES)**. The name is not a claim of success and should be changed if the final contribution is primarily evaluative. Its components are fixed conceptually before outcome inspection:

1. **Hybrid text retrieval:** BM25 and BGE-M3 retrieve documents independently; reciprocal-rank fusion produces a broad document pool; sentence candidates are then reranked with a claim--sentence cross-encoder.
2. **Hybrid image retrieval:** SigLIP2 visual similarity is combined with OCR text, an image-caption representation, and available article context. Exact/perceptual duplicate groups are retained in the audit and suppressed during final selection rather than silently deleted from evaluation.
3. **Content-aware membership scoring:** replace the similarity/length-only logistic scorer with modality-specific claim--candidate cross-encoders, while keeping the logistic scorer as a required low-capacity control. Training uses hard negatives mined from frozen label-blind rankings.
4. **Claim-grouped calibration:** calibrate text and image scores separately on a training-only calibration fold. Compare temperature scaling, Platt scaling, and one monotonic non-parametric calibrator within a fixed trial budget. Selection of a calibrator uses Brier score first, with NLL and ECE as diagnostics.
5. **Precision-aware set decoding:** compare original expected-F1, calibrated thresholding, learned count, and a predeclared penalized decoder
   \[
   U_{\lambda,\gamma}(S)=\mathbb{E}[F_1(S,G)]-\lambda |S|-\gamma R(S),
   \]
   where $R(S)$ penalizes same-source and near-duplicate evidence. The penalties and maximum set size are selected on training-only folds.
6. **Uncertainty behavior:** if the scorer has no sufficiently credible evidence, the selector records a low-confidence/abstention flag for analysis. Because MOCHEG claims have released evidence, abstention is not scored as a correct empty set.

The key ablation order is retrieval only, then retrieval plus stronger scorer, then retrieval plus scorer plus decoder. This makes it possible to attribute any gain rather than presenting one inseparable system change.

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

### 7.4 Retrieval development sequence and gate

Run retrieval improvements in this order so each added source of complexity has a matched ablation:

1. reproduce the frozen BGE-M3/SigLIP2 rankings;
2. add BM25 to text retrieval and evaluate fusion;
3. add text cross-encoder reranking with frozen hard negatives;
4. add OCR/caption text to image retrieval;
5. add image reranking; and
6. evaluate the single predeclared hybrid configuration selected on training-only folds.

The proposed minimum gate for investing in a new selector is an absolute Recall@50 improvement of at least `+0.10` for text and `+0.10` for images over the frozen full-corpus values, with no reduction in complete-gold claim coverage or packet-conditioned oracle F1. These values are provisional until copied into the frozen Phase-2 protocol; they may not be changed after selector outcomes are viewed. The aspirational targets are text Recall@50 `>= 0.60` and image Recall@50 `>= 0.40`.

If the minimum retrieval gate fails, do not compensate by tuning the decoder. Complete the retrieval-conditioned analysis, report the failed retrieval variants, and decide whether the framework alone supports a paper.

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

### 8.1 Scorer gate

The content-aware scorer must be evaluated before any new decoder is tuned. Compared with the frozen logistic scorer on identical retrieved packets, it must:

- improve within-packet MAP or average precision on every declared seed and on average;
- improve or preserve Brier score after training-only calibration;
- preserve candidate order and gold joins in the audit records; and
- avoid obtaining an apparent ranking gain through duplicate leakage or source overlap.

If these conditions fail, retain the logistic scorer and treat stronger scoring as a negative ablation. Do not search for a decoder that happens to exploit one failed scorer checkpoint.

### 8.2 Decoder matrix

The deployable decoder comparison is limited to:

1. strongest permitted fixed top-$k$;
2. per-modality calibrated threshold;
3. learned count with the same scorer representation;
4. original CEFES;
5. size-penalized CEFES; and
6. size- plus redundancy-penalized RC-CEFES.

All methods receive the same probabilities and rankings where their definitions permit. Thresholds, count classes, $\lambda$, $\gamma$, modality rules, and maximum selected size are fixed using training-only selection data. A decoder that selects more evidence may not be described as better without reporting its precision and mean/median set size.

### 8.3 Superiority gate

A new selector-superiority claim requires all of the following on the locked development check:

- mean claim-level evidence-F1 gain of at least `+0.015` over the strongest permitted fixed budget;
- a positive effect in every one of five frozen seeds;
- a positive seed-aligned paired 95% interval after Holm correction for the declared primary comparisons;
- calibrated Brier score non-worse than the uncalibrated scorer in every seed;
- verdict macro-F1 paired non-inferiority lower bound above `-0.01`;
- mean selected set size no more than twice the mean gold-set size unless a predeclared recall-oriented analysis explicitly justifies it; and
- clean reproduction and independent artifact recomputation.

Failure of any condition keeps the MOCHEG test sealed and changes the manuscript claim from method superiority to retrieval-conditioned evaluation or negative/mixed evidence.

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

### 11.1 Configuration and compute budget

To prevent outcome-driven search, the expanded development budget is capped before scientific runs:

- at most **6 retrieval configurations**, including the frozen baseline and all ablations;
- at most **8 scorer/calibrator configurations**;
- at most **8 decoder configurations**, including fixed and threshold controls;
- one training-only model-selection pass, followed by one locked MOCHEG development check;
- five frozen seeds only for configurations that survive the relevant single-seed screening gate; and
- no formal-test reruns or post-test repairs presented as fresh confirmation.

Single-seed screening may eliminate broken or clearly dominated configurations, but it may not support a scientific claim. The screening seed, elimination rule, and minimum effect must be written into the protocol. All eliminated configurations remain in the experiment ledger.

Every run records wall-clock time, GPU-hours, peak VRAM, peak RAM, and storage. The plan should be reduced before execution if the complete declared matrix cannot be finished and audited on the available RTX 4090-class workstation.

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
- [ ] Training fit/calibration/selection/internal-holdout claim IDs frozen.
- [ ] Retrieval, scorer, and decoder gates copied into a machine-readable protocol.
- [ ] Configuration-trial and compute budgets frozen.

### M3 — MOCHEG expanded development study

- [ ] Reproduce the frozen full-corpus rankings and metrics.
- [ ] Build and audit the declared sparse, dense, reranked, and hybrid retrieval artifacts.
- [ ] Apply the retrieval gate before training a new selector.
- [ ] Train and audit the logistic and content-aware membership scorers.
- [ ] Apply the scorer gate before tuning a new decoder.
- [ ] Run the matched fixed, threshold, count, CEFES, and RC-CEFES matrix.
- [ ] Apply the superiority gate without opening formal-test annotations.
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

### Dependency-based working schedule

The schedule is expressed in work blocks rather than calendar promises:

| Block | Deliverable | Exit condition |
|---|---|---|
| A | Phase-0 corrected manuscript statistics and audit note | Frozen outputs reproduce and all wording matches implementation |
| B | Machine-readable Phase-2 protocol and train-only split manifest | Independent review finds no outcome-dependent choice |
| C | Retrieval index, rankings, audit, and comparison table | Retrieval gate decision recorded |
| D | Scorer checkpoints, hard-negative manifest, and calibration report | Scorer gate decision recorded |
| E | Matched decoder records and primary paired analysis | Superiority/no-superiority decision recorded |
| F | Controlled degradation and qualitative error analysis | Prespecified robustness matrix complete |
| G | Second-benchmark audit and replication | Replication result frozen or documented access failure |
| H | Manuscript and artifact release candidate | All claims trace to immutable records |

Blocks C--E must not overlap in a way that allows later outcomes to change earlier-stage configurations. Dataset-access auditing for Block G may proceed in parallel because it does not use selector outcomes.

## 16. Immediate next actions

1. Finish Phase 0 using only frozen artifacts where possible, beginning with the aggregate paired interval, Holm correction, verifier non-inferiority interval, and reliability diagrams.
2. Reconcile stale documentation against the completed frozen run; in particular, remove statements that still describe the GPU pilot as unfinished.
3. Create `research/q1_phase2/protocol.md` and `experiments/configs/q1_phase2/protocol.yaml` containing the train-only split policy, trial budgets, component gates, seeds, and test lock.
4. Add tests that prevent Phase-2 outputs from writing into the frozen namespaces and prevent test-annotation access.
5. Write the official-MOCHEG-baseline compatibility audit before attempting reproduction.
6. Implement and reproduce the frozen retrieval configuration through the new Phase-2 interface before adding new models.
7. Implement BM25/BGE-M3 fusion and its audit; do not begin cross-encoder training until the label-blind rankings reproduce.
8. Audit candidate second benchmarks for license, splits, evidence identity, and corpus access in parallel with retrieval engineering.
9. Review and sign off the frozen protocol before any outcome-bearing Phase-2 run.
10. Record every deviation, failed run, and eliminated configuration in `research/q1_phase2/experiment_ledger.md`.

No action in this plan authorizes formal-test access, overwriting frozen artifacts, weakening the comparator, or suppressing negative findings.
