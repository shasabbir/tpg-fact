# Living Study Plan

Last updated: 2026-10-26  
Study status: **replanned; public-release preflight passed; schema audit is next**  
Primary venue class: strong Q1 journal or A* conference extension; no quartile or acceptance is guaranteed.

## 1. Study identity

Working title:

**Evidence-Traceable Multimodal Claim Verification with Counterfactual Citation Grounding**

Working method name: **TRACE-SCV** (Traceable Scientific Claim Verification).

Central claim to test:

> A multimodal verifier should not receive credit merely for predicting the correct verdict or producing a fluent, label-consistent explanation. Its cited evidence must be sufficient for the verdict, decisive when removed, and stable to irrelevant evidence.

This replaces the former US-TPMFC/decision-time-provenance study. The old private pilot, ClaimReview collection path, and temporal benchmark construction are out of scope.

## 2. Why this direction

The two supplied base papers are strong but leave the same unresolved problem.

- **M-RAV** retrieves text and images, verbalizes image--text consistency, ranks by relevance, and prompts an LLM to verify a claim. Its free-form verbalizations are not tied to explicit evidence identifiers, retrieval quality for images is weak, and its relevance module is neutral or harmful in some reported ablations.
- **MEVER** jointly retrieves, verifies, and explains with a multimodal graph. Its consistency regularizer aligns the explanation with the predicted label, but label consistency is not evidence attribution. Mean aggregation can also conceal which evidence actually caused a decision. Explanation evaluation is dominated by lexical overlap and a VLM judge.

Recent work already occupies several tempting alternatives:

- evidence sufficiency and selective abstention: SAFE-MR (2026 preprint);
- utility-based visual evidence selection: ACL 2026;
- adaptive visual-evidence necessity: AMuFC (2026 preprint);
- generic multimodal fact-level attribution: MuRGAt (2026 preprint).

TRACE-SCV therefore focuses on **fact-checking-specific, counterfactually tested citation grounding across text, tables, figures, and images**. It extends text-only citation masking and recovery to multimodal verification and turns attribution into a training objective rather than only an evaluation protocol.

## 3. Research questions and hypotheses

### RQ1: metric validity

How often do verdict accuracy, ROUGE, METEOR, and label--explanation consistency overstate evidence faithfulness?

**H1.** Systems can obtain a correct verdict and fluent explanation while citing the wrong evidence or remaining insensitive to removal of the cited evidence.

### RQ2: grounded verification

Does a sparse citation bottleneck with counterfactual citation grounding improve evidence attribution without materially degrading verdict quality?

**H2.** TRACE-SCV improves evidence-item citation F1 and counterfactual faithfulness while keeping macro-F1 within one percentage point of the matched verifier.

### RQ3: robustness

Does explicit grounding improve behavior under irrelevant additions, evidence duplication, decisive-evidence deletion, and cross-modal swaps?

**H3.** TRACE-SCV is more invariant to irrelevant or duplicated evidence and more sensitive to removal or swapping of decisive evidence than relevance-only and label-consistency baselines.

### RQ4: transfer

Do attribution gains transfer across scientific and general-domain datasets?

**H4.** A model trained on public scientific datasets retains an attribution advantage on a public general-domain benchmark, even when absolute verdict performance drops.

## 4. Public datasets only

No new claims, articles, images, or labels will be collected from the web. Official public releases are used through download scripts; copyrighted source papers or images are not republished beyond their licenses.

### Core datasets

1. **SciVer** -- 3,000 expert-annotated examples from 1,113 papers with text, tables, figures, supporting evidence, and rationales. Official Hugging Face release `chengyewang/SciVer`, public, ungated, CC BY 4.0. This is the primary dataset.
2. **MuSciClaims** -- public Hugging Face release `StonyBrookNLP/MuSciClaims`, with multimodal scientific claims and diagnostic perturbation types. This is the main out-of-dataset scientific test.
3. **MOCHEG** -- public author release and Apache-2.0 code repository. It supplies general-domain multimodal evidence, three-way verdicts, and ruling explanations, and permits direct comparison with both base papers. The official release is downloaded by the user-facing script; redistribution is governed by the dataset authors' terms.

### Auxiliary dataset

4. **AIChartClaim** -- public release in the MEVER repository. It is used only after its release terms and split integrity pass the dataset audit. It is not required for the main claim.

### Dataset gates

A dataset enters the main tables only if:

- its official release is publicly accessible when the experiment is frozen;
- the license or author usage terms permit research use;
- all required modalities are present for at least 90% of the selected split;
- exact and perceptual duplicate checks are reported;
- claim/evidence identifiers and official splits can be frozen in a manifest;
- no test labels or explanations leak through file names, URLs, or prompt fields.

If MOCHEG access or media recovery is incomplete, it becomes an optional external evaluation and the main study remains SciVer + MuSciClaims.

## 5. Task definition

For claim `c` and candidate evidence set `E`, the system outputs:

1. verdict probabilities over the dataset's official label set;
2. an ordered set of cited evidence identifiers `S`;
3. a short explanation in which every factual sentence contains at least one valid citation.

An evidence identifier refers to an immutable dataset item: a text block, table, figure, image, or linked caption. A citation is correct only when it points to an official gold evidence item for that claim or passes the dataset-specific support rule.

## 6. Proposed method

### 6.1 Evidence graph

Represent each claim and candidate evidence packet as a heterogeneous graph:

- nodes: claim, text block, table, figure/image, caption;
- edges: same document, caption-of, refers-to, adjacent section, retrieved-for, and cross-modal similarity;
- node features: frozen text and vision embeddings plus evidence type and position.

The graph is document-local. No source-reputation or private metadata is used.

### 6.2 Sparse citation bottleneck

A claim-conditioned selector scores evidence nodes and selects a sparse set with a straight-through top-k or hard-concrete gate. The verifier and explanation decoder can access only selected nodes. This makes the claimed evidence path inspectable and prevents an explanation from citing evidence that the predictor never used.

### 6.3 Verdict and cited explanation

A compact multimodal backbone produces the verdict. The explanation decoder emits ordinary text plus constrained citation tokens drawn only from evidence identifiers in the current packet. Invalid or invented citations are impossible at decoding time.

### 6.4 Counterfactual citation grounding

Training uses four paired evidence interventions:

- **decisive deletion:** remove gold/selected decisive evidence; the correct-label margin should fall;
- **irrelevant addition:** add an unrelated item; the verdict and citations should remain stable;
- **duplicate addition:** duplicate a cited item; the prediction should not strengthen merely because of repetition;
- **cross-modal swap:** replace a figure/image while retaining text, or vice versa; the model should change only when the swapped modality is necessary.

The objective combines verdict loss, evidence-selection loss, explanation loss, sparsity, comprehensiveness, sufficiency, and invariance terms. Every loss has a registered ablation.

### 6.5 Resource profile

The reference implementation uses a frozen or 4-bit multimodal backbone with LoRA adapters and a small graph/selection head. The target is a single Kaggle 2xT4 or P100 session for pilot runs. Full five-seed experiments may require a stronger GPU, but the code must support gradient accumulation and resumption.

## 7. Comparisons

### Required baselines

1. majority label;
2. claim-only text model;
3. image/table-only model where applicable;
4. frozen multimodal backbone without retrieval;
5. relevance-ranked top-k multimodal RAG;
6. M-RAV or its authors' released outputs/configuration;
7. MEVER or its authors' released outputs/configuration;
8. matched verifier with unconstrained explanation generation;
9. matched verifier with label-consistency regularization only;
10. TRACE-SCV.

When an original baseline cannot be reproduced under available hardware, the paper must label it as a cited published result and must not mix it into statistical tests against locally reproduced systems.

## 8. Evaluation

### Primary endpoints

- evidence-item citation macro-F1;
- verdict macro-F1.

Both are required. A method is not declared better if attribution improves by sacrificing more than one absolute verdict-F1 point.

### Secondary endpoints

- evidence Recall@k, MAP, and nDCG@k;
- accuracy, per-class F1, and balanced accuracy;
- explanation ROUGE-L, METEOR, BERTScore, and citation coverage;
- citation masking-and-recovery accuracy;
- comprehensiveness: correct-label probability drop after cited-evidence deletion;
- sufficiency: verdict performance using cited evidence only;
- irrelevant-addition and duplicate invariance;
- cross-modal swap sensitivity;
- latency, peak GPU memory, tokens, and selected evidence count.

### Statistical protocol

- tune once on validation data, then freeze configuration;
- five seeds for trained systems after the pilot;
- paired bootstrap over claims or paper groups, 10,000 samples;
- 95% confidence intervals;
- Holm correction within verdict, attribution, and robustness comparison families;
- group resampling by source paper when multiple claims share one paper;
- report effect sizes and intervals, not only p-values.

## 9. Ablations

Remove or replace one component at a time:

- citation bottleneck;
- graph relations (flat evidence set);
- decisive-deletion loss;
- irrelevant-addition loss;
- duplicate-invariance loss;
- cross-modal-swap loss;
- sparsity penalty;
- constrained citation vocabulary;
- textual evidence;
- visual/table evidence.

## 10. Falsification and stop rules

- If cited-evidence deletion does not reduce the correct-label margin more for TRACE-SCV than for the matched baseline, do not claim causal grounding.
- If citation F1 improves but verdict macro-F1 falls by more than one absolute point, report a trade-off rather than superiority.
- If automatic attribution metrics fail validation against an existing gold field, drop them from primary evaluation.
- If a public dataset has unresolved leakage or license ambiguity, exclude it from the main study.
- If MEVER or M-RAV cannot be reproduced fairly, compare against a matched local reimplementation and clearly separate published numbers.
- No result placeholder may be converted into a factual claim before the corresponding artifact and audit entry exist.

## 11. Project layout

```text
ET2/
├── Base papers/                 # the two user-supplied anchor papers
├── Reference_good_papers/       # retained source archive
├── plan.md                      # this living source of truth
├── README.md
├── research/                    # literature, novelty, dataset and claim registers
├── experiments/                 # reusable Python package, configs, tests, notebooks
├── kaggle/                      # Kaggle entry notebook and instructions
└── paper/                       # LaTeX manuscript and its direct supporting assets only
```

Generated datasets, checkpoints, caches, and outputs remain ignored by Git.

## 12. Milestones

### M0 -- completed: reset and replan

- inspect the two base papers page by page;
- audit current 2025--2026 closest work;
- reject overlapping old novelty;
- remove the obsolete pilot, Kaggle bundles, virtual environment, outputs, and manuscript;
- create the public-data study plan, new implementation scaffold, and pre-results manuscript.

### M1 -- current: data feasibility

- [completed 2026-10-26] run the public dataset preflight notebook;
- freeze exact repository revisions and manifests;
- inspect schemas, modality completeness, label balance, duplicate rates, and leakage;
- select final core datasets using the gates above.

### M2 -- metric validation

- implement evidence-item citation scoring;
- validate masking/recovery, sufficiency, and comprehensiveness on gold evidence;
- run deterministic intervention unit tests;
- establish claim-only and relevance-only baselines.

### M3 -- model pilot

- train one-seed TRACE-SCV and matched baseline on a fixed 10% pilot;
- confirm memory fit, checkpoint resumption, and non-degenerate citation selection;
- pass the two-primary-endpoint go/no-go gate.

### M4 -- frozen main experiments

- freeze config and hashes;
- run five seeds, baselines, and ablations;
- run cross-dataset and intervention tests;
- generate all tables and figures from saved artifacts.

### M5 -- paper completion

- replace all pre-results statements with audited outputs;
- complete limitations, reproducibility, and ethics;
- compile and visually inspect paper and supplement;
- verify target venue requirements and quartile at submission time.

## 13. Current decisions and next action

Decisions:

- public datasets only;
- no ClaimReview/GDELT/Common Crawl collection;
- no new benchmark claim without a released dataset manifest;
- no generic sufficiency/abstention novelty claim;
- paper uses conventional section names;
- notebook orchestrates experiments, while reusable logic remains in `.py` modules.

Next action:

> Download the pinned SciVer and MuSciClaims snapshots, inspect their actual schemas, modalities, labels, duplicates, and license files, then freeze the dataset manifests before model training.

## 14. Validation record

Verified on 2026-10-26:

- SciVer and MuSciClaims are public, ungated, reachable, and match the pinned revisions in the registry;
- MOCHEG and the MEVER/AIChartClaim repositories are publicly reachable but remain optional pending data-term audits;
- all 10 unit tests pass;
- the deterministic synthetic end-to-end smoke check passes;
- the Kaggle preflight notebook executes all 12 cells without an error;
- `paper/paper.tex` compiles successfully with the desktop LaTeX compiler.

These checks validate access and software plumbing only. They are not scientific results.
