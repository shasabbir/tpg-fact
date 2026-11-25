# Publication-Ready Execution Plan

> Historical execution record. The active, bounded final-revision plan is
> [`publication_one_shot_plan.md`](publication_one_shot_plan.md). Completed
> protocol decisions and evidence in this file remain valid; its older
> multi-gate action sequence is no longer the working checklist.

Last updated: 2026-11-12
Active manuscript: `paper/paper.tex`  
Current execution checkpoint: `b2075d0` plus one locked development run
Current verdict: **external study complete; manuscript and submission packaging in progress**

## 1. Publication objective

The primary target is a strong journal paper whose central contribution is a
retrieval-conditioned and calibration-aware evaluation of multimodal evidence
selection. The paper does not depend on inventing a uniformly superior
selector. A rigorous external nonreplication is acceptable and must remain in
the paper.

The preferred publication story is:

> Evidence selection cannot be evaluated independently of candidate
> reachability and probability calibration. Calibrated thresholding is the
> strongest confirmed MOCHEG policy; expected-F1 is a compact operating point;
> and an independently frozen AVerImaTeC experiment tests which relationships
> transfer under different evidence units and class balance.

Method-superiority, state-of-the-art, deployment-readiness, and broad-
generalization language remains prohibited unless new checked evidence supports
it.

## 2. Checked starting point

| Component | Current evidence | Status |
|---|---|---|
| MOCHEG development study | Retrieval, calibration, matched selectors, robustness, stronger verifier, five deterministic partitions | Complete |
| Main selector conclusion | Calibrated thresholding is strongest; expected-F1 is smaller; completeness switching is inconclusive | Complete, mixed result |
| Formal MOCHEG test | Annotations remain sealed | Protected |
| Canonical manuscript | 14-page PDF, 51 rendered references, visual QA complete | Strong foundation |
| Implementation suite | 163 tests pass; bytecode compilation and independent reporting regeneration pass | Complete |
| External annotations | AVerImaTeC train 793, development 152; identities and class distribution audited | Complete |
| External organizer store | Complete train and validation archives extracted and hash-audited | Complete |
| External model fitting | Five train-only seeds fit and frozen before development preparation | Complete |
| Locked external comparison | One 152-claim run; independent aggregate and bootstrap recomputation passed | Complete |
| Official external score | Gemma-3-27B judge not executed | Missing |
| Blind external test | No prediction or submission | Missing and authorization-gated |

The organizer-store result is deliberately diagnostic rather than a shared-
task score. Exact-source pool recall is 25.48%, 59.87% of development claims
have no reachable target, and the official semantic judge remains unavailable
on the local single-GPU machine.

## 3. Actual remaining gaps

The declared external experiment has run on the organizer store. The remaining
gaps concern official semantic evaluation, the authorization-gated blind test,
and submission packaging rather than missing selector experiments.

| Required component | Current state | Required work |
|---|---|---|
| Official training store | Complete archives extracted and hash-audited | Preserve hashes and acquisition notes in the supplement |
| Text retrieval | BM25, BGE-M3, reciprocal-rank fusion, and BGE reranking completed | Report the low exact-source ceiling without treating it as semantic score |
| Image retrieval | Actual image discovery and SigLIP2/filename-BGE ranking completed | Preserve the large-image warning and duplicate/source audit |
| Multimodal packet | 100 text/20 image retained; every seed selected 50/10 | Complete |
| Calibration | Modality-specific train-only selection completed | Complete; calibration is the only passed external selector hypothesis |
| Selector matrix | Five policies completed in one locked run | Complete; superiority and compactness gates mostly failed |
| Shared verifier | Frozen four-class verifier evaluated once on development | Complete; switch non-inferiority failed |
| Official prediction output | Five fixed-family development exports pass schema | Keep local unless blind-test authorization is given |
| Official semantic evaluation | Hardware feasibility audited only | Run pinned Gemma-3-27B evaluator on suitable hardware or report it unavailable |
| Reporting | Independent recomputation, simultaneous intervals, class-wise verifier results, reachability audit, tables, and figure completed | Finish PDF and reference/build QA |

No post hoc selector search or locked-development rerun is permitted.

## 4. Critical-path work packages

### Gate A: Data and compute availability

Work:

1. obtain complete official AVerImaTeC training text and image stores;
2. record source URLs, revisions, byte sizes, SHA-256 hashes, license, missing
   files, duplicate URLs, and extraction manifests;
3. verify at least one faithful route for the official Gemma-3-27B evaluator:
   suitable multi-GPU hardware, an organizer-provided evaluation service, or a
   documented blocked status; and
4. keep test data and leaderboard access outside the development path.

Pass condition:

- every training store shard is complete and hash-verified;
- the extracted store covers all 793 training claims;
- no development or test outcome influenced acquisition or filtering; and
- the official-evaluator route is resolved before locked development.

Stop rule: if the official training store cannot be obtained, do not train on
the participant-filled validation store. Route the paper to the bounded-study
fallback in Section 8.

### Gate B: Faithful external retrieval implementation

Text branch:

1. deterministic 1,500-character chunks;
2. BM25 and BGE-M3 rankings;
3. equal-weight reciprocal-rank fusion;
4. one training-selected fusion/reranking configuration; and
5. packet width selected from `{50, 100}` using training roles only.

Image branch:

1. actual candidate-image loading and source association;
2. exact and perceptual duplicate audit without deleting audit rows;
3. SigLIP2 claim-text/claim-image scoring;
4. frozen tie-breaking and missing-image behavior; and
5. packet width selected from `{10, 20}` using training roles only.

Required artifacts:

- `external_retrieval.py` and focused tests;
- label-blind rankings written and hashed before gold joins;
- source and image mapping audit;
- Recall@10/20/50/100, MAP/nDCG, complete-gold reachability, packet-oracle
  evidence F1, zero-reachable claims, runtime, RAM/VRAM, and storage; and
- a failure/deviation log for inaccessible or corrupt sources.

Pass condition: all retrieval variants are complete on training roles, every
selector receives the same frozen packet, and no gold question or answer is a
retrieval feature.

### Gate C: Train-only end-to-end pilot

Work:

1. deterministically partition official training claims into fit,
   calibration, model-selection, and internal-holdout roles for seeds
   `13, 37, 73, 101, 137`;
2. audit claim/source/article and available image-hash overlap across roles;
3. fit the external evidence-membership scorer from scratch;
4. select modality-specific calibrators by training-role Brier score;
5. run fixed, threshold, expected-F1, uncalibrated expected-F1, and the
   completeness-aware switch on internal holdout;
6. train the shared linear and two-layer four-class verifiers; and
7. export a small training-only batch through the exact official schema.

Pass condition:

- the complete path runs without development data;
- labels, probabilities, evidence limits, source URLs, and four-class mappings
  pass regression tests;
- the verifier reports macro-F1 and every class-wise F1, not accuracy alone;
- peak runtime/memory fit the declared budget; and
- the official evaluator completes on training-only examples or is formally
  marked unavailable.

### Gate D: Protocol freeze

Freeze in one immutable manifest:

- dataset/store/code/model revisions and hashes;
- split manifests and overlap audit;
- candidate ID and URL/image normalization rules;
- retrieval matrix and packet widths;
- scorer features, negative sampling, calibrators, thresholds, and switch;
- verifier architectures and optimization;
- seeds, trial limits, primary comparisons, bootstrap seeds, Holm family,
  exclusion rules, and failure policy;
- official-evaluator version, precision, hardware, and prompt/template hashes;
- blind-test selection rule; and
- a statement that no locked development result was inspected.

Pass condition: an independent audit reproduces every non-runtime hash and
rejects development-label access, test-label access, and overwrite of frozen
outputs.

### Gate E: Full training-role execution

Run every declared configuration across all five seeds. Preserve failed and
dominated configurations.

Required outputs:

- per-candidate raw and calibrated probabilities;
- per-claim selected evidence IDs for every policy;
- calibration reliability data, Brier, NLL, ECE, prevalence Brier, and average
  precision;
- completeness-switch predictions and routing decisions;
- verifier probabilities and predictions by class;
- runtime and resource manifests; and
- clean-rerun hashes for non-runtime artifacts.

Pass condition: no missing declared run, no unexplained exclusion, and the
train-only freeze selects exactly one setting for locked development.

### Gate F: One locked development comparison

Run the frozen matrix once on all 152 official development claims. Do not tune
after observing results.

Primary comparisons:

1. calibrated threshold minus strongest train-selected fixed top-k;
2. expected-F1 minus calibrated threshold; and
3. calibrated minus uncalibrated expected-F1.

Required reporting:

- mean-over-seed then claim-level effect;
- 10,000 claim-aligned paired bootstrap samples;
- 95% intervals and Holm-adjusted values;
- every seed effect;
- precision, recall, F1, exact match, and selected count;
- reachability strata and evidence medium;
- macro-F1, class-wise F1, and confusion matrices for the shared verifier;
- official semantic evidence score and conditional verdict score when the
  pinned evaluator is available; and
- exact-source metrics explicitly labeled diagnostic.

Decision rules:

- threshold superiority requires at least `+0.010` evidence score with a
  positive lower interval bound;
- expected-F1 is a confirmed compact operating point only if its evidence score
  is non-inferior within `-0.010` and it selects at least 15% fewer items;
- the completeness switch supports superiority only if its mean gain is at
  least `+0.010`, the multiplicity-adjusted interval is positive, every seed is
  positive, and mean set size is no larger than thresholding; and
- verifier preservation requires the paired macro-F1 lower bound above
  `-0.010`.

Failure is a result. Do not search for another decoder after this gate.

### Gate G: Blind test decision

After Gate F, select exactly one pipeline using the frozen decision rule:

1. completeness switch if every switch safeguard passes;
2. otherwise expected-F1 if it is non-inferior and at least 15% smaller;
3. otherwise calibrated threshold if its fixed-baseline hypothesis passes;
4. otherwise strongest train-selected fixed top-k.

Generating and submitting test predictions requires explicit author approval.
Only one submission is permitted by this project. Preserve the submitted file,
SHA-256 hash, code commit, configuration, timestamp, and leaderboard receipt.
No post-test repair may be presented as fresh confirmation.

### Gate H: Evidence reconciliation and manuscript revision

Before writing, create a claim-evidence register linking every headline number
to raw records, aggregation code, configuration, and data identity.

Update only `paper/`:

1. abstract and contribution list last;
2. data/setup with the external evidence unit, split roles, imbalance, and
   store construction condition;
3. method with the retrieval, calibration, selector, completeness, verifier,
   and output paths actually run;
4. results with reachability first, then calibration, matched selectors,
   verifier, external transfer, and failures;
5. discussion with replication and nonreplication separated;
6. limitations with official-evaluator availability, alternative-source
   validity, class imbalance, and test status; and
7. conclusion restricted to checked claims.

The paper must not use internal phase labels. It must retain the MOCHEG
selector-superiority failure and seed-specific verifier failures.

### Gate I: Venue and submission package

Select the venue only after the external result establishes whether the final
paper is primarily multimedia methodology, information-retrieval evaluation,
or an applied-system study. Recheck the venue's current official scope, page
limit, anonymity policy, supplementary rules, AI-disclosure policy, and
required declarations at that time.

Required package:

- venue-formatted main paper and supplement;
- editable LaTeX, bibliography, figure/table sources, and build instructions;
- environment lock and acquisition scripts;
- code/data availability and license notes;
- claim-evidence and reference-support ledgers;
- clean source archive that builds outside the working tree;
- anonymized version when required;
- cover letter, highlights, or graphical abstract only when required; and
- final visual QA and independent scientific audit.

No preprint, repository release, leaderboard submission, or journal submission
is authorized by this plan.

## 5. Parallel non-experimental work

These tasks can proceed while data access or compute is pending:

1. audit every manuscript citation for DOI identity and direct claim support;
2. reconcile method/equation/diagram descriptions with current code;
3. prepare the external dataset and reproducibility tables with result cells
   left blank;
4. prepare a clean-build script and source-package manifest;
5. draft, but do not invent, the required declaration templates; and
6. maintain an unresolved-action register.

## 6. Author decisions and resources required

The authors must provide or approve:

- access to the complete official AVerImaTeC training store or an authoritative
  mirror;
- suitable official-evaluator compute or acceptance that the official score
  will remain unavailable;
- explicit authorization before the one blind-test submission;
- target venue after the external result;
- final author order, affiliations, and corresponding author;
- CRediT roles;
- funding/grant or no-funding statement;
- competing-interest statement;
- data/code availability and media-license wording;
- acknowledgments and ethics wording; and
- venue-required AI-assistance disclosure.

These facts cannot be inferred from the repository.

## 7. Dependency-based execution order

| Order | Work | Exit artifact |
|---:|---|---|
| 1 | Official store and evaluator access | Updated access audit and hashes |
| 2 | BM25/BGE-M3 and actual SigLIP2 retrieval | Frozen label-blind rankings and retrieval report |
| 3 | External switch, verifier, and official exporter | End-to-end training-only pilot audit |
| 4 | Protocol freeze | Immutable freeze manifest and commit |
| 5 | Five-seed full training roles | Complete train-only result bundle |
| 6 | One locked development run | Frozen external comparison report |
| 7 | Scientific decision | Selected blind policy and claim-boundary report |
| 8 | Optional one-shot blind test with approval | Submission hash and receipt |
| 9 | Statistics, tables, figures, and paper revision | Audited canonical manuscript |
| 10 | Venue conversion, declarations, clean build | Submission-ready archive |

The main critical path is data access -> faithful retrieval -> train-only pilot
-> freeze -> locked development -> manuscript. Reference and package work can
run in parallel.

## 8. Publication routes

### Route A: strong external diagnostic paper

Use when the external protocol completes, regardless of whether expected-F1 or
the switch wins. The central contribution is the cross-benchmark evidence about
reachability, calibration, compact selection, and failure transfer.

### Route B: method-plus-evaluation paper

Use only if the completeness switch or another prespecified policy clears every
superiority and verifier safeguard on untouched data. This route is not the
default expectation.

### Route C: bounded single-benchmark paper

Use if official training data, evaluator compute, or blind evaluation remains
unavailable. Preserve the current MOCHEG development framing, include the
external access failure and reachability audit, remove external-confirmation
language, and target a venue compatible with a rigorous measurement/negative-
results contribution. Do not call this Q1-ready external generalization.

### Route D: contradictory external result

Report the contradiction and investigate only prespecified integrity failures.
Do not retune on external labels. A credible nonreplication may strengthen the
paper's diagnostic contribution.

## 9. Definition of publication-ready

### Scientific readiness

- [x] Core MOCHEG study is frozen, reproduced, and honestly reported.
- [x] Formal MOCHEG test remains sealed.
- [x] External annotations, access, label distribution, and fallback store are audited.
- [x] Complete official training store is hash-verified.
- [x] BM25/BGE-M3 and actual SigLIP2 retrieval are implemented and audited.
- [x] Completeness switch, shared verifier, and official exporter pass an end-to-end execution.
- [x] Train-only protocol and all decisions are frozen before development outcomes.
- [x] Five-seed training-role execution is complete.
- [x] One locked external development comparison is complete.
- [x] Official semantic score is recorded or its absence is an explicit claim limitation.
- [ ] Blind test is executed once with authorization or explicitly marked blocked.
- [x] Every main quantitative claim maps to valid raw evidence and checked aggregation.

### Manuscript readiness

- [x] Canonical source and 18-page reading PDF exist in `paper/`.
- [x] Current PDF builds without undefined references or overfull boxes.
- [x] Current negative and inconclusive findings are visible.
- [x] External methods and results match the final executed code exactly.
- [x] All 51 cited claims pass the recorded human support review and automated bibliography audit.
- [x] Final abstract, contributions, tables, discussion, limitations, and conclusion are synchronized.
- [ ] Target venue and current rules are verified.

### Submission readiness

- [ ] Author declarations are approved.
- [ ] Venue-formatted main paper and supplement pass visual QA.
- [x] Clean extracted source package builds without machine-local paths.
- [ ] Reproduction instructions and availability statements are complete.
- [ ] Independent final review reports no blocking scientific, reference, or packaging issue.

The study is publication-ready only when the applicable scientific,
manuscript, and submission boxes are checked. A visually polished PDF alone is
not completion.

## 10. Immediate next actions

1. re-run the complete implementation suite and bytecode compilation;
2. build and visually inspect the canonical manuscript;
3. complete bibliography and clean-source build audits;
4. seek suitable hardware only if an official semantic score is required;
5. obtain explicit authorization before generating the single blind-test file;
6. select a venue and verify its current rules; and
7. complete author declarations and the submission package.

No action in this plan authorizes paid compute, external contact, opening
sealed annotations, public release, or submission.
