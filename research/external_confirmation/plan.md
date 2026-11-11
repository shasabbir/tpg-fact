# Independent External Confirmation Plan

Date: 2026-11-11  
Status: execution in progress; train-only models and the locked development
comparison have not yet run, and no leaderboard submission has been performed
Checkpoint before planning: Git commit `b79ae39` (`v4`)

## 1. Decision

Use **AVerImaTeC** as the primary external benchmark. This is a new,
benchmark-specific confirmation protocol, not an unchanged replication of the
MOCHEG experiment.

The earlier access audit remains historically correct about incompatibility
with the MOCHEG evidence unit, but its statement that no usable test was
available is now superseded by the official task record. The organizers
released test data and a claim-specific knowledge store, published test results
in the 2026 FEVER task report, and maintain a running test leaderboard.

MMCV is not the primary confirmation benchmark because it has no official
fit/calibration/test split, requires reconstruction from MultiModalQA, and adds
table evidence. It may support a later three-modality study but must not replace
AVerImaTeC after results are seen.

## 2. Why AVerImaTeC is suitable

- It contains real-world image--text claims rather than claims generated from
  the MOCHEG corpus.
- Its evidence is obtained from the web and represented with text and/or image
  content, URLs, and question--answer annotations.
- The official task supplies separate training, development, and hidden test
  roles.
- The organizers provide a static knowledge store, avoiding a paid search API
  and making candidate construction reproducible.
- Its official score conditions verdict correctness on evidence quality, which
  directly tests whether a selected evidence set remains useful downstream.
- The official task caps output at ten evidence pieces per claim, giving a
  meaningful review-set budget.

The benchmark is not identical to MOCHEG. The evidence unit, label space,
candidate store, and official metric all differ. Confirmation therefore means
replication of the **scientific relationships** under a separately frozen
protocol, not equality of raw scores.

## 3. Confirmation objective

Test whether the central findings transfer to an independent real-world
benchmark:

1. candidate reachability must be reported before selector quality;
2. calibrated probabilities improve evidence-set decisions over the same
   uncalibrated ranking;
3. calibrated thresholding is a strong matched baseline;
4. expected-F1 can provide a smaller evidence set without an unacceptable loss
   in evidence quality; and
5. any adaptive-selector gain must preserve downstream verdict quality.

The packet-completeness switch is included as a secondary replication. Its
MOCHEG result was inconclusive, so it is not assumed to improve externally.

## 4. Prespecified research questions and hypotheses

### RQ1: reachability

How much annotated evidence is reachable from the frozen knowledge-store
packet before selection?

Report source-normalized Recall@10/20/50/100, complete-gold reachability,
zero-reachable claims, and packet-oracle evidence quality.

### RQ2: calibration and selection

On identical packets and rankings, how do fixed top-k, calibrated thresholding,
expected-F1, and uncalibrated expected-F1 compare?

- **H1:** calibrated thresholding exceeds the strongest train-selected fixed
  top-k by at least `+0.010` mean official evidence score, with a paired 95%
  interval lower bound above zero.
- **H2:** calibrated expected-F1 is non-inferior to calibrated thresholding in
  official evidence score with margin `-0.010`, while returning at least 15%
  fewer evidence pieces on average.
- **H3:** calibrated expected-F1 exceeds the uncalibrated expected-F1 ablation
  by at least `+0.010` evidence score, with a paired 95% interval lower bound
  above zero.

### RQ3: completeness routing

Does a train-only packet-completeness estimate reliably choose between
thresholding and expected-F1?

The switch supports a superiority claim only if its mean evidence-score gain
over thresholding is at least `+0.010`, the Holm-adjusted paired interval is
positive, every frozen seed is positive, and its mean set size is no larger
than thresholding. Otherwise it remains an inconclusive or negative result.

### RQ4: downstream preservation

Does the selected policy preserve verdict quality under one shared verifier?

The claim-level macro-F1 non-inferiority margin is `-0.010`. The official
AVerImaTeC score, evidence score, verdict accuracy, class-wise F1, and selected
count are all reported separately.

## 5. Data, versions, and access boundary

### Official sources checked before planning

- Dataset page: `https://fever.ai/dataset/averimatec.html`
- Task page and test-release notice: `https://fever.ai/task.html`
- Dataset repository: `https://huggingface.co/datasets/Rui4416/AVerImaTeC`
- Dataset revision audited: `7e77fb26a1ba86159ff5ba27b59f373b450d6fd0`
- Published test archive SHA-256: `21369f74caf95b2681cf168d540b6ecf1ab81b719523c5fb51b207d7eaa30dab`
- Official implementation HEAD checked during planning:
  `5fd2e03763e198eb6ddcea9e41b7210020f37127`
- Shared-task report DOI: `10.18653/v1/2026.fever-1.6`
- Test leaderboard: `https://huggingface.co/spaces/FEVER-IT/AVerImaTeCTest`
- License reported by the organizers: CC BY-NC 4.0.

Before implementation, download manifests and every knowledge-store shard must
receive a SHA-256 hash. Record the actual split counts, storage size, missing
files, duplicate URLs, inaccessible pages, and valid text/image counts. Do not
commit copyrighted web content or organizer data to Git; commit only manifests,
hashes, derived identifiers, and permitted statistics.

### Split roles

| Official data | Role in this study | Permitted use |
|---|---|---|
| Training | fit, calibration, model selection, internal holdout | All fitting and hyperparameter selection |
| Development | one locked matched-policy evaluation | No tuning after outcomes are inspected |
| Test | one final selected-pipeline evaluation | Prediction generation only; labels remain unavailable |

The official training split is deterministically partitioned by claim group
into 55% fit, 15% calibration, 15% model selection, and 15% internal holdout.
The five frozen seeds remain `13, 37, 73, 101, 137`. Group checks must use
normalized original-claim URL, fact-check article, reporting source, and a
near-duplicate claim/image hash where available.

No MOCHEG checkpoint, threshold, calibrator, completeness model, or fitted
weight is transferred. Only algorithm families and evaluation questions are
transferred.

## 6. Evidence-unit definition

The external unit is an organizer-store evidence item, not a MOCHEG item.

Each candidate receives a deterministic identifier built from:

```text
claim_id + normalized_source_url + medium + content_sha256 + local_ordinal
```

Candidate types are:

- text document chunk with source URL;
- image with source URL and perceptual/content hash; and
- a paired text--image item when the official store explicitly associates both.

Gold source membership is constructed only after candidate rankings are
written and hashed. Exact source labels use normalized annotated source URLs
and released image identifiers/hashes. When a gold document contains multiple
chunks, one deterministic answer-bearing chunk is chosen by a train-frozen
mapping rule. Mapping precision and recall are manually audited on a stratified
training-only sample before the rule is frozen.

Because alternative valid evidence may not share the annotated URL, exact
source-set F1 is a transparent diagnostic, not the external primary endpoint.
The official semantic evidence score and evidence recall remain primary.

## 7. Candidate construction

Use only the organizer-provided static knowledge store. Do not use live web
search, paid APIs, or post-test scraping.

### Text branch

1. normalize and chunk store documents with the organizer's frozen 1,500-character maximum;
2. retrieve with BM25 and BGE-M3 independently;
3. combine ranks with reciprocal-rank fusion;
4. optionally rerank with the already pinned cross-encoder family, refitted or
   used frozen without MOCHEG labels; and
5. retain a train-selected packet width from the declared set `{50, 100}`.

### Image branch

1. deduplicate exact and perceptual image matches without deleting audit rows;
2. rank candidate images with the pinned SigLIP2 family;
3. use claim text, claim image, and available organizer-store context only;
4. retain a train-selected packet width from `{10, 20}`.

The allowed retrieval matrix is capped at four configurations: sparse,
dense/visual, equal-weight fusion, and one train-selected weighted fusion. All
policies receive the same retained packet.

The official knowledge store was constructed partly with gold-question-derived
queries and includes annotated evidence URLs. This makes retrieval easier than
unrestricted web search and must be stated as a benchmark condition rather
than hidden as an implementation detail.

## 8. Membership scoring and calibration

Reuse the current low-capacity feature pipeline where compatible, adding only
AVerImaTeC-specific source and medium indicators. Fit the scorer from scratch
on the external training roles.

Candidate calibrators are limited to identity, temperature, prior shift,
Platt, beta, and isotonic mappings. Text and image probabilities are calibrated
separately. Selection uses training-role Brier score, with NLL, ECE, reliability
plots, prevalence Brier, and average precision reported.

The external scorer/calibrator budget is capped at six configurations. A
candidate cannot be selected because it performs favorably on official
development outcomes.

## 9. Matched selector matrix

Every selector receives identical candidate identifiers, rank order, fitted
probabilities, and a maximum output size of ten items:

1. fixed top-k, with `k` selected from `{1, 3, 5, 10}` using training roles;
2. calibrated global threshold, selected from `{0.1, ..., 0.9}`;
3. calibrated expected-F1 over ranked prefixes of length 0--10;
4. uncalibrated expected-F1 ablation; and
5. the prespecified packet-completeness switch.

Learned count is retained as a secondary diagnostic only. It is not part of the
primary comparison family because its MOCHEG behavior was already strongly
inferior and the external confirmation budget should prioritize the central
claims.

## 10. Shared verifier and official output

Train one compact four-class verifier on external training roles only. It uses
claim features and pooled selected-evidence representations; architecture,
optimization, and class weighting are shared across policies. Retain a linear
capacity control and one two-layer MLP, matching the current safeguard logic.

The output adapter must enforce the official schema:

- no more than ten evidence pieces;
- no evidence item above 1,500 characters;
- source URL retained for every item;
- text and image components represented as required by the official evaluator;
- exactly one of the four official verdict labels; and
- no test label or reference-evidence field in the prediction path.

Use the official development evaluator at its pinned revision. Record its
model/checkpoint requirements because the official evidence metric uses a
multimodal judge. Report judge version, precision, hardware, runtime, and any
unavailable component. A replacement evaluator invalidates direct comparison
unless both official and replacement scores are reported separately.

## 11. Statistical protocol

- Independent unit: claim.
- Stochastic repetitions: five frozen seeds.
- Primary external comparison set: threshold vs fixed, expected-F1 vs
  threshold, and calibrated vs uncalibrated expected-F1.
- Main effect: mean within claim across seeds before resampling.
- Uncertainty: 10,000 claim-aligned paired bootstrap resamples.
- Multiplicity: Holm correction across the three primary comparisons.
- Report mean, median, 95% interval, every seed effect, and mean/median selected
  count.
- Report reachability strata, evidence medium, gold-set size, verdict class,
  claim date, and image-related question status as descriptive subgroups.
- No subgroup is promoted to a confirmatory claim after outcomes are seen.

## 12. Locked development run and blind test rule

The official development split is evaluated once after all training-role
choices, code tests, dry runs, and artifact audits pass. A failed scientific
hypothesis is not a reason to rerun or retune.

The blind-test policy is selected by this frozen rule:

1. choose the completeness switch only if every superiority safeguard passes;
2. otherwise choose expected-F1 if its evidence score is non-inferior to
   thresholding and its evidence set is at least 15% smaller;
3. otherwise choose calibrated thresholding if H1 passes;
4. otherwise choose the strongest train-selected fixed top-k.

The full matched selector comparison is therefore made on locked development;
the hidden test evaluates one selected end-to-end pipeline. This avoids using
multiple leaderboard submissions to probe hidden outcomes.

Only one final test submission is allowed by this project, even if the public
leaderboard technically permits more. Submission requires explicit author
authorization after reviewing the frozen development and integrity audits.
This plan does not authorize a submission.

## 13. Work packages and exit conditions

| Block | Work | Exit condition |
|---|---|---|
| A. Access and identity | Pin revisions, download manifests, hash every archive/store shard, verify license and evaluator availability | Complete data/access audit; no labels or hidden scores exposed |
| B. Data adapter | Implement schema, deterministic evidence IDs, URL/image normalization, role guards, and output converter | Schema and leakage tests pass; train-only mapping audit accepted |
| C. Train-only pilot | Measure runtime/storage, verify gold mapping, ranking, scorer fitting, calibration, verifier, and official evaluator on training roles | End-to-end train-only run completes within budget; no scientific claim |
| D. Protocol freeze | Freeze YAML, split manifest, revisions, packet widths, budgets, seeds, hypotheses, selection rule, and statistics | Independent audit reproduces hashes and rejects outcome-dependent settings |
| E. Full training | Execute retrieval, scoring, calibration, selectors, switch, verifier, and clean reproduction on training roles | Every declared configuration complete; source/result audits pass |
| F. Locked development | Run the frozen matrix exactly once and apply all prespecified decisions | Result bundle and deviation log frozen; no post-outcome tuning |
| G. Blind test | With explicit author approval, generate and submit the one selected test prediction | Leaderboard receipt and immutable submitted-file hash recorded |
| H. Reporting | Independently recompute results and update the canonical paper | All claims trace to artifacts; negative/nonreplicated findings retained |

Blocks A--D may be executed without inspecting official development outcomes.
Block G remains authorization-gated.

## 14. Repository implementation map

Create a separate namespace; do not write into frozen MOCHEG outputs:

```text
experiments/configs/external_confirmation/averimatec.yaml
experiments/src/trace_scv/data/averimatec.py
experiments/src/trace_scv/external_retrieval.py
experiments/src/trace_scv/external_selection.py
experiments/src/trace_scv/external_verifier.py
experiments/src/trace_scv/external_reporting.py
experiments/scripts/audit_averimatec_access.py
experiments/scripts/prepare_averimatec.py
experiments/scripts/freeze_external_confirmation.py
experiments/scripts/run_external_confirmation.py
experiments/scripts/build_external_reporting.py
experiments/outputs/external_confirmation/averimatec/
research/external_confirmation/
```

Required regression tests include:

- deterministic candidate/evidence identifiers;
- split/group disjointness and duplicate audits;
- no development labels in fitting or configuration selection;
- no test labels in any executable path;
- label-blind rankings hashed before gold join;
- maximum ten returned items and 1,500-character limit;
- exact official label mapping and submission schema;
- paired-resampling alignment and Holm decisions;
- immutable output namespaces and failed-run preservation; and
- byte-identical clean reproduction for non-runtime artifacts.

## 15. Resource and search budget

Available workstation resources are sufficient for a feasibility audit: RTX
4090, approximately 64 GiB RAM, and about 1.29 TB free on the project drive at
planning time. Actual organizer-store size must still be measured before a
full download is approved.

Caps before locked development:

- four retrieval configurations;
- six scorer/calibrator configurations;
- five selector policies;
- two verifier architectures;
- five deterministic seeds only for configurations surviving train-only
  integrity and non-degeneracy checks;
- no paid API calls; and
- no additional model family introduced after protocol freeze.

If the complete declared run cannot fit the workstation budget, reduce the
matrix before freezing; do not drop unfavorable configurations afterward.

## 16. Risks and fallbacks

| Risk | Required response |
|---|---|
| Static store is unavailable, incomplete, or too large | Stop before outcomes; document the access failure. Do not use a favorable subset. |
| Gold URL to candidate mapping is unreliable | Use the official semantic evidence metric as primary, narrow exact-set claims, and report mapping audit error. |
| Official evaluator/model cannot be reproduced | Do not claim official comparability. Report transparent source-ID metrics separately and keep blind confirmation pending. |
| Development outcomes have already influenced a setting | Reclassify development as model-selection evidence; preserve the hidden test as the only confirmation set. |
| Expected-F1 or the switch fails externally | Retain the nonreplication and narrow the paper; do not search for a new decoder. |
| Test leaderboard is unavailable | Freeze predictions and hashes, report the blocked test, and do not call development results blind confirmation. |
| Blind result is weak | Report it. The submission decision may change, but the result must not be hidden or repaired on test. |

MMCV is not an automatic fallback after any unfavorable AVerImaTeC result. It
can be reconsidered only after a new three-modality protocol and split freeze.

## 17. Paper update rules

Do not update the abstract or contributions with external claims until Blocks
A--F are checked. After the locked development result:

- add AVerImaTeC data/evidence-unit and compatibility details;
- add a cross-benchmark comparison table with non-comparable metrics clearly
  separated;
- report both replication and nonreplication outcomes;
- retain the MOCHEG selector and per-seed failures;
- distinguish locked development evidence from hidden-test evidence; and
- keep the canonical manuscript in `paper/`; do not create a second paper tree.

The paper may claim external confirmation only after a valid hidden-test score
or an equivalently untouched, independently scored result is recorded. A
locked public development result alone is external-benchmark evidence, not a
blind-test result.

## 18. Completion criteria

The external study is complete only when:

- dataset, store, evaluator, model, and code identities are pinned;
- access, license, duplicate, leakage, and mapping audits are recorded;
- every declared train-only and locked-development run is complete;
- primary hypotheses and failures are reported with paired uncertainty;
- one final policy is selected by the frozen rule;
- the blind test is either executed once with explicit authorization or marked
  blocked with a concrete reason;
- independent result recomputation and clean reproduction pass;
- the claim--evidence register and manuscript are synchronized; and
- raw predictions, manifests, hashes, failures, and deviation logs are
  preserved.

No step in this plan authorizes publishing, external submission, paid API use,
or access to MOCHEG formal-test annotations.

