# Living Study Plan

Last updated: 2026-10-28

Active study status: **Pinned BGE-M3/SigLIP2 development Trial 2 and its shared matched-verifier comparison are measured and independently checked. The selector criteria pass and the verifier mean macro-F1 change is non-inferior at the declared point margin, but seed 13 is individually worse. Full-corpus label-blind retrieval and a clean rerun remain mandatory; the full development gate is false and the official MOCHEG test set remains untouched.**

Target quality: a defensible strong-journal or A* conference study. Q1/Q2 placement and acceptance are objectives, never guaranteed outcomes.

## 0. New-PC handoff contract

This section is the entry point for a new computer or a new ChatGPT session. It
must be read before running code or editing the paper.

### 0.1 Sources of truth

Use these files in this order:

1. plan.md: scientific protocol, current state, gates, stop rules, and next action.
2. paper/paper.tex and paper/sections/: the active IEEE pre-results manuscript.
3. experiments/configs/mocheg_development.yaml: frozen development choices.
4. experiments/configs/workstation_4090.yaml: hardware and execution profile.
5. research/mocheg_access_audit_2026-10-28.md: data identity and access evidence.
6. RUN_ON_4090_WORKSTATION.md: concise operator instructions.

If prose conflicts with executable code or configuration, stop and reconcile the
conflict before treating any result as valid. paper/ now belongs to the active
CEFES/MOCHEG study. The old SciVer/DACS manuscript is recoverable from Git
history and its frozen archives, but it is not the active paper.

### 0.2 State vocabulary

- **Proposed**: specified in this plan but not yet implemented.
- **Implemented**: code exists and targeted software tests pass.
- **Measured**: a declared run produced saved outputs.
- **Checked**: measurements were independently recomputed and linked to exact
  code, configuration, data, and feature identities.

At handoff, the exact CEFES decoder, calibration, MOCHEG adapter, packet builder,
pinned feature builder, and selector runner are implemented. The CPU hash smoke
is measured for software behavior only and is prohibited from supporting a
scientific claim. The pinned BGE-M3/SigLIP2 pilot, matched verifier, full-corpus
retrieval study, and active-study formal test are not measured.

### 0.3 Hardware and data assumptions

The destination PC has an RTX 4090 with 24 GiB VRAM, 64 GiB RAM, an Intel i9,
and about 1 TiB free storage. No paid API is required. MOCHEG and model weights
are public. A read-only Hugging Face token is optional for rate limits and must
be supplied through the destination environment or credential store, never
written into this repository, a notebook, a configuration file, or a commit.

The 69.2 GiB authoritative MOCHEG archive is a recovery source, not the default
download. The development runner uses pinned public mirrors and selectively
downloads the validated packet. Do not copy the old partial data cache to the
new PC.

### 0.4 Transfer options

After the handoff commit, the preferred complete transfer is the Git bundle
dist/cefes-study-handoff.bundle. Clone it into any folder. The resulting folder
containing plan.md is the study root; its absolute path and directory name are
not part of the protocol.

    git clone X:\Transfer\cefes-study-handoff.bundle my-study
    cd my-study
    git status
    git log -1 --oneline

The source-only alternative is dist/cefes-workstation-source.zip. It contains
the paper, plan, code, research records, and reference PDFs but no Git history,
public data, outputs, virtual environments, credentials, or old ZIP archives.
Extract it into any chosen folder; that extracted folder becomes the study root.

The GitHub remote is https://github.com/shasabbir/tpg-fact.git, branch main.
The handoff commit is local unless it is explicitly pushed later. A normal clone
from GitHub will not contain the handoff commit until that push occurs.

### 0.5 Exact first run on the destination PC

Open PowerShell anywhere inside the cloned repository, resolve the study root,
and enter the experiments directory:

    Set-Location -LiteralPath (git rev-parse --show-toplevel)
    Set-Location experiments
    powershell -ExecutionPolicy Bypass -File scripts\setup_workstation.ps1

The setup creates .venv-workstation, installs the formal, analysis, and
development dependencies, runs the complete test suite, and fails unless CUDA,
VRAM, RAM, and storage checks pass. If PyTorch is installed without CUDA, obtain
the current Windows/Pip/CUDA index URL from the official PyTorch selector and
rerun:

    powershell -ExecutionPolicy Bypass -File scripts\setup_workstation.ps1 -TorchIndexUrl "OFFICIAL_PYTORCH_INDEX_URL"

Then run the real semantic development pilot:

    powershell -ExecutionPolicy Bypass -File scripts\run_workstation_pilot.ps1

Expected audit targets:

- experiments/outputs/preflight/workstation.json
- experiments/outputs/mocheg/pilot_features.npz
- experiments/outputs/mocheg/pilot_features.manifest.json
- experiments/outputs/mocheg/selector_pilot/summary.json
- one per-seed JSON and claim-record JSONL file for seeds 13, 37, and 73.

Do not run a formal-test command. The current runner intentionally sets
full_development_gate_pass to false because the matched verifier has not been
evaluated.

### 0.6 Decision immediately after the run

First audit feature identities, scientific_feature_backend, candidate counts,
per-seed effects, calibration metrics, and raw claim records. If the selector
numeric gate fails, keep the test sealed and either use the remaining declared
development trials or stop this direction. If it passes, implement and run the
shared matched verifier, full-corpus retrieval comparison, and clean
reproduction. Only the complete Section 13 gate can authorize protocol freeze
and formal evaluation.

Never choose a favorable fixed budget after seeing the formal test, replace the
best fixed comparator with a weaker one, hide a failed seed, or describe the
hash smoke as a scientific pilot.

### 0.7 Continuation prompt for ChatGPT on the new PC

Use this prompt after opening the transferred repository:

> Continue the CEFES/MOCHEG study from this repository. Treat the current cloned
> repository folder containing plan.md as the study root; do not assume a fixed
> drive letter or directory name. Read plan.md in full,
> then read paper/paper.tex, every included paper section,
> RUN_ON_4090_WORKSTATION.md, experiments/configs/mocheg_development.yaml, and
> research/mocheg_access_audit_2026-10-28.md. Inspect Git status and the actual
> code before acting. Preserve the official MOCHEG test seal. Run the workstation
> preflight, tests, pinned BGE-M3/SigLIP2 400-claim development pilot, and audit
> its saved artifacts. Do not claim scientific success from the CPU hash smoke.
> Do not access formal-test annotations or rewrite pending results as completed.
> If the selector numeric gate passes, implement the shared matched verifier and
> full-corpus development comparison, then apply every Section 13 gate. Keep
> plan.md and the manuscript synchronized, generate numerical tables from frozen
> outputs, compile and inspect the PDF after manuscript changes, and commit only
> secret-free reproducible source.

### 0.8 Known handoff limitations

The active manuscript is a pre-results draft, not a submission-ready paper.
The automated bibliography audit records 48 references as verified and two as
pending automated metadata reconciliation. M-RAV was available online in 2026
but is assigned to a 2027 issue; the bibliography cites the online year and
notes the issue year. The Shah and Ganatra record has a creator-name parsing
difference in the helper cache even though the live Crossref record identifies
Minal Nirav Shah and Amit Ganatra. These documented warnings do not change the
scientific protocol, but they must be rechecked before submission readiness is
claimed.

## 1. Executive decision

The completed SciVer study is preserved as a checked negative result. Its formal five-seed archive passed artifact, code-hash, prediction, invariant, and grouped-bootstrap audits, but DACS-SCV obtained 0.4702 evidence-set F1 versus 0.4810 for fixed top-two. Predicting the gold citation count more accurately did not improve the selected evidence set.

That result is not retuned, hidden, or rewritten as a positive finding. It establishes the design problem for the new study:

> Citation-count accuracy is not the same objective as evidence-set utility.

The new study tests a method that directly optimizes the expected quality of the selected evidence set on a fresh public benchmark. The old SciVer official test is closed and cannot provide confirmatory evidence for the new method.

## 2. Study identity

Working title:

**Rethinking Evidence Selection in Multimodal Fact-Checking: Calibrated Adaptive Set Optimization Beyond Fixed Top-\(k\) Retrieval**

Working method name:

**CEFES** -- Calibrated Expected-F1 Evidence Selection.

Primary research question:

> Can a calibrated selector that chooses the evidence-set size by exact expected set-F1 outperform the strongest validation-selected fixed evidence budget while preserving fact-verification performance?

Primary contribution type: an evidence-selection and decoding method, supported by a controlled positive or falsifying experiment on the untouched MOCHEG test set.

## 3. Scientific rationale and novelty boundary

### 3.1 Evidence-backed problem

MOCHEG defines end-to-end multimodal evidence retrieval, verdict prediction, and explanation generation. Its official baseline retrieves a fixed top-$K$ set. The supplied M-RAV paper also tunes a global top-$K$ on MOCHEG development data and uses a fixed top-five evidence/alignment budget. These designs impose the same evidence count on claims whose evidence requirements differ.

The completed SciVer study supplies direct project evidence that count-prediction loss is misaligned with evidence-set F1. Fixed top-two beat the learned cardinality head even though the head had better count mean absolute error and exact match.

### 3.2 Closest competing ideas

- **MOCHEG**: separate text/image retrieval followed by fixed top-$K$ evidence and multimodal verification. DOI: `10.1145/3539618.3591879`.
- **M-RAV**: BGE/CLIP retrieval, image-text verbalization, relevance filtering, and fixed top-five evidence on MOCHEG. DOI: `10.1016/j.ipm.2026.104988`.
- **Utility-Oriented Visual Evidence Selection**: estimates visual helpfulness for multimodal RAG but still hands a fixed top-$K$ visual set to the main model and does not optimize a fact-checking evidence-set metric. DOI: `10.18653/v1/2026.acl-long.1620`.
- **Learning to Mine All Minimal Evidences**: searches for irreducible textual evidence sets with neural-symbolic reasoning, but does not address calibrated adaptive multimodal evidence-set decoding. DOI: `10.1016/j.ins.2025.122505`.
- **DACS-SCV**: predicts citation count from a frozen ranking; the audited SciVer result shows why matching count is insufficient.

### 3.3 Proposed distinction

CEFES does not predict a count class and does not use a global fixed budget. It estimates a calibrated probability that each text or image candidate belongs in the gold evidence set, then chooses the prefix whose **exact expected set-F1** is maximal. The decoder therefore optimizes the evaluation target rather than a surrogate count target.

The intended novelty is the combination of:

1. calibrated item-level gold-evidence probabilities;
2. exact expected-F1 decoding over variable evidence-set sizes;
3. a matched frozen-verifier comparison that isolates evidence selection;
4. multimodal fact-checking evaluation under an untouched official test;
5. duplicate, modality, evidence-count, and calibration analyses.

Novelty remains provisional until the focused literature update and implementation-equivalence audit are complete. No claim of being the first is allowed without that audit.

## 4. Public data and access

### 4.1 Primary dataset: MOCHEG

Official release and code: `https://github.com/PLUM-Lab/Mocheg`

Authoritative archive: `http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/latest_dataset/mocheg_with_tweet_2023_03.tar.gz`

Dataset license: CC BY 4.0. Associated code license: Apache 2.0.

Official paper statistics:

| Partition | Claims | Text evidence | Image evidence |
|---|---:|---:|---:|
| Train | 11,669 | 23,545 | 8,927 |
| Development | 1,490 | 4,067 | 1,178 |
| Test | 2,442 | 6,268 | 2,007 |

Labels are Supported, Refuted, and Not Enough Information. The released corpus described by the paper contains 91,822 documents and 122,246 images.

The form-exposed official archive is one 74,304,681,291-byte gzip stream. It is retained as the authoritative recovery source, but it is not the default transport because keeping both it and a full extraction would exceed a safe local storage budget.

Development uses pinned public mirrors only after identity checks. Train revision `985e4206cf9990ff0a875473616be204cfbd09c3` and development revision `9dabf73ad41ddecafb3b525fb82a3de018e3e378` reproduce the official claim, text-evidence, and gold-image-evidence counts. The shared `Corpus3.csv` mirror at revision `d9b15bf2b7d2be790c40efa960e8f05a58487ce2` has exactly 91,822 rows. The sealed test transport is pinned at `a72342a67b6bb5e930f8a2aa5fc3b051690c0f74`, but its annotations are not downloaded during development. The complete access audit is in `research/mocheg_access_audit_2026-10-28.md`.

### 4.2 Dataset roles

- **MOCHEG train**: model fitting and an internal calibration partition.
- **MOCHEG development**: model/configuration selection and the development pilot gate.
- **MOCHEG official test**: one confirmatory evaluation after protocol freeze.
- **SciVer**: completed retrospective diagnostic only; never used to select or confirm CEFES.
- **MuSciClaims**: excluded while its release lacks authoritative license metadata and provides only an evaluation split.
- **AIChartClaim/ChartCheck**: optional future external evaluation only after license and split terms are verified.

### 4.3 Execution resources

The primary execution host is a user-provided workstation with an RTX 4090
(24 GiB), 64 GiB system RAM, an i9 CPU, and approximately 1 TiB free storage.
This meets the declared full-run resource floor. The source project and small
manifests are transferred; pinned public data and model weights are downloaded
directly on that workstation. Kaggle remains an optional independent rerun, not
the main execution path.

### 4.4 Access and release gates

Before any model fitting, the MOCHEG snapshot must pass all of the following:

- authoritative archive identity, headers, and license source preserved;
- every transport mirror pinned by immutable revision and treated as non-authoritative for licensing;
- SHA-256 and byte-count manifest recorded;
- official split identities preserved;
- claim, text, and image counts reconciled with the release documentation;
- missing-media rates reported;
- exact claim/text/image duplicate checks completed;
- perceptual image duplicate checks completed;
- fact-check URL/source/event overlap across splits reported where metadata permit;
- verdict strings mapped deterministically to the three official labels;
- gold evidence identifiers verified against the released corpus;
- test labels, explanations, and gold evidence excluded from all development code paths.

If the release lacks the corpus or gold identifiers needed for controlled evidence selection, stop and document the gap before redesigning the task.

## 5. Test-sealing policy

The official MOCHEG test is treated as unavailable during development even if its file is physically present.

Implementation requirements:

1. `development` profiles may load only train and development annotations; the shared unlabeled retrieval corpus may be indexed only after columns that reveal claim association are removed from model inputs.
2. Formal test evaluation requires a frozen-protocol file containing code, configuration, dataset, feature, and checkpoint hashes.
3. Test evaluation requires an explicit `--formal-test` flag and refuses unfrozen checkpoints.
4. No test metric is printed during data preparation, feature extraction, or pilot execution.
5. Test annotations are downloaded only after the development gate and protocol freeze; test results are produced once for the frozen five-seed matrix and retained in full.
6. Any later method change makes the old test result exploratory; it cannot be presented as a fresh confirmation.

## 6. Task definition

For a claim $c$ and multimodal corpus $D$, the system must:

1. retrieve a bounded candidate packet $E=\{e_1,\ldots,e_n\}$ containing text paragraphs and images;
2. predict an ordered evidence set $S\subseteq E$;
3. predict a verdict in {Supported, Refuted, NEI};
4. optionally render an extractive evidence summary using only selected identifiers and bounded released text.

An evidence item is correct only when its immutable released identifier matches an official gold evidence item under the dataset's evaluation rule. Explanation fluency is not a primary contribution and no LLM-generated explanation is treated as proof of attribution.

## 7. Proposed method: CEFES

### 7.1 Candidate retrieval

Use the same candidate pool for every controlled selector:

- frozen BGE-family text embeddings for first-stage paragraph retrieval;
- frozen CLIP/SigLIP-family image embeddings for first-stage visual retrieval;
- reciprocal-rank fusion only when a unified candidate order is required;
- deterministic reranking to a bounded packet, initially at most 40 text and 20 image items;
- immutable evidence IDs, raw retrieval scores, modality, source, and duplicate-cluster metadata stored in the feature artifact.

The exact encoder revisions and candidate widths are selected on development data within the declared tuning budget, then frozen. Retrieval Recall@10/50 and gold-set ceiling coverage are reported separately from selection quality.

The 400-claim feasibility run is explicitly a **controlled selector pilot**: it
ranks the released qrel candidate universe by label-blind semantic similarity,
then retains at most 40 text and 20 image candidates. The final end-to-end run
must repeat the comparison on candidates retrieved from the full released
corpus. A positive qrel-candidate pilot alone is not a final scientific claim.

### 7.2 Shared evidence encoder and verifier

Reuse the checked dual-channel representation where practical:

- semantic claim/evidence features;
- separate pixel features with an availability mask;
- modality embeddings;
- claim-conditioned evidence scoring;
- compact evidence interaction layers;
- a verifier consuming the selected evidence representation.

The same verifier checkpoint, evidence features, and candidate packet are used for CEFES and all matched fixed/threshold/count baselines. This prevents a larger verifier from masquerading as a selection improvement.

### 7.3 Calibrated item inclusion probabilities

For each candidate $e_i$, a lightweight head predicts

$$
p_i=P(y_i=1\mid c,e_i,E),
$$

where $y_i$ indicates official gold-evidence membership. Inputs may include the frozen selector score, local score gaps, modality, visual availability, claim-verdict entropy, and duplicate-cluster features.

The default head is trained using class-balanced binary cross-entropy with fixed
$L_2$ regularization. A soft set-F1 term is reserved for a declared ablation and
cannot be added after inspecting formal-test results. Calibration uses a
claim-grouped fold carved from the official training partition. Temperature
scaling is the default; isotonic calibration is allowed only as a prespecified
ablation because it can overfit small strata.

### 7.4 Exact expected-F1 decoding

Sort candidates by calibrated $p_i$. For every allowed prefix $S_k$, compute

$$
U(k)=\mathbb{E}\left[\frac{2\,TP_k}{k+G}\middle|p_1,\ldots,p_n\right],
$$

where $TP_k$ is the number of gold items inside $S_k$ and $G$ is the total number of gold items in the packet. The expectation is calculated exactly under the declared independent-Bernoulli approximation using a Poisson-binomial dynamic program; it is not replaced by a ratio of expectations.

CEFES returns

$$
k^*=\arg\max_{k\in\mathcal{K}} U(k),\qquad S=S_{k^*}.
$$

Ties select the smaller set. The admissible set $\mathcal{K}$ and whether an empty set is allowed are determined from MOCHEG's official evidence semantics before the development pilot and then frozen.

The metadata audit found at least one positive qrel candidate for every official development claim. The development protocol therefore freezes `empty_selection_allowed: false`; train claims without any released qrel candidates are excluded from selector fitting but remain counted in the data audit. The evaluated evidence identity is the modality-qualified qrel `DOCUMENT#`, with identical duplicate rows collapsed and conflicting relevance treated as a release error.

The development evidence-count audit found a median of 6 and mean of 11.875
gold items in the sampled development claims, with 75 of 200 claims above ten.
Accordingly, the label-blind candidate ceiling is 60 and the fixed-budget family
is frozen at {1, 2, 3, 5, 10, 20, 40, 60} before the pinned semantic run. CEFES
uses the identical 60-item ceiling. This revision was made from train/development
statistics only; the formal test remained sealed.

### 7.5 Why this can address the observed failure

DACS minimizes count-classification loss. CEFES instead uses each item's calibrated probability and the expected set-F1 consequence of adding another item. It can reject a seemingly correct count when the next ranked item is unlikely to be gold, and it can select an additional item when the expected recall gain exceeds the precision cost.

This is a mechanistic hypothesis, not a promised result.

## 8. Research questions and hypotheses

### RQ1: evidence-set superiority

Does CEFES improve evidence-set macro-F1 over the strongest fixed evidence budget selected on MOCHEG development data?

**H1.** On the untouched official test, CEFES has a positive paired evidence-set F1 effect over the validation-selected fixed top-$k$ baseline, with a practically relevant mean improvement of at least 0.01.

### RQ2: count versus set-utility alignment

Does expected-F1 decoding outperform count prediction when both use the same candidate scores and verifier?

**H2.** CEFES improves evidence-set F1 over DACS while count MAE may be equal or worse, confirming that set utility rather than count accuracy explains the gain.

### RQ3: verification preservation

Does better evidence selection preserve downstream verdict quality?

**H3.** CEFES is non-inferior to the best fixed-budget baseline in verdict macro-F1 with a prespecified margin of 0.01.

### RQ4: robustness and modality

Where does adaptive selection help?

**H4.** Effects are reported by verdict, gold evidence count, text/image requirement, and duplicate/noise condition. Consistency across every subgroup is not required for the primary claim; rare strata remain exploratory.

## 9. Claim-test map

| Intended claim | Required comparison | Measurement | Unit and uncertainty | Failure action |
|---|---|---|---|---|
| CEFES improves evidence selection | CEFES vs validation-selected best fixed top-$k$ | evidence-set macro-F1 | paired claim/source-group bootstrap, five seeds | do not write a positive superiority paper |
| Set utility is better than count prediction | CEFES vs DACS on identical rankings/features | F1, precision, recall, count MAE | paired grouped bootstrap | narrow/drop mechanism claim |
| Verdict quality is preserved | CEFES vs best fixed top-$k$ with same verifier | verdict macro-F1 difference | paired grouped non-inferiority interval | drop preservation claim or method |
| Calibration matters | calibrated vs uncalibrated CEFES | F1, Brier, ECE | paired claims and seed variation | remove calibration contribution |
| Exact expected-F1 decoding matters | exact dynamic program vs ratio approximation/global threshold | evidence F1 | paired claims | simplify method and contribution |
| Multimodal behavior is credible | modality removal/swap and text/image strata | evidence/verdict effects | stratum-specific intervals | narrow multimodal claims |
| Duplicate robustness improves | injected syndicated/near-duplicate items | F1 and selected duplicate count | paired claims at each injection level | remove robustness claim |

## 10. Baselines and comparison fairness

Required controlled baselines:

1. official MOCHEG retrieval/verifier baseline where reproducible;
2. semantic relevance with fixed top-1, top-2, top-3, top-5, and top-10;
3. the single best fixed top-$k$ selected once on development data;
4. calibrated global probability threshold;
5. DACS count-classification head;
6. uncalibrated expected-F1 selector;
7. CEFES;
8. oracle ranking-cardinality and gold-evidence verifier diagnostics, explicitly labeled as upper bounds.

Flat and graph evidence scorers may both be evaluated, but the main selector comparison must use one shared frozen ranking/verifier family. M-RAV, MEVER, and other published numbers appear in a separate author-reported context table unless their exact data, inputs, and tuning access can be matched.

All trainable selector families receive the same development access and a maximum of 20 declared configuration trials. No baseline is intentionally under-tuned.

## 11. Metrics

### Primary

- evidence-set macro precision, recall, and F1;
- verdict macro-F1 for the non-inferiority requirement.

### Secondary

- joint verdict-plus-complete-evidence accuracy;
- per-class verdict F1 and balanced accuracy;
- Recall@5/10/50, MAP, and nDCG for candidate retrieval;
- evidence-set exact match and Jaccard score;
- selected evidence count, count MAE, and count exact match as diagnostics only;
- probability Brier score, negative log-likelihood, ECE, and reliability plots;
- text-only, image-involving, and mixed-evidence performance;
- performance by gold-set size and claim source;
- runtime, examples per second, peak GPU memory, and selected items per claim.

The manuscript must not use count accuracy as a substitute for evidence-set quality.

## 12. Statistical protocol

- Development pilot: seeds 13, 37, and 73.
- Frozen formal run: seeds 13, 37, 73, 101, and 137.
- Paired bootstrap with 10,000 samples.
- Group by fact-check source/event/article when reliable grouping metadata exist; otherwise use claim-level resampling and state the limitation.
- Average aligned seed-level records before the main grouped bootstrap and report seed dispersion separately.
- Report 95% confidence intervals and effect sizes.
- Apply Holm correction within the prespecified selector-comparison family.
- For H3, report a paired non-inferiority interval against the fixed-budget baseline with margin -0.01.
- Do not interpret non-significance as equivalence.

## 13. Development gate before consuming the test

The official test may be evaluated only if all conditions pass on the development set:

1. dataset and cross-split release gates pass or any retained limitation is explicitly accepted before training;
2. all scientific-logic unit and integration tests pass;
3. CEFES exceeds the validation-selected best fixed top-$k$ evidence F1 by at least 0.015 on average over seeds 13, 37, and 73;
4. the CEFES-minus-fixed effect is positive for all three pilot seeds;
5. verdict macro-F1 is no more than 0.01 below the best fixed baseline;
6. calibrated probabilities improve or preserve Brier score relative to the uncalibrated head;
7. no verifier checkpoint, candidate ranking, or feature artifact differs across matched selector comparisons;
8. a clean end-to-end rerun reproduces the development table from saved artifacts.

If the gate fails, do not inspect the test. Development may continue within the fixed 20-trial budget. If the budget is exhausted without passing, preserve the failed pilot and stop this direction instead of forcing a positive paper.

## 14. Formal success rule

The study supports the positive primary claim only when both conditions hold on the untouched official test:

1. CEFES minus the validation-selected best fixed top-$k$ has a positive 95% paired interval for evidence-set macro-F1 and a mean effect of at least +0.01;
2. the lower bound of the CEFES-minus-fixed verdict macro-F1 interval is above -0.01.

H2 and the ablations explain the gain but cannot replace H1. A favorable result against fixed top-one or top-five does not count if the validation-selected best fixed budget is stronger.

## 15. Ablations

Minimum required ablations:

- no calibration;
- global calibrated threshold instead of expected-F1 decoding;
- ratio-of-expectations approximation instead of exact decoding;
- BCE-only training without the soft set-F1 term;
- DACS count prediction instead of set decoding;
- no modality indicator;
- no duplicate-cluster feature;
- flat evidence scorer instead of graph interaction;
- text-only and image-disabled features.

Only ablations tied to a retained claim enter the main paper.

## 16. Robustness analyses

- inject 1, 5, and 10 duplicate evidence items from one source cluster;
- vary candidate-packet width without retuning the selector;
- remove all image candidates and all text candidates separately;
- corrupt a controlled fraction of top-ranked candidates with irrelevant items;
- report official full-test results and a label-independent de-duplicated subset if cross-split duplicates are found;
- analyze errors by evidence count, verdict, source, and modality requirement.

These analyses are secondary. They cannot rescue a failed primary comparison.

## 17. Implementation plan

Reusable scientific logic remains in Python modules; PowerShell and Python
scripts orchestrate setup, data preparation, feature extraction, and the pilot.
The internal package name trace_scv is retained to avoid a risky mechanical
rename before the development gate.

Implemented active paths:

```text
experiments/
|-- configs/
|   |-- mocheg_development.yaml
|   |-- mocheg_formal.yaml
|   `-- workstation_4090.yaml
|-- src/trace_scv/
|   |-- cefes.py
|   |-- data/mocheg.py
|   |-- mocheg_features.py
|   `-- selection/
|       |-- calibration.py
|       `-- mocheg_study.py
|-- scripts/
|   |-- prepare_mocheg.py
|   |-- build_mocheg_features.py
|   |-- run_mocheg_selector_pilot.py
|   |-- check_workstation.py
|   |-- setup_workstation.ps1
|   `-- run_workstation_pilot.ps1
`-- tests/
    |-- test_mocheg.py
    |-- test_mocheg_features.py
    |-- test_cefes.py
    |-- test_calibration.py
    `-- test_mocheg_study.py
```

Still to implement after a passing selector numeric gate:

- the shared matched verifier and verdict non-inferiority runner;
- full-corpus label-blind retrieval evaluation;
- protocol-freeze and formal-test commands with hash enforcement;
- formal five-seed aggregation, robustness, and manuscript asset generation.

A Kaggle notebook is deferred because the RTX 4090 workstation is the primary
host. It may be added later as an independent rerun wrapper around the same
Python package; core scientific logic must not exist only in a notebook.

## 18. Reproducibility outputs

Every run must preserve:

- dataset archive identity and license record;
- raw/processed manifests and split hashes;
- exact encoder/model revisions;
- feature-store hashes;
- configuration and code hashes;
- seed and environment metadata;
- training histories and selected checkpoints;
- complete candidate rankings;
- calibrated per-item probabilities;
- selected evidence IDs and verdict probabilities;
- raw per-claim comparison records;
- derived metrics, bootstrap samples/seeds, and tables;
- artifact inventory with byte counts and SHA-256 values.

All paper tables and figures must be generated from frozen output artifacts. The archive's internal `scientific_results_validated` flag remains false until an independent recomputation audit passes.

## 19. Manuscript policy

The active paper is now the CEFES/MOCHEG pre-results manuscript in paper/.
It uses the conventional structure Introduction, Related Work, Task and Data,
Method, Experimental Setup, Results, Discussion, Limitations and Ethical
Considerations, and Conclusion. It records checked readiness facts and explicit
pending work but makes no active-study performance claim.

The old SciVer/DACS manuscript is preserved by Git history and its audited
archives. Its negative result may motivate CEFES, but its consumed official test
cannot confirm or tune the new method.

After each checked experimental milestone:

1. update result prose only from frozen artifacts and independent recomputation;
2. preserve negative, failed, and inconclusive comparisons;
3. generate numerical tables and plots from scripts rather than manual copying;
4. keep method, configuration, diagram, outputs, and claims synchronized;
5. compile and visually inspect the current PDF;
6. run reference, requirement, and clean-package audits before any readiness claim.

For a Q1-level journal claim, one positive untouched benchmark may still be insufficient. A second licensed external evaluation is desirable, but it must not delay or contaminate the MOCHEG confirmatory test. Q1/Q2 status is verified against the chosen ranking database and year at submission time.

## 20. Risks and fallbacks

| Risk | Consequence | Planned response |
|---|---|---|
| MOCHEG archive unavailable or incomplete | cannot run controlled retrieval | stop and request the official archive; do not rebuild from live web pages |
| official test overlaps train/dev | inflated benchmark result | report overlap, use label-independent de-duplicated analysis, narrow claims |
| first-stage retrieval misses gold evidence | selection ceiling dominates | improve retrieval on development within budget; report oracle ceiling |
| expected-F1 independence approximation is poor | decoder miscalibrated | compare exact decoder with threshold/count baselines and calibration diagnostics |
| development gain fails | no reason to consume test | revise only within budget or stop direction |
| test gain fails | no positive superiority paper | preserve result; do not retune on test |
| verdict degrades | evidence set not operationally useful | fail non-inferiority gate |
| effect exists only in one subgroup | weak generalization | narrow claim and seek external validation |
| compute exceeds Kaggle limits | incomplete five-seed matrix | cache frozen features, reduce batch size, never reduce required seeds silently |

## 21. Milestones

### M0 - completed: direction and protocol

- preserve the audited SciVer negative result;
- select MOCHEG as the fresh public benchmark;
- verify MOCHEG paper statistics and CC BY 4.0 release terms;
- define CEFES, closest-work boundary, hypotheses, metrics, gates, and stop rules;
- update the living plan and dataset registry.

### M1 - completed: obtain and audit MOCHEG

- submit the official researcher form and obtain the dataset link;
- store the untouched archive outside generated experiment outputs;
- record archive identity and license;
- implement schema inspection and preflight;
- create train/development/test manifests;
- implement test sealing and duplicate/media audits.

Exit criterion met for the development packet: pinned metadata, reconstructed
sentence text, and bounded media pass the declared gates. Full-corpus and
formal-test identities remain governed by later milestones.

### M2 - completed: implementation and synthetic verification

- implement MOCHEG adapter and candidate artifacts;
- implement exact expected-F1 dynamic program with brute-force small-case tests;
- implement calibration and CEFES selection;
- integrate fixed, threshold, and DACS baselines;
- pass unit tests and a miniature end-to-end run.

Exit criterion met: 76 tests pass and the end-to-end hash smoke is explicitly
non-scientific with no official-test metrics.

### M3 - development pilot

- extract/cache real MOCHEG features;
- run three development seeds;
- use no more than 20 declared selection configurations;
- analyze retrieval ceiling, calibration, selector effects, runtime, and errors;
- apply the full development gate.

Exit criterion: frozen-protocol recommendation of `advance` or `stop`.

### M4 - frozen formal run

- freeze code/config/data/features/checkpoints and test identities;
- run five seeds once on the official test;
- export complete artifacts and inventory;
- independently recompute all endpoints and comparisons.

Exit criterion: audited formal archive and an explicit positive/negative decision under Section 14.

### M5 - manuscript construction

- proceed only after an audited positive formal result;
- rewrite the paper around CEFES and MOCHEG;
- retain the SciVer count-utility mismatch as motivation/diagnostic, not confirmation;
- complete literature, reference, claim-evidence, render, and clean-package audits.

## 22. Current state table

| Item | State | Evidence/version | Unresolved issue | Next action |
|---|---|---|---|---|
| SciVer DACS result | checked, negative | `dacs-scv-formal-outputs.zip` and independent audit | none; test consumed | preserve only |
| MOCHEG license/statistics | checked from official paper/repository | CC BY 4.0; 15,601 claims; official splits | none for metadata | preserve authority record |
| MOCHEG transport | 400-claim packet gate passed | pinned train/dev mirrors; 91,822-row corpus; 1,612/1,612 pilot images readable; zero missing pilot gold candidates | full media integrity not measured; 3 train gold images unresolved outside pilot | scale acquisition on 4090 workstation |
| MOCHEG data quality | sentence candidate gate passed | all 378,864 distinct train/dev qrel sentence IDs recovered; 4,556 blank non-qrel train evidence texts retained; duplicate policy audited | image-candidate integrity pending | finish bounded media pilot |
| CEFES concept | Trial 2 selector and matched verifier checked | exact decoder, guarded calibration, shared verifier checkpoint per seed, three seeds, paired bootstrap | threshold control remains stronger for evidence F1; verifier seed 13 is negative; full-corpus/clean rerun pending | run full-corpus comparison |
| exact expected-F1 decoder | implemented and checked | dynamic program matches brute-force enumeration; edge policies tested | independence/calibration behavior unmeasured | connect to calibrated candidate probabilities |
| MOCHEG adapter | implemented and checked | both schemas normalized; candidate records deduplicated; 378,864/378,864 sentence IDs resolved; matched 400-claim packet passes; test sealed | encoder features pending | build pilot feature store |
| development claim | partial advance; full gate false | selector mean gain +0.06833; verifier mean macro-F1 change -0.00013; independent audits pass | threshold beats CEFES evidence F1 all seeds; verifier seed 13 is -0.01594; full-corpus effect unknown | full-corpus retrieval, clean rerun |
| formal positive claim | unavailable | official test sealed | pilot/protocol not frozen | do not evaluate test |
| CEFES manuscript | implemented as pre-results draft | conventional IEEE structure synchronized with this plan; no scientific performance claim | GPU pilot, matched verifier, formal results, generated result figures pending | update only from checked artifacts |

### 22.1 RTX 4090 workstation execution record (2026-10-28)

The destination environment now uses Python 3.13.14 and PyTorch
2.14.1+cu130. Strict preflight measured 32 logical CPU threads, 63.82 GiB RAM,
1,298.77 GiB free storage, and an NVIDIA GeForce RTX 4090 with 24,564 MiB
VRAM. CUDA is available and the preflight status is `pass`.

The first setup attempt installed the default CPU-only PyTorch 2.14.1 wheel.
The preflight correctly returned failure, but Windows PowerShell continued and
printed a false success because native exit codes were not checked. This
recoverable orchestration defect was fixed by a shared fail-closed native-command
guard used by both workstation scripts, and a regression test now exercises a
nonzero child-process exit. No scientific experiment was run under the failed
environment. The full suite passes after the fix.

The selector export was also hardened before the semantic run to retain every
bounded per-seed candidate record, raw and calibrated probabilities, ranks,
selection flags, complete bootstrap samples and resampling seeds, code/config
hashes, artifact hashes, and runtime/environment metadata. These changes affect
auditability only; they do not change the frozen selector, candidate ceilings,
seeds, fixed-budget family, or numeric gate.

### 22.2 M3 development Trial 1: pinned temperature scaling

Trial 1 used the frozen BGE-M3 revision
`5617a9f61b028005a4858fdac845db406aefb181` and SigLIP2 revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` on the checked 400-claim packet.
The feature artifact has SHA-256
`c4ad754255cec61c7988b5ac9ca84bbd286b40e6c48a79157cb896d982a32c8a`,
15,289 rows, 4,156 gold rows, and 367 claims with at least one available
candidate. Feature construction took 626.16 seconds and recorded peak allocated
GPU memory of 1,714,396,672 bytes.

The independently recomputed selector results are:

| Seed | selected fixed k | CEFES F1 | fixed F1 | difference | raw Brier | calibrated Brier | paired 95% interval |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 20 | 0.71273 | 0.65313 | +0.05961 | 0.150434 | 0.149620 | [0.04134, 0.07877] |
| 37 | 20 | 0.71811 | 0.65603 | +0.06208 | 0.150813 | 0.151011 | [0.04331, 0.08124] |
| 73 | 10 | 0.72176 | 0.64519 | +0.07657 | 0.152813 | 0.151061 | [0.05118, 0.10209] |

Mean CEFES F1 is 0.71753, mean fixed F1 is 0.65145, and the mean paired
difference is +0.06608. The effect is positive in all seeds, but calibrated
Brier worsens by 0.00019784 for seed 37. The selector numeric gate therefore
fails. The global calibrated 0.5-threshold control also exceeds CEFES F1 in all
three seeds (0.73351, 0.75153, and 0.73979), a negative mechanism finding that
must remain visible even if a later trial passes the fixed-budget gate.

Independent recomputation found zero discrepancies across feature/sidecar
hashes, per-candidate probabilities, temperature transforms, candidate ranks,
claim-level selections, aggregate F1/Brier values, and all 10,000 bootstrap
samples per seed. The deterministic 23-member development archive has SHA-256
`96f38d39945c5ce6951cface7d6016a9fb02245d3c097d2d7d68c17d6dfccc69`.
It records `formal_test_annotations_accessed: false` and preserves the failed
gate outcome.

Trial budget used: 1 of 20. Trial 2 is declared before execution as a
claim-grouped calibration safety guard. Within each seed's training calibration
fold, one claim-disjoint subset fits the temperature and a second chooses
between that temperature and identity scaling by Brier score. Validation labels
do not choose the scaler. Candidates, features, scorer settings, fixed budgets,
seeds, and gate thresholds remain unchanged. Trial 1 remains the default-method
result and cannot be hidden or overwritten.

### 22.3 M3 development Trial 2: calibration safety guard

Trial 2 reused the exact Trial 1 feature artifact and changed only calibration.
For each seed, the original training calibration claims were split again by
claim identity. One subset fit a candidate temperature and the other retained
that temperature only when its Brier score did not exceed identity scaling.
No validation label selected the scaler. Seeds 13, 37, and 73 applied
temperatures 1.0, 0.912246, and 0.756374, respectively.

The independently recomputed selector results are:

| Seed | selected fixed k | CEFES F1 | fixed F1 | difference | raw Brier | guarded Brier | paired 95% interval |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 20 | 0.71248 | 0.65313 | +0.05935 | 0.150434 | 0.150434 | [0.04146, 0.07779] |
| 37 | 20 | 0.72091 | 0.65603 | +0.06489 | 0.150813 | 0.149584 | [0.04551, 0.08429] |
| 73 | 10 | 0.72595 | 0.64519 | +0.08076 | 0.152813 | 0.150676 | [0.05485, 0.10677] |

Mean CEFES F1 is 0.71978, mean fixed F1 is 0.65145, and the mean paired
difference is +0.06833. The difference is positive in every seed and guarded
Brier is improved or preserved in every seed, so the selector numeric criteria
pass. Independent recomputation again found zero discrepancies. The
deterministic 23-member Trial 2 archive has SHA-256
`5d62d1250cc13b296a9e7e0972cd3fbdb00d3ca5e002e9f2d42ec11aab722b8e`
and records `formal_test_annotations_accessed: false`.

This is a selector-only advance, not a complete development-gate pass. The
0.5-threshold control still exceeds guarded CEFES F1 for every seed (0.73351,
0.75153, and 0.73979). This weakens the exact-decoder mechanism claim and must
be carried into the matched/full-corpus decision. Trial budget used: 2 of 20.
Section 13 conditions 5, 7, and 8 remain open because the shared verifier,
full-corpus matched comparison, and clean end-to-end reproduction are pending.

### 22.4 Shared matched verifier on the 400-claim packet

The verifier uses one balanced three-class logistic checkpoint per seed for all
selection methods. It is fitted only on claim-grouped training-fit claims; the
method-specific input is an aggregation of the selected evidence features and
scores. No verdict label or gold evidence flag is exposed as an input. The
saved prediction records link every method to the same checkpoint hash within
each seed and retain probabilities, selected evidence IDs, labels, and 10,000
claim bootstrap samples.

| Seed | CEFES macro-F1 | fixed macro-F1 | difference | threshold macro-F1 | paired 95% interval |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.48093 | 0.49687 | -0.01594 | 0.47227 | [-0.05278, 0.02048] |
| 37 | 0.55102 | 0.54202 | +0.00900 | 0.52089 | [-0.01665, 0.03687] |
| 73 | 0.52184 | 0.51528 | +0.00656 | 0.48143 | [-0.03417, 0.04903] |

The mean CEFES-minus-fixed macro-F1 change is -0.00013, which is above the
declared -0.01 point non-inferiority margin. The direction is heterogeneous:
seed 13 falls by 0.01594 and neither the mean nor any seed interval excludes
zero. The result therefore satisfies the plan's mean point-margin check but is
not evidence of a verdict improvement. An independent auditor recomputed every
macro-F1/accuracy value, probability argmax, artifact hash, and bootstrap
sample and found zero discrepancies. Formal-test annotations remain absent.

## 23. Immediate next action

Trial 2 passes the selector numeric criteria and the checked shared verifier
passes the declared mean point non-inferiority margin without showing a verdict
improvement. Implement the full-corpus label-blind retrieval comparison, then
perform a clean end-to-end reproduction while preserving the stronger
threshold-control and negative seed-13 verifier findings. Do not download the
69.2 GiB archive unless a required public asset is otherwise unavailable, and
do not inspect formal-test annotations. Apply every Section 13 condition before
any protocol freeze; these partial results do not authorize the formal test.
