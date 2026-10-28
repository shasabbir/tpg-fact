# Living Study Plan

Last updated: 2026-10-28
Study status: **formal five-seed SciVer run complete and independently audited; the frozen release gate, artifacts, endpoints, invariants, and grouped-bootstrap comparisons pass; the prespecified adaptive-superiority hypothesis fails because DACS-SCV underperforms fixed top-two; the ten-page IEEE manuscript is synchronized, compiled, and visually verified as a controlled negative study with no post-test tuning**
Primary venue class: strong Q1 journal or A* conference extension; no quartile or acceptance is guaranteed.

## 1. Study identity

Working title:

**Decoupling Citation-Set Size from Verification: A Controlled Study of Multimodal Scientific Claims**

Working method name: **DACS-SCV** (Decoupled Adaptive Citation-Set Scientific Claim Verification).

Central claim to test:

> Separating verdict prediction from citation-set sizing can improve evidence-item citation quality without changing the verifier's decision or evidence ranking.

This replaces the former US-TPMFC/decision-time-provenance study. The old private pilot, ClaimReview collection path, and temporal benchmark construction are out of scope.

## 2. Why this direction

The two supplied base papers are strong but leave the same unresolved problem.

- **M-RAV** retrieves text and images, verbalizes image--text consistency, ranks by relevance, and prompts an LLM to verify a claim. Its free-form verbalizations are not tied to explicit evidence identifiers, retrieval quality for images is weak, and its relevance module is neutral or harmful in some reported ablations.
- **MEVER** jointly retrieves, verifies, and explains with a multimodal graph. Its consistency regularizer aligns the explanation with the predicted label, but label consistency is not evidence attribution. Mean aggregation can also conceal which evidence actually caused a decision. Explanation evaluation is dominated by lexical overlap and a VLM judge.

The reviewed peer-reviewed literature already covers multimodal fusion, out-of-context detection, evidence retrieval, scientific claim verification, generated explanations, citation evaluation, and visual-evidence utility. The corrected pilots falsified the planned causal-grounding mechanism but exposed a separate design problem: a shared fixed top-$k$ bottleneck couples verdict computation to citation length even though required evidence-set sizes vary by claim. DACS-SCV therefore freezes the verifier and ranking path and learns citation-set cardinality separately. The literature baseline contains 50 DOI-verified works and excludes arXiv-only sources.

## 3. Research questions and hypotheses

### RQ1: fixed-budget limitation

How much citation precision and recall are lost when every claim receives the same citation budget?

**H1.** Fixed top-$k$ selection over-cites claims requiring short evidence sets and under-cites claims requiring longer sets.

### RQ2: decoupled adaptive selection

Does a separately trained citation-cardinality head improve evidence-item citation F1 over the same frozen graph ranking?

**H2.** DACS-SCV improves citation macro-F1 over each prespecified fixed top-one, top-two, and top-three selection rule while preserving identical verdict probabilities and predictions by construction.

### RQ3: ranking and cardinality

Are gains attributable to adaptive set sizing rather than a changed evidence ranking or verifier?

**H3.** With ranking and verdicts held fixed, adaptive sizing improves the precision--recall balance and citation F1 relative to fixed budgets.

### RQ4: subgroup robustness

Do attribution gains persist across unseen SciVer source papers and across text-only versus visual-required claim strata?

**H4.** DACS-SCV's citation effect is reported separately for text-only and visually grounded claims and across gold evidence-set sizes, with rare strata treated as exploratory.

## 4. Public datasets only

No new claims, articles, images, or labels will be collected from the web. Official public releases are used through download scripts; copyrighted source papers or images are not republished beyond their licenses.

### Formal datasets

1. **SciVer** -- 3,000 expert-annotated examples from 1,113 papers with text, tables, figures, supporting evidence, and rationales. Official Hugging Face release `chengyewang/SciVer`, public, ungated, CC BY 4.0. This is the primary dataset.
2. **MuSciClaims (conditional, disabled)** -- a relevant public Hugging Face release with multimodal scientific claims, but the main release currently lacks authoritative license metadata. Its adapter remains off unless reuse terms are confirmed before the formal freeze.
MOCHEG and AIChartClaim remain related-work resources rather than formal datasets. They are not downloaded or evaluated by the frozen pipeline.

### Dataset gates

A dataset enters the main tables only if:

- its official release is publicly accessible when the experiment is frozen;
- the license or author usage terms permit research use;
- all required modalities are present for at least 90% of the selected split;
- exact and perceptual duplicate checks are reported;
- claim/evidence identifiers and official splits can be frozen in a manifest;
- no test labels or explanations leak through file names, URLs, or prompt fields.

If MuSciClaims reuse terms cannot be confirmed, it is omitted and the formal study is limited to SciVer.

## 5. Task definition

For claim `c` and candidate evidence set `E`, the system outputs:

1. verdict probabilities over the dataset's official label set;
2. an ordered set of cited evidence identifiers `S`;
3. a deterministic extractive rationale composed only from selected evidence identifiers and bounded snippets.

An evidence identifier refers to an immutable dataset item: a text block, table, figure, image, or linked caption. A citation is correct only when it points to an official gold evidence item for that claim or passes the dataset-specific support rule.

## 6. Proposed method

### 6.1 Frozen dual-channel features and evidence graph

Encode the claim and the semantic content of every evidence item with frozen BGE-small-en-v1.5; this includes passages, captions, and serialized tables. Encode recoverable media separately with frozen CLIP ViT-B/32 and store an explicit pixel-availability mask. Separate trainable projections and a learned scalar gate fuse the semantic and pixel channels before adding the modality-type embedding. Missing media leaves the pixel channel unavailable and never suppresses the semantic channel. The claim is projected separately and is **not** inserted as a graph node. The graph contains at most 12 valid evidence nodes and uses two masked self-attention layers over a complete within-packet adjacency matrix. The implementation is modality-typed at the node level, not relation-typed at the edge level.

### 6.2 Frozen verifier and evidence ranking

A claim-conditioned MLP scores `[claim; evidence; claim*evidence]`. The matched base verifier uses its fixed top-three aggregate for verdict prediction. After training, this verifier and its complete ranked evidence list are frozen. DACS-SCV never changes the base checkpoint, verdict probabilities, or evidence order.

### 6.3 Decoupled adaptive citation cardinality

For each claim, DACS-SCV sorts the frozen selector probabilities and appends normalized counts of valid and pixel-backed candidates. A two-layer MLP predicts a citation count from one through the maximum gold count observed in training. It is trained only on training labels and selected on the grouped development validation split. Inference returns the predicted number of identifiers from the unchanged ranking. The renderer normalizes whitespace, truncates cited snippets to 180 characters, and emits stable identifiers; invented identifiers are impossible by construction.

### 6.4 Training and inference separation

The base verifier is trained with verdict cross-entropy, supervised evidence-selection BCE, and a small sigmoid sparsity penalty. The cardinality head is then trained with cross-entropy while the verifier is frozen. Development validation chooses the cardinality checkpoint by citation macro-F1; the official test is evaluated once after the full protocol is frozen. Because citation sizing is downstream of the verifier, DACS-SCV and its matched fixed-budget baseline must have identical verdict-probability records.

### 6.5 Resource profile

The reference implementation uses frozen BGE-small text features and frozen CLIP ViT-B/32 image features with a compact graph/selection head. Pilot and formal encoding batch unique text and media inputs and reuse one encoder instance across all splits. A deterministic hash/text-caption backend is restricted to synthetic smoke tests. Validation-only pilots run locally on CPU; the frozen full-data, five-seed matrix will run once on Kaggle and reuse fingerprint-bound feature stores and checkpoints.

## 7. Comparisons

### Required baselines

1. majority label;
2. claim-only text model;
3. mean pooling over all candidate evidence without selection;
4. scaled dot-product relevance over trainable claim/evidence projections with top-$k$ evidence;
5. learned flat selector without graph message passing;
6. graph selector with fixed top one, top two, and top three outputs;
7. graph selector with training-mode citation count;
8. DACS-SCV with a learned, decoupled citation count.

M-RAV and MEVER remain published context because their original tasks, data variants, and resource profiles are not controlled comparisons on the grouped SciVer protocol. Their published values must not enter paired statistical tests against local runs.

## 8. Evaluation

### Primary endpoints

- evidence-item citation macro-F1;
- verdict macro-F1.

Both are required. A method is not declared better if attribution improves by sacrificing more than one absolute verdict-F1 point.

### Secondary endpoints

- evidence Recall@k, MAP, and nDCG@k;
- accuracy, per-class F1, and balanced accuracy;
- citation coverage and bounded extractive-rationale coverage;
- citation-count mean absolute error, exact match, and predicted versus gold mean;
- citation precision and recall by gold evidence-set size;
- text-only and visually grounded citation macro-F1;
- diagnostic fixed-ranking oracle-cardinality performance, clearly labeled as an upper bound;
- runtime, examples per second, peak GPU memory, and selected evidence count.

### Statistical protocol

- tune once on validation data, then freeze configuration;
- five seeds for trained systems after the pilot;
- paired bootstrap over aligned predictions, grouped by source paper and averaged over shared seeds, 10,000 samples;
- 95% confidence intervals;
- exact seed-level sign-flip effects as a complementary stability check;
- Holm correction within the verdict and attribution comparison families;
- group resampling by source paper when multiple claims share one paper;
- report effect sizes and intervals, not only p-values.

## 9. Ablations

Remove or replace one component at a time:

- evidence graph layers (flat evidence set);
- graph evidence encoder versus flat selection;
- fixed versus adaptive citation count;
- joint versus decoupled adaptive selection;
- selector-score profile without visual-availability count;
- selector-score profile without valid-candidate count;
- text-only and pixel-disabled evidence features.

## 10. Falsification and stop rules

- DACS-SCV and the matched fixed-ranking verifier must have identical verdict outputs; any mismatch invalidates the decoupling claim.
- If the five-seed official-test citation effect is non-positive against any of the matched fixed top-one, top-two, or top-three rules, do not claim adaptive improvement; report the method as unsuccessful. Report all three paired grouped-bootstrap intervals and apply the prespecified Holm correction to this comparison family.
- If gains occur only in one-item claims or disappear in visually grounded claims, narrow the contribution accordingly.
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

### M1 -- completed: data feasibility implementation

- [completed 2026-10-26] run the public dataset preflight notebook;
- [completed] pin exact revisions and implement dataset manifests;
- [completed] verify the real SciVer schema and preserve its official test file;
- [implemented; outcome pending formal download] run modality coverage, missing-media, exact content/media duplicate, perceptual duplicate, and license gates as part of the split manifest.

### M2 -- completed in code and synthetic integration

- implement evidence-item citation scoring;
- validate citation-only sufficiency and selected-citation deletion comprehensiveness, with gold-evidence deletion retained as a diagnostic;
- run deterministic intervention unit tests;
- establish claim-only and relevance-only baselines.

### M3 -- current: encoder-valid public-data pilot

- [completed 2026-10-26] run and audit the v4 one-seed hash-feature engineering pilot;
- [completed] confirm end-to-end execution, T4 fit, checkpoint loading, aligned predictions, and non-degenerate citation selection;
- [completed] record that the v4 archive is not independently recomputable and that its ordered 300-row official-test slice is label-skewed;
- [completed] replace unsafe prefix sampling, add full audit exports, bind checkpoints to input/configuration fingerprints, and enforce zero source-paper overlap across splits;
- [completed 2026-10-26] run and independently audit the v5 one-seed pilot with frozen BGE/CLIP features and real media on 600 training and 150 validation examples; the official test was not evaluated;
- [passed mechanically] verify the complete artifact inventory, code/input hashes, checkpoint fingerprints, zero group overlap, aligned predictions, and exact metric/bootstrap recomputation;
- [passed provisionally] against TRACE-no-CF, improve citation macro-F1 by 1.73 points while limiting the verdict macro-F1 decrease to 0.79 points;
- [failed mechanism gate] selected-citation comprehensiveness remains negative, visual removal improves the verdict, and the full model is less sensitive to visual swaps than TRACE-no-CF;
- [completed in v6 code] align training and evaluation on exact selected-citation deletion, make visual swaps modality-matched, report swap coverage, add modality/duplicate audits, and disable the unlicensed MuSciClaims path by default;
- [completed 2026-10-27] independently audit the returned v7 pilot archive; all artifact, code, checkpoint, alignment, primary-metric, seed-comparison, and grouped-bootstrap checks pass;
- [failed readiness gate] TRACE-SCV gains 1.73 citation-F1 points over TRACE-no-CF but loses 1.94 verdict-F1 points, has negative selected-citation comprehensiveness, has lower visual-swap sensitivity, and improves when visual evidence is removed;
- [completed 2026-10-27] implement separate semantic-text and pixel-visual evidence channels with masked learned fusion;
- [completed] make pixel swaps/removal preserve semantic content, identifiers, node types, and graph nodes;
- [completed] export per-claim clean, selected-deleted, gold-deleted, sufficient, irrelevant, duplicate, swapped, semantic-removed, and pixel-removed probabilities plus swap and visual-grounding flags;
- [completed] export item-level cross-split duplicate records with fingerprint, claim, group, evidence ID, modality, and gold role, and add a formal data-release gate;
- [completed] update the methodology, architecture figure, configuration, tests, Kaggle discovery/version guard, and run instructions for v8;
- [completed 2026-10-27] mechanically audit the returned v8 pilot: all archive, code, checkpoint, alignment, and metric checks pass, but the data-release gate detects six cross-split gold-content overlaps;
- [invalidated] trace the overlaps to hierarchical section matching that returned empty parents before exact annotated children; v8 diagnostic metrics cannot support any manuscript claim;
- [completed] implement v9 exact-first subsection recovery, a flagged gold fallback for genuinely missing annotations, split-level fallback counts, and two regression tests;
- [completed] reconstruct the identical 600/150 selection locally with all 363 paper records and 357 referenced images: visual-media coverage is 165/165 train and 44/44 validation, fallback count is zero, and group/content/exact-visual/near-visual overlap checks all pass;
- [hard stop] run one corrected v9 validation-only pilot; if its causal/visual gates fail, retire the causal-grounding claim and pivot to a narrower citation-selection study.
- [completed 2026-10-27] run the corrected v9 pilot locally on CPU using the frozen BGE/CLIP features; preserve the output archive and independently reproduce checkpoint fingerprints and primary metrics;
- [passed] corrected data-release gate, artifact inventory, claim alignment, and metric recomputation;
- [failed scientific gate] causal TRACE-SCV loses 1.33 verdict-F1 points, gains only 0.51 citation-F1 points, retains negative comprehensiveness, is less swap-sensitive, and improves after visual removal;
- [decision] retire the causal-grounding hypothesis and do not run its formal matrix;
- [completed] test a joint adaptive citation selector; it predicts count accurately but fails the citation/verdict gate because it changes ranking and representation;
- [completed] implement and independently audit DACS-SCV, which freezes the verifier/ranking and learns citation count downstream;
- [passed pilot gate] validation citation macro-F1 rises from 0.4851 to 0.5303 while every verdict-probability vector remains unchanged; grouped-bootstrap uncertainty still includes zero, so this is a feasibility signal only;
- [freeze] no further method tuning on the validation split; complete final-path integration, tests, audit exports, manuscript synchronization, and Kaggle packaging.

### M4 -- completed: frozen main experiments

- [completed] freeze configuration, inputs, code hashes, and output inventory;
- [completed] run five seeds, baselines, and ablations on SciVer;
- [completed] run grouped SciVer subgroup and intervention analyses; omit cross-dataset transfer because no external release cleared the frozen license gate;
- [completed] generate all numerical tables and the formal analysis figure from the frozen archive.

### M5 -- completed: paper construction

- [completed] replace pre-results statements with audited outputs and preserve the failed primary hypothesis;
- [completed] complete limitations, reproducibility, ethics, discussion, and conclusion;
- [completed] compile and visually inspect the ten-page main paper; no supplement is required for the current study artifact;
- [submission task] verify the selected venue's current requirements and quartile, then add its required declarations and forms.

## 13. Current decisions and next action

Decisions:

- public datasets only;
- no ClaimReview/GDELT/Common Crawl collection;
- no new benchmark claim without a released dataset manifest;
- no generic sufficiency/abstention novelty claim;
- paper uses conventional section names;
- notebook orchestrates experiments, while reusable logic remains in `.py` modules.

Next action:

> Freeze the audited study and manuscript. The remaining work is venue selection and submission administration: verify the venue's current requirements and quartile, add required funding/conflict/data-and-code declarations, and build the final venue-specific source package. Preserve the failed adaptive-superiority result. Do not tune a revised cardinality method on the consumed SciVer official test or present any such revision as confirmatory.

## 14. Validation record

Verified through 2026-10-28:

- all 37 unit/integration tests pass after the transparency-safe correction, including fixed top-one/top-two exports, DACS count-feature ablations, checkpoint/ranking safeguards, transient-rate-limit retry behavior, referenced-file selection, transparent-image compositing, a clean complete smoke run, and checkpoint reuse;
- the returned executed notebook proves the v1 formal attempt stopped during the initial SciVer snapshot at an HTTP 429 from Hugging Face's Xet token endpoint; no preprocessing, feature extraction, training, or official-test inference ran;
- runner v2 requires an authenticated `HF_TOKEN`, disables Xet, downloads only release-referenced SciVer assets, caps Hub concurrency, retries transient failures with a bounded wait, and persists data/features/checkpoints/outputs outside the replaceable source checkout so same-session reruns resume safely;
- the second Kaggle attempt reached deterministic dataset preparation but stopped before feature extraction because three distinct transparent `LA` figures collapsed to the same zero dHash; their raw SHA-256 values, captions, papers, and white-composited dHashes differ;
- runner v3 composites transparent images onto white before both dHash auditing and CLIP encoding; a full local 3,000-row reconstruction then passes the unchanged release gate with zero group overlap, zero exact visual overlap, and zero exact gold-content overlap;
- the frozen v3 upload was rediscovered from a renamed nested input in an isolated workspace, verified as package `0.3.2`, and completed all 11 smoke variants with a 56-file output inventory; its exact external archive identity is recorded in `research/final_protocol_freeze_2026-10-27.md`;
- the returned formal archive is 86,517,243 bytes with SHA-256 `06FE5026B871197B9768DDF31027A7FD47E3F050882084166374229A6B567F7C`; all eight notebook code cells completed without errors;
- all 252 declared artifacts, 29 code hashes, 100,000 recomputed endpoint records, 50,000 matched invariant rows, seed comparisons, and 10,000-sample grouped-bootstrap comparisons pass independent audit;
- DACS-SCV citation F1 is 0.4702: effects are +0.0067 versus fixed top-one, -0.0108 versus fixed top-two, and +0.0133 versus fixed top-three; the required all-budget superiority condition therefore fails;
- DACS-SCV improves count MAE/exact match to 0.4954/0.5565, but fixed top-two obtains the best citation F1 (0.4810), establishing a count-accuracy/set-utility mismatch under the frozen ranking;
- generated manuscript tables and the formal analysis figure derive directly from the frozen archive; the Results, Discussion, Limitations, Conclusion, title, and abstract have been synchronized to the negative finding;
- `dacs-scv-paper-source-2026-10-28.zip` (SHA-256 `9AE061A4EB93E75A3C9426704F94DC4A0FE40AE1AF33799D449C205C5F316ED6`) contains only the manuscript source, direct figures/tables, bibliography, README, and compiled PDF; a clean extraction rebuilds to ten pages without LaTeX warnings or overfull boxes;
- the DACS-SCV validation pilot is preserved as `dacs-scv-local-pilot-outputs.zip` (17,549 bytes; SHA-256 `c17c12299235f361fa96e55c56cb049bc3ba15a03f1d6a66ca07bd4d51f32713`);
- the decoupled head improves validation citation macro-F1 by 4.52 points with identical verdict vectors; its 10,000-sample grouped-bootstrap interval for the citation difference is [-0.0089, 0.0991], so the official-test confirmation remains essential;
- a frozen-ranking cutoff audit finds fixed top one at 0.5236 and fixed top two at 0.5047; DACS-SCV's validation advantage over the best fixed budget is only 0.0067 with interval [-0.0077, 0.0219], so formal adaptive superiority requires beating the best fixed cutoff, not merely top three;
- the validation gain is positive in text-only and visually grounded strata but negative in the small three- and four-gold-item strata; these are recorded diagnostics, not final subgroup claims;
- the corrected v9 local pilot is preserved as `trace-scv-v9-local-pilot-outputs.zip` (13,309,503 bytes; SHA-256 `7ff604d7cde6615889aeb6343302a2c836f24abbd6629200674a3e7d7fc66449`);
- all 36 declared pilot artifacts match their byte counts and hashes, all six checkpoint fingerprints independently recompute, and all primary per-variant metrics reproduce exactly;
- the causal readiness gate fails four checks on corrected data, so the counterfactual causal-grounding claim and formal causal ablation matrix are retired rather than retuned;
- local pilot execution is feasible on the Ryzen 7 7700 CPU with 15.19 GB RAM: corrected feature extraction took about nine minutes and all six training runs together took under thirty seconds; no CUDA device is available, so the full five-seed study remains allocated to Kaggle;
- pilot execution policy changed at the user's request: iterative validation-only pilots run locally, while Kaggle is reserved for the frozen final matrix; the current host has 16 GB RAM, CPU-only PyTorch 2.14.1, and no CUDA device, so local encoder extraction is expected to be slower but remains methodologically equivalent;
- the returned v8 pilot archive (13,292,553 bytes; SHA-256 `a733f4727bfafab3761de85bd00e3047b8b51eacdae293827f1d5ced36bc8f3d`) is mechanically reproducible, but its data-release gate fails because six gold-content fingerprints cross train and validation;
- item-level audit traces those overlaps to the v8 section resolver returning generic parent headings before exact annotated subsections, so all v8 scientific metrics are invalidated and excluded from the manuscript;
- v9 uses exact-first subsection matching, never falls upward from a requested child, flags missing-annotation abstract fallbacks, and reports fallback counts;
- the corrected identical 600/150 local dataset reconstruction has zero fallback gold items, complete required visual media (165/165 train and 44/44 validation), zero group/content/exact-visual/near-perceptual cross-split overlaps, and a passing release gate;
- all 29 unit/integration tests pass after the v9 parser and audit corrections;
- `trace-scv-kaggle-v9.zip` (75,422 bytes; SHA-256 `7595a825ffdf4bc8174e57fd6f3276b31687f1a05dfc0e5fa9b8709fc5c769e6`) contains 40 Python files, requires package version 0.2.1, contains the exact-section fix, has no cache/build artifacts, and had zero workspace-file mismatches at packaging time;
- the v9 notebook completed all cells from a randomly renamed ZIP nested below a simulated Kaggle input tree; its schema-4 smoke archive contained a complete 37-file inventory with zero byte-count or SHA-256 mismatches and remained explicitly non-scientific;
- v8 stores BGE semantic features for every evidence item and separately stores CLIP pixel features plus an availability mask; a learned gate fuses the independently projected channels;
- v8 pixel removal and same-modality swapping preserve semantic features, identifiers, node types, validity masks, and graph nodes; the intervention behavior is covered by dedicated unit tests;
- evaluation exports all nine clean/intervention probability vectors, swap-applied and visual-grounding flags for every claim, visual-grounded subset metrics, and artifact schema version 4;
- the SciVer manifest now exports item-level exact/near duplicate records and a release gate over group overlap, exact visual overlap, and exact gold-content overlap;
- all 27 unit/integration tests pass after the v8 representation, intervention, evaluation, and data-audit changes, including a distinct-channel feature-store test for media-backed evidence;
- `trace-scv-kaggle-v8.zip` (74,067 bytes; SHA-256 `1c7b7f502166f8c3b94fa7888758abce952fbf893fd7576b01d98035d5e2a104`) contains 40 Python source/test/script files with zero workspace hash mismatches and no bytecode/cache files;
- the v8 notebook executes all 16 cells from a deeply nested, randomly renamed ZIP in a clean Kaggle-like working directory, verifies package version 0.2.0, produces a schema-4 smoke archive with a complete 37-file inventory and zero hash mismatches, and exports all required per-claim intervention fields;
- the notebook comparison plot was visually inspected and is legible without clipping or overlapping labels;
- the final controlled-negative-study IEEE manuscript compiles successfully with both the built-in editor and local `latexmk`; its ten-page PDF has no LaTeX warnings or overfull boxes, and its architecture, formal-results figure, tables, and all pages pass visual inspection;
- SciVer is public, ungated, licensed, reachable, and pinned; MuSciClaims is reachable but disabled because the main release does not provide authoritative license metadata;
- MOCHEG and the MEVER/AIChartClaim repositories are publicly reachable but remain optional pending data-term audits;
- all 25 unit/integration tests passed after the earlier v6 intervention and data-audit corrections;
- the deterministic synthetic end-to-end smoke check passes;
- the one-notebook Kaggle workflow executes all 16 cells without an error and its comparison plot was visually inspected;
- the v4 Kaggle package preserves the nested `experiments/src/trace_scv/data` source package when Kaggle exposes the upload as a directory, rejects incomplete uploads, clears stale modules, and pins imports to the uploaded source tree;
- the uploaded v4 pilot archive passes mechanical prediction/checkpoint integrity checks but fails the superiority gate on its non-scientific hash-feature slice; the complete audit is recorded in `research/pilot_audit_2026-10-26.md`;
- the v5 pipeline exports gold/prediction comparison records, dataset/configuration/code/feature hashes, preflight evidence, and a complete artifact inventory; stale checkpoints are rejected by fingerprint;
- the v5 pilot uses formal BGE/CLIP features and real media on development train/validation data only, with deterministic bounded sampling;
- all 21 automated tests pass after the v5 changes, and the notebook executes end to end from a clean extraction of `trace-scv-kaggle-v5.zip`; its smoke export contains every required audit artifact;
- the returned v5 encoder-valid pilot archive (SHA-256 `8bf2843ffc2cf952afabfed642d718561e9ea1f718bf08317e482af84769bc19`) passes every integrity and recomputation check; its matched F1 feasibility gate passes, but its causal deletion and visual-swap behavior do not, so the formal run remains blocked;
- v6 deletes the exact selected citation set used by the clean prediction, trains/evaluates visual swaps only against same-modality replacements, records swap coverage, and emits split-level modality, missing-media, exact-duplicate, perceptual-duplicate, and cross-split overlap audits;
- the replacement upload `trace-scv-kaggle-v6.zip` (68,163 bytes; SHA-256 `df9d04f8ac834d11b4044c140a9cddf1987a89c5920a6d0fc9b245324ccbdfca`) contains the complete source tree and was validated by executing its notebook from a clean extraction in smoke mode; the exported 37-file inventory had zero hash or byte-count mismatches;
- after Kaggle failed to expose v6 at the assumed filename/layout, v7 changed discovery to search every nested extracted repository and every ZIP independent of its filename, added safe ZIP-member validation and actionable visible-file diagnostics, and was executed successfully in simulations of both a renamed deeply nested ZIP and a deeply nested extracted dataset tree;
- `trace-scv-kaggle-v7.zip` (68,888 bytes; SHA-256 `2f5b7a0048a45ae724289378a5f84a9fb1acdfc04e14e1a383a7a33026e4e6ee`) is the only active Kaggle upload; its bundled notebook, imports, smoke study, output export, and comparison plot were verified from both simulated Kaggle layouts;
- the returned v7 pilot archive (10,916,361 bytes; SHA-256 `c07364083c062cc5c4e9f738af9a1eab7ffad6f16c23c9c7c8e14d04450e13ca`) has a complete 37-file inventory with zero mismatches; all primary metrics and grouped-bootstrap comparisons recompute exactly, but the readiness gate fails four of seven checks and the formal run remains blocked;
- the v7 data audit finds zero group or visual-media overlap but six exact content fingerprints shared by train and validation; the archive lacks item-level duplicate details, so their gold/non-gold role remains unresolved;
- formal feature extraction and grouped-bootstrap evaluation are batched/vectorized, and completed checkpoints are reused on rerun;
- a pinned real-SciVer miniature run downloaded 8 train, 4 validation, and 4 official-test examples and completed normalization, training, checkpoint reload, and evaluation;
Where a check is explicitly identified as a pilot diagnostic, it is not a final scientific result and does not support a manuscript performance claim. The main numerical claims derive only from the independently audited frozen formal archive.
