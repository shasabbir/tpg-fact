# Publication Plan

Last updated: 2026-11-09  
Active manuscript: `paper/paper.tex`  
Pre-publication checkpoint: `321af11`  
Status: **not submission-ready; controlled study and manuscript foundation are strong**

## 1. Decision summary

The current work is publishable in principle, but it is not yet ready for a
strong-journal submission. Its strongest contribution is a reproducible
diagnosis of how retrieval reachability, probability calibration, evidence-set
decoding, and verdict preservation interact. It does **not** establish that
expected-F1 is the best selector or that the full system is better end to end.

The recommended route is to strengthen the paper before submission:

1. freeze a new confirmatory protocol without opening formal-test labels;
2. improve retrieval at the operating packet depths;
3. test one justified retrieval-completeness-aware selector against the actual
   strongest baseline, calibrated thresholding;
4. rerun the shared-verifier safeguard with a stronger verifier and a frozen
   precision target;
5. run one untouched external or formal-test evaluation;
6. complete the reference-support ledger and author declarations;
7. revise only the canonical manuscript in `paper/`, then create and verify the
   final submission package.

The project will not create or use `paper_q1/`. The existing paper depth will
be preserved. Venue length limits will be met by separating main and
supplementary material, not by deleting important comparisons, negative
results, or literature context.

## 2. Frozen starting point

The plan begins from the following checked state. Any changed implementation or
protocol invalidates only the affected results and must trigger the specified
rerun.

| Item | Current checked state | Publication implication |
|---|---|---|
| Evaluation population | 1,490 locked MOCHEG validation claims over five deterministic training-role partitions | Controlled development evidence, not independent confirmation |
| Retrieval population | 10,970 eligible complete-gold claims | Adequate for retrieval diagnosis |
| Candidate packet | 40 text + 20 image items for every selector | Fair matched comparison |
| Packet oracle evidence F1 | 0.3560 | Retrieval imposes a low ceiling |
| Complete-gold packet coverage | 4.39% | Complete evidence is rarely reachable |
| Zero-reachable claims | 3,086 | Many errors cannot be repaired by selection |
| Best tested selector | Calibrated threshold, F1 0.1807 | Must be the primary selector comparator |
| Expected-F1 | F1 0.1759; 3.50 items; +0.00851 over fixed top-5 | Useful compact operating point, not the best method |
| Expected-F1 interval | 95% CI [0.00302, 0.01400] vs fixed top-5 | Small but reproducible development effect |
| Verifier safeguard | +0.00099 macro-F1; 95% CI [-0.01053, 0.01279] | Declared non-inferiority is not established |
| Robustness | Probability perturbations reverse the expected-F1 gain | Calibration drift is a material failure mode |
| Reproduction | 126 tests; 20/20 core artifacts reproduced exactly | Strong implementation foundation |
| Manuscript | 14 pages; 55 cited references; full-page visual QA passed | Strong presentation foundation |
| Bibliography audit | 52 claim-support reviews pending; 3 strict-policy conflicts | Blocks final reference readiness |
| Formal test | Annotations unopened | Remains available only after protocol freeze and author authorization |

Authoritative state files:

- `research/calibration_corrected_study/manuscript_readiness.md`
- `research/calibration_corrected_study/remaining_actions.md`
- `research/calibration_corrected_study/claim-evidence-register.csv`
- `research/calibration_corrected_study/paper_release.json`
- `research/calibration_corrected_study/reference_audit/audit.md`

## 3. Publication story and claim boundary

### 3.1 Recommended central claim

Adaptive multimodal evidence selection must be evaluated as a
retrieval-conditioned and calibration-dependent decision. On the current
development population, prior-aware calibration is essential, calibrated
thresholding is the strongest overall policy, and expected-F1 is a smaller,
more precise operating point whose benefit is concentrated where the complete
gold set is already reachable.

This claim is valuable because it separates four failures that are often
collapsed into one score:

1. missing evidence at retrieval;
2. incorrect within-packet ranking or membership scoring;
3. probability miscalibration;
4. a poor set-decoding or verdict decision.

### 3.2 Stronger claim available only after new evidence

A new method-superiority claim may be added only if a prespecified
retrieval-completeness-aware policy beats calibrated thresholding under a
matched protocol, preserves verdict quality, and replicates on untouched data.
If it does not, the paper remains a diagnostic and benchmarking contribution.
Negative or mixed results stay in the paper.

### 3.3 Claims that remain prohibited

- state-of-the-art performance without a same-protocol comparison;
- end-to-end superiority from evidence F1 alone;
- broad generalization from MOCHEG validation only;
- robustness when rank-preserving calibration perturbations reverse the gain;
- efficiency without measured hardware, latency, memory, and energy boundaries;
- a Q1-ready or acceptance-ready claim based only on layout and reference count.

## 4. Venue strategy

No quartile label will be asserted until the authors choose a ranking database,
subject category, and ranking year and record the lookup date. Scope fit and
submission rules are separate from quartile status.

| Route | Fit and requirement | Use this route when |
|---|---|---|
| **Stretch: IEEE Transactions on Multimedia** | Direct multimedia scope and strong reproducibility fit. Current official author information limits an initial regular paper to 10 double-column pages and TMM supplementary material to four double-column pages. | External confirmation, improved retrieval, and a convincing method or diagnostic contribution are all complete. Build a 10-page main paper plus a four-page supplement without deleting scientific depth. |
| **Primary pragmatic target: Information Processing & Management** | Strong fit for information retrieval, information science, social computing, and research-method contributions. | The external evaluation confirms the retrieval-conditioned diagnostic story, even if no new selector is uniformly superior. Convert from IEEE layout only after the venue is selected. |
| **Fallback: Expert Systems with Applications** | Fits applied intelligent systems, information retrieval, and multimedia mining, but requires genuine technical contribution rather than renamed standard components. | The final contribution is an application-centered system and evaluation with strong external evidence. |

Official scope/rule checks used for this plan:

- [IEEE Transactions on Multimedia scope](https://signalprocessingsociety.org/publications-resources/ieee-transactions-multimedia)
- [IEEE Signal Processing Society information for authors](https://signalprocessingsociety.org/publications-resources/information-authors)
- [Information Processing & Management scope](https://shop.elsevier.com/journals/information-processing-and-management/0306-4573)
- [Expert Systems with Applications scope](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)

The venue is selected after the external-evidence gate, because the result will
determine whether the contribution is primarily multimedia methodology,
information-retrieval evaluation, or an applied intelligent system.

## 5. Publication gates

Each gate produces an immutable report and a Git commit. A failed gate changes
the paper's claim; it does not trigger outcome-driven tuning.

| Gate | Required evidence | Pass condition |
|---|---|---|
| G0: baseline preservation | Clean tree, checkpoint, hashes, current release report | **Passed** at `321af11` |
| G1: protocol and data freeze | Dataset roles, candidate widths, models, trial budget, seeds, exclusions, metrics, statistical tests, hardware plan | Independent audit finds no label access or outcome-dependent choice |
| G2: retrieval | Complete-gold coverage, packet oracle F1, Recall@40 text, Recall@20 image, MAP/nDCG, runtime and memory | Selected variant improves the frozen packet oracle and operating-depth recall without degrading the other modality; all selection occurs on training roles |
| G3: selector | Fixed budgets, calibrated threshold, learned count, expected-F1, and one completeness-aware method on identical packets and probabilities | Superiority language requires at least +0.010 evidence F1 over calibrated threshold, a positive paired 95% interval after multiplicity control, positive effects in all five partitions, and no larger mean set than thresholding |
| G4: verifier | Stronger shared verifier, frozen non-inferiority margin and claim-aligned interval | Lower 95% confidence limit above -0.01 |
| G5: external confirmation | One untouched formal-test or compatible external evaluation run | Same protocol, no retuning, complete artifact audit; conclusions reported whether positive or negative |
| G6: reference and manuscript integrity | Per-claim source support, eligible bibliography, code/method/results consistency, venue format | No unresolved blocking finding |
| G7: release and submission package | Clean source build, PDF QA, supplement, code/data statement, declarations, cover letter | Extracted package builds and every headline claim traces to a frozen artifact |

The numerical G3 threshold is a proposed practical-effect requirement and must
be copied into the frozen protocol before any new outcome is inspected. If the
authors prefer a different minimum effect, it must be changed before G1 closes.

## 6. Confirmatory experimental protocol

### 6.1 Data roles and leakage controls

- Preserve claim-level independence across fitting, calibration, model
  selection, internal holdout, and final evaluation roles.
- Fit vocabulary, retrievers, hard-negative mining, scorer parameters,
  calibration maps, thresholds, count models, and completeness estimators only
  on allowed training roles.
- Keep the official MOCHEG test annotations inaccessible until G1 is signed and
  the authors explicitly authorize the one-shot evaluation.
- Run duplicate, near-duplicate, source, and event-overlap checks before
  interpreting external performance. Where source/event metadata are absent,
  narrow the claim instead of declaring confirmed independence.
- Preserve the existing 40-text/20-image packet as the primary matched setting.
  Wider packets and claim-adaptive budgets are secondary analyses.

### 6.2 Retrieval workstream

The current oracle ceiling is the main scientific bottleneck. Limit the search
to a small, declared matrix:

1. frozen BGE-M3 text and SigLIP2 image baseline;
2. BM25 text retrieval;
3. BGE-M3 + BM25 reciprocal-rank fusion;
4. fusion plus a frozen text cross-encoder reranker;
5. SigLIP2 plus OCR/caption/article-context image fusion;
6. one combined hybrid configuration selected on training roles.

For every configuration retain label-blind rankings before gold joins, model
revision, preprocessing, index hash, missing-media log, Recall@5/10/20/40/50,
MAP, nDCG, complete-gold coverage, packet-oracle F1, wall time, peak RAM/VRAM,
and index size. Energy is reported only if it is measured with a declared
boundary.

Primary retrieval selection uses packet-oracle F1 at 40 text + 20 image, with
operating-depth modality recall as a guardrail. Aspirational engineering
targets are an absolute oracle-F1 improvement of at least 0.05 and
complete-gold coverage of at least 8%; these are targets, not fabricated
outcomes or guaranteed gates.

### 6.3 Scoring and calibration workstream

- Retain the low-capacity logistic scorer as a control.
- Compare one content-aware text scorer and one content-aware image scorer with
  the same hard-negative and training access.
- Compare identity, temperature, Platt, prior shift, prior-aware Platt, and
  prior-aware isotonic maps under the frozen calibration role.
- Select calibration using Brier score first; report NLL, ECE, reliability
  curves, prevalence, and selected-set-size effects.
- Treat naturally occurring calibration drift and rank-preserving temperature
  changes as required stress tests.

### 6.4 Selector workstream

Required matched policies:

1. fixed top-1/3/5/10 controls;
2. calibrated global or per-modality threshold;
3. learned count;
4. exact expected-F1;
5. uncalibrated expected-F1 ablation;
6. one retrieval-completeness-aware policy.

The completeness-aware policy is justified by the present post hoc finding and
must be specified before new outcomes. A minimal design is a training-only
estimate of packet completeness that chooses between a compact expected-F1 set
and the higher-recall calibrated threshold. Its inputs, calibration, threshold,
and fallback behavior must be frozen. Do not add a large new architecture
unless this controlled design first shows value.

The primary selector outcome is claim-level complete-gold evidence F1.
Secondary outcomes are precision, recall, Jaccard, exact match, selected count,
count error, Brier/NLL/ECE, and review burden. Calibrated thresholding—not fixed
top-5—is the comparator for any new superiority claim.

### 6.5 Verifier workstream

- Freeze one stronger shared verifier before selector-set evaluation.
- Give every selector the same verifier, training data, prompt/model revision or
  checkpoint, decoding settings, and tuning access.
- Report macro-F1, class-wise F1, calibration if applicable, failure counts,
  latency, and the paired selector-minus-baseline interval.
- Use the existing -0.01 non-inferiority margin. A point estimate above zero is
  insufficient when the lower confidence limit crosses the margin.

### 6.6 External or formal-test workstream

Preferred order:

1. use the sealed MOCHEG test once, only after G1-G4 and explicit author
   authorization;
2. in parallel, re-audit an independent benchmark only if it supplies a
   research-compatible license, stable evidence identifiers, text and image
   evidence, an accessible corpus or defensible candidate construction, and an
   untouched evaluation split;
3. adapt the protocol only before seeing external outcomes and document every
   unavoidable change in evidence units or labels.

AVerImaTeC and MMCV are adaptation candidates, not interchangeable replicas of
MOCHEG. If no independent benchmark passes the access/compatibility audit, do
not force a misleading cross-dataset comparison. The formal test then becomes
the only confirmatory route; without it, the paper stays a bounded
single-benchmark development study.

### 6.7 Statistical analysis

- Freeze one primary comparison and a finite family of secondary comparisons.
- Estimate the sample size or attainable interval width from current
  claim-level paired differences before final evaluation. Target at most a
  +/-0.005 half-width for the main evidence-F1 effect when the available test
  population permits it.
- Use claim-aligned paired bootstrap intervals with 10,000 resamples and Holm
  correction for the declared family.
- Report effect sizes and intervals, not only p-values.
- Prespecify reachability strata: no gold reachable, partially reachable, and
  complete gold reachable. Treat them as secondary unless powered and declared
  otherwise.
- Report all five partition effects and all failed configurations. Do not
  select the best seed.
- Preserve raw per-claim probabilities, selections, verifier predictions, and
  bootstrap inputs so every table can be independently recomputed.

## 7. Manuscript revision plan

All manuscript work remains in `paper/`. Experimental changes are written only
after their outputs pass the relevant audit.

### 7.1 Main-paper narrative

1. **Abstract:** task, retrieval-conditioned gap, protocol, one main result,
   one failure, and bounded implication. Update last, after results freeze.
2. **Introduction:** retain one contribution list; make the diagnostic novelty
   and strongest baseline explicit.
3. **Related work:** preserve the technical taxonomy and comparison depth, but
   tie every comparison row to verified sources.
4. **Problem and data:** make evidence units, complete-gold assumptions,
   unreachable evidence, and independent data roles unambiguous.
5. **Method:** describe the actual frozen retrieval, scoring, calibration,
   selection, and verifier paths. Clearly label reused, adapted, and new parts.
6. **Experimental setup:** add external-data audit, hardware/runtime settings,
   trial budget, seeds, selection rule, and statistical family.
7. **Results:** lead with retrieval reachability, then calibration, matched
   selectors, verifier, external confirmation, and robustness.
8. **Discussion:** explain why thresholding is strong, when compact selection is
   useful, and how incomplete retrieval changes the decision problem.
9. **Limitations and ethics:** retain unfavorable results, external-validity
   boundary, dataset/licensing facts, and misuse/automation limits.
10. **Conclusion:** state only the frozen result and its operational meaning.

Do not use internal labels such as “phase 1,” “phase 2,” or “phase 3” in the
paper. Use scientific section names and experiment descriptions.

### 7.2 Result tables and figures

Main paper should contain:

- related-work comparison matrix;
- dataset, split, and information-boundary table;
- retrieval matrix with operating-depth recall and packet oracle;
- calibration table and reliability figure;
- selector main-effects table with intervals and selected count;
- reachability-stratified table;
- verifier non-inferiority table;
- external/formal-test table;
- end-to-end architecture/information-boundary figure;
- robustness summary figure.

Partition-level results, the full robustness matrix, detailed audit tables,
hyperparameter grids, and reference-support material belong in the supplement
when the venue has a tight main-paper limit.

### 7.3 IEEE TMM length strategy

The present 14-page manuscript cannot be submitted unchanged under the current
10-page TMM initial limit. If TMM is selected:

- keep motivation, method, main comparisons, negative result, and external
  evidence in the 10-page main paper;
- move the reporting appendix, full retrieval matrix, partition results,
  absolute robustness table, extended implementation settings, and additional
  source-support details to a maximum four-page double-column supplement;
- retain the 55-reference-quality literature coverage as eligible and relevant;
  do not reduce references merely to create visual space;
- rebuild and visually inspect both artifacts at final submission size.

## 8. Reference and integrity plan

The number of references is not a readiness metric. Each cited claim must have
an eligible source and a recorded support location.

1. Review all 55 cited records against the claim-evidence ledger.
2. Record DOI identity, cited version, publication year, peer-review state,
   source passage/page, manuscript claim, and correction/retraction check.
3. Replace or remove the three current strict-policy conflicts:
   - 2009 reciprocal-rank-fusion citation without an eligible DOI/year;
   - 2017 calibration citation outside the study's reference policy;
   - 2025 MMCV record without a verified DOI.
4. Prefer 2022 onward peer-reviewed sources; retain 2020-2021 only when they are
   the best eligible support. Do not cite ordinary arXiv versions when a
   peer-reviewed version exists.
5. Rerun the automated bibliography audit after the human support review.
6. Confirm that every visible reference has a correct DOI link and that every
   bibliography entry is cited for a claim it actually supports.

## 9. Reproducibility and release package

Every outcome-bearing run records:

- Git commit and resolved configuration;
- Python, framework, library, CUDA/driver, and OS versions;
- CPU, GPU, RAM/VRAM, numerical precision, batch size, and seeds;
- dataset, corpus, media, model, feature, and split hashes;
- label-blind candidate rankings and gold-joined audit records;
- checkpoints, calibration maps, per-claim probabilities and selections;
- verifier predictions, raw metrics, bootstrap inputs, and corrected tests;
- wall time, peak memory, index/storage size, failures, and deviations.

Final deliverables:

1. editable LaTeX source in `paper/`;
2. venue-formatted main PDF and supplement;
3. figure and table sources linked to immutable result files;
4. environment lock and one-command reproduction instructions;
5. public-data acquisition script and license notes, without redistributing
   restricted data;
6. claim-evidence and reference-support registers;
7. clean source archive that builds outside the working tree;
8. anonymized package if required by the venue;
9. code/data availability statement and archival plan;
10. cover letter, highlights or graphical abstract only when required.

No repository, archive, preprint, or journal submission will be published
without explicit author authorization.

## 10. Author facts required

The authors must supply or approve these facts before G7:

- final author order, corresponding author, and affiliations;
- CRediT contribution statement;
- funding and grant identifiers, or an explicit no-funding statement;
- competing-interest statement;
- data and code availability statement;
- permission/license constraints for released data and media;
- target venue and chosen ranking database/category/year if a quartile claim is
  desired;
- AI-assisted writing/code disclosure required by the selected venue;
- authorization before any sealed formal-test access or public release.

These facts will not be inferred from code or invented for the manuscript.

## 11. Work schedule

The schedule is dependency-based. A typical execution on one RTX 4090-class
workstation is approximately ten focused work blocks; data access or model
runtime may extend it.

| Block | Work | Exit artifact |
|---|---|---|
| 0 | Preserve current state | Commit `321af11` (**done**) |
| 1 | Venue shortlist, protocol, formal-test lock, author decisions | Signed protocol and decision record |
| 2 | External-data/license and overlap audit; baseline runtime profile | Data audit and hardware manifest |
| 3 | Text retrieval matrix | Frozen rankings and retrieval report |
| 4 | Image and hybrid retrieval matrix | G2 decision |
| 5 | Scoring, calibration, and completeness estimator | Calibration report and frozen selector inputs |
| 6 | Matched selector and stronger-verifier evaluation | G3/G4 decision and raw claim records |
| 7 | One-shot external/formal-test evaluation | G5 report with immutable hashes |
| 8 | Statistics, error analysis, figures, and tables | Audited result bundle |
| 9 | Reference ledger and full manuscript revision | Venue-formatted release candidate |
| 10 | Clean reproduction, visual QA, declarations, cover letter | Submission package and readiness report |

## 12. Decision rules

### Outcome A: strongest route

External evidence replicates the diagnostic findings, retrieval improves, the
new selector beats calibrated thresholding under G3, and verifier
non-inferiority passes. Submit the method-plus-evaluation paper to the stretch
venue after current ranking and submission-rule verification.

### Outcome B: strong diagnostic route

External evidence confirms that reachability and calibration change selector
conclusions, but thresholding remains best or verifier non-inferiority remains
unresolved. Submit an honest retrieval-conditioned evaluation paper to the
best-fitting information-retrieval or multimedia journal. Remove method
superiority language.

### Outcome C: bounded single-benchmark route

No external/formal evaluation is available. Retain the current controlled
development framing, complete the reference and declaration work, and choose a
less ambitious venue. Do not claim broad generalization or Q1 readiness.

### Outcome D: contradictory external result

Report the contradiction, investigate only predeclared data-quality and
implementation failures, and do not retune on the external labels. The paper
may still be valuable as a cross-dataset failure analysis if the protocol and
artifacts remain defensible.

## 13. Definition of publication-ready

The manuscript is publication-ready only when all applicable boxes are true:

- [x] Current study and manuscript preserved in Git.
- [x] Current implementation tests and clean core reproduction pass.
- [x] Current PDF compiles and all pages have been visually inspected.
- [ ] Target venue and current official rules verified.
- [ ] Confirmatory protocol frozen before external labels are accessed.
- [ ] Retrieval workstream completed and fully audited.
- [ ] Strongest calibrated selector retained as comparator.
- [ ] Verifier safeguard resolved or the claim narrowed.
- [ ] Untouched external or formal-test evaluation completed, or the paper is
      explicitly routed as a bounded single-benchmark study.
- [ ] Every headline number traces to raw per-claim evidence.
- [ ] All 55 reference claims reviewed; ineligible records replaced or removed.
- [ ] Funding, conflicts, contributions, availability, and AI-use statements
      approved by the authors.
- [ ] Venue-formatted main paper and supplement pass build and visual QA.
- [ ] Clean extracted package builds without machine-local dependencies.
- [ ] Independent internal review finds no blocking scientific, reference, or
      packaging issue.

## 14. Immediate execution order

1. Commit this publication plan separately from the preserved baseline.
2. Create a concise machine-readable confirmatory protocol and test-lock file.
3. Re-audit external/formal-test access and dataset compatibility.
4. Capture the actual hardware/environment/runtime baseline.
5. Execute the bounded retrieval matrix before changing the selector.
6. Freeze the completeness-aware selector and stronger verifier only after G2.
7. Run G3 and G4, then decide whether formal-test access should be requested.
8. Complete the reference ledger in parallel because it does not depend on
   experimental outcomes.
9. Revise `paper/paper.tex` only after the result bundle is frozen.
10. Produce the venue-specific main paper, supplement, clean archive, and final
    readiness audit.

No step authorizes opening sealed labels, publishing files, or submitting to a
journal without explicit author approval.
