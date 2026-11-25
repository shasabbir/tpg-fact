# One-Shot Q1-Primary / Q2-Fallback Publication Plan

Last updated: 2026-11-25  
Canonical manuscript: `paper/paper.tex`  
Study status: MOCHEG study complete; five-seed, train-only AVerImaTeC study complete; locked 152-claim development evaluation executed once; formal tests remain sealed.

## 1. Clear goal

Produce one final, evidence-dense journal manuscript that is credible for a
well-matched Q1 attempt and is immediately suitable for a strong Q2 fallback.
The paper will make a measurement and evaluation contribution, not claim a new
state-of-the-art selector:

> Multimodal evidence-selection results are conditional on evidence
> reachability and probability calibration. Controlled MOCHEG experiments and
> a frozen external AVerImaTeC nonreplication show which apparent selector gains
> survive changes in retrieval, evidence units, and class balance.

Q1 is a target, not a promised outcome. The study will not be enlarged with
unrelated models merely to appear more complex.

## 2. What the final paper contributes

The final contribution list is limited to four defensible items:

1. a retrieval-conditioned framework that separates corpus availability,
   candidate reachability, evidence scoring, set selection, and verification;
2. a controlled multimodal study of fixed, threshold, expected-F1, and
   completeness-aware evidence selection under matched retrieval and
   calibration conditions;
3. a preregistered-style, train-only external confirmation whose negative and
   inconclusive results are retained rather than tuned away; and
4. an auditable artifact trail linking claims, configurations, predictions,
   uncertainty estimates, tables, and figures.

The novelty is primarily empirical and methodological: the paper establishes
when selector comparisons are valid and why gains fail to transfer. It will
not present standard BM25/BGE/SigLIP components as individually novel.

## 3. Fixed scope and stop rules

To keep the revision coherent and finish it in one shot:

- no third benchmark;
- no human audit, because it is unavailable;
- no new selector family or post hoc tuning on the locked AVerImaTeC
  development labels;
- no major verifier replacement; the verifier remains a downstream diagnostic;
- no reopening the failed MOCHEG superiority gate and no formal-test access;
- no blind-test prediction or external submission without explicit author
  authorization;
- no Gemini/Gemma API result will be called an official score unless it uses
  the organizer's exact checkpoint, prompt, decoding, and evaluator;
- no citation padding, speculative qualitative examples, or synthetic result;
  and
- every added table or figure must answer a stated research question and be
  generated from preserved artifacts.

If a required resource is unavailable, the paper will disclose the limitation
and follow the Q2 route rather than substitute an incomparable experiment.

## 4. Remaining work, in execution order

| Order | Work package | Exact output | Completion rule |
|---|---|---|---|
| 1 | Freeze and reconcile | checkpoint commit, result manifest, claim--evidence map | Every current headline number resolves to a frozen artifact and code revision |
| 2 | Official semantic evaluation | organizer-compatible semantic scores, raw judge outputs, hashes, run metadata | Exact pinned evaluator runs successfully on suitable hardware; otherwise unavailability is documented and the Q2 route is selected |
| 3 | External calibration analysis | Brier, NLL, ECE, prevalence-Brier baseline, average precision, and reliability data for raw versus calibrated scores | Derived once from frozen predictions with no refitting; denominators and modality aggregation are explicit |
| 4 | Automated failure decomposition | stage-wise counts and one failure-flow figure | Every exact-source target is assigned to the earliest observable failure stage using deterministic rules |
| 5 | Cross-benchmark synthesis | one compact transfer table and policy-disagreement table | MOCHEG and AVerImaTeC use clearly separated endpoints; no false direct score equivalence |
| 6 | Manuscript revision | updated `paper/paper.tex`, figures, tables, supplement, and PDF | Abstract, contributions, methods, results, discussion, limitations, and conclusion agree with the frozen evidence |
| 7 | Reproducibility and venue package | clean source archive, run instructions, environment/config manifest, availability statement, venue checklist | Fresh extraction builds; commands regenerate main derived results; current official venue requirements are verified |

### 4.1 Official semantic evaluation

This is the only remaining experiment that materially strengthens the external
validity claim. Run the organizer's pinned Gemma-3-27B evaluator with its exact
prompt, decoding, and aggregation on hardware that can execute the checkpoint
faithfully. Record model revision, file hashes, precision, hardware, software,
prompt, decoding, raw responses, failures, and cost/runtime.

The inputs are the already-frozen five policy outputs from the locked
152-claim development run. There is no retraining and no selector change after
seeing semantic scores. If faithful hardware is unavailable, do not replace the
endpoint with a free-tier hosted model and call it equivalent. A hosted model
may be reported only as a clearly labelled supplementary sensitivity analysis,
and it is not required for completion.

### 4.2 External calibration analysis

Use the saved candidate probabilities and exact-source labels. Report raw and
calibrated results separately for text, image, and the declared combined
aggregation:

- Brier score and prevalence-only Brier baseline;
- negative log-likelihood with a recorded numerical clipping rule;
- ECE with a fixed, declared binning rule;
- average precision because positives are sparse; and
- reliability-bin counts, mean confidence, and empirical positive rate.

Use claim-grouped uncertainty where an interval is reported. These are
exact-source calibration diagnostics, not semantic-relevance calibration.
They test the paper's calibration-transfer claim without creating a new model.

### 4.3 Automated failure decomposition

Assign each gold source to the earliest stage at which it becomes unavailable:

1. absent from the supplied knowledge store;
2. present in the store but absent from the discovered retrieval pool;
3. in the broad pool but below the retained packet depth;
4. in the retained packet but omitted by the evidence selector; or
5. selected, but associated with an incorrect final verdict.

Report both target-level counts and claim-level complete-reachability rates.
Keep textual and visual evidence separate before presenting any aggregate. This
analysis uses deterministic identities and existing outputs; it does not
pretend to replace a human semantic audit.

### 4.4 Cross-benchmark synthesis

Create one table with matched questions rather than pretending the datasets
have identical metrics:

- candidate reachability and complete reachability;
- calibration change relative to the prevalence baseline;
- fixed versus threshold versus expected-F1 evidence-set behavior;
- selected-set size;
- paired uncertainty and which preregistered gates passed; and
- verifier preservation or degradation.

Add one concise policy-disagreement analysis showing how often policies return
different sets and whether those disagreements occur mainly under low
reachability. This strengthens the mechanism story without adding another
architecture.

## 5. Manuscript revision plan

The paper will be revised around one argument: retrieval reachability defines
the achievable ceiling, calibration determines whether adaptive policies can
use their scores, and neither guarantees downstream verifier improvement.

Required changes:

1. **Title and abstract:** state the conditional evaluation contribution and
   external nonreplication; avoid method-superiority language.
2. **Introduction:** use one contribution list; explain why negative transfer
   is scientifically informative.
3. **Related work:** perform a narrow 2024--2026 refresh covering multimodal
   fact-checking retrieval, calibration, set prediction, and semantic evidence
   evaluation. Add only peer-reviewed, DOI-verified sources that support a
   specific claim.
4. **Methods:** present the stage decomposition and frozen external protocol;
   distinguish exact-source and semantic endpoints.
5. **Results:** order evidence as reachability, calibration, matched selection,
   verifier, external transfer, then failure decomposition.
6. **Discussion:** explain why calibration transferred while selector and
   verifier gains did not; label causal explanations as hypotheses where the
   experiment does not identify causality.
7. **Limitations:** retain the missing official semantic endpoint if unresolved,
   exact-identity strictness, external class imbalance, low reachability, weak
   verifier performance, and absence of human semantic adjudication.
8. **Presentation:** repair awkward humanized wording, remove repeated result
   narration, keep tables readable, and preserve the current polished visual
   standard rather than increasing length for its own sake.

The paper should become denser in evidence, not longer through generic prose.

## 6. Reproducibility package

Prepare locally before any public release:

- one command for tests and one command for each derived main table/figure;
- locked environment and model/data identity manifests;
- frozen split IDs, seeds, configs, raw prediction paths, and artifact hashes;
- a claim--evidence register linking every main numerical statement to its
  originating output and aggregation script;
- a clean source package that builds without machine-local paths;
- data-acquisition instructions without redistributing restricted archives;
  and
- an anonymized or public repository availability statement, selected by the
  target venue and released only with author authorization.

## 7. Q1 decision gate and Q2 fallback

### Submit on the Q1-primary route only if

- the official semantic evaluator is completed faithfully, or the chosen
  venue's contribution can be defended without that endpoint after an explicit
  editorial audit;
- external calibration and failure decomposition are complete;
- all central claims have traceable artifacts and appropriate paired
  uncertainty;
- the manuscript clearly sells the evaluation framework and nonreplication,
  not a failed state-of-the-art method;
- a clean reproduction package and credible availability statement exist; and
- the current journal scope, quartile database/category/year, length,
  declarations, and artifact rules are verified from official sources.

### Use the Q2 fallback immediately if

- faithful official semantic evaluation remains infeasible;
- the target Q1 venue expects algorithmic novelty or end-to-end benchmark
  superiority that this study does not claim; or
- an independent final audit finds a major validity gap that cannot be repaired
  from frozen outputs.

The Q2 version is not a diminished paper. It is the same rigorous
retrieval-conditioned measurement study with a narrower external-validity
claim. Do not delay it for speculative model additions.

## 8. Definition of done

The one-shot revision is complete when:

- [ ] current results are frozen in a checkpoint and manifest;
- [ ] the official semantic endpoint is either completed or formally recorded
      as unavailable;
- [ ] external calibration metrics and reliability data are generated;
- [ ] the deterministic failure decomposition and transfer synthesis are
      generated;
- [ ] every new number is independently regenerated from preserved outputs;
- [ ] `paper/paper.tex` and the supplement contain the final evidence and no
      unsupported claims;
- [ ] tests, compilation, reference audit, clean-package build, and full-page
      PDF visual inspection pass;
- [ ] venue requirements and ranking status are verified for the submission
      year and category; and
- [ ] an independent final read finds no blocking scientific, reporting, or
      packaging issue.

At that point the paper is ready to submit to the selected venue. Acceptance
and quartile outcome remain decisions of the journal, not properties that can
be guaranteed by the plan.
