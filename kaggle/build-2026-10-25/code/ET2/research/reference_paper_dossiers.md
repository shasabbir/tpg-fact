# Reference-paper audit and study implications

**Audit date:** 2026-10-21  
**Scope:** all 17 PDFs in `Reference_good_papers/`  
**Priority rule:** recent A* conference and Q1 journal work receives the greatest design weight; venue labels are a prioritization aid, not a substitute for methodological relevance. The dated venue check is recorded in `venue_quality_audit.md`.

The papers were reviewed at the task, data, method, evaluation, ablation, and limitation levels. The comparison below records what each paper establishes, what it leaves unresolved, and how that affects this study. Page-level extracted text and file hashes are preserved in `research/pdf-index/`.

## Highest-priority design anchors

### RED-DOT, IEEE TCSS 2025

- **Contribution:** jointly trains evidence relevance and verdict prediction, then reranks retrieved text and images before multimodal fact-checking.
- **Strongest lesson:** retrieval quality matters more than simply adding more documents; its experiments often favor a single highly relevant item per modality.
- **Boundary:** the task is binary and centered on out-of-context misinformation. Evidence is not restricted by a decision-time cutoff, copied reports are not collapsed into source families, and the model has no explicit insufficiency or deferral objective.
- **Implication here:** compare against relevance reranking under the same candidate pool, and test whether cluster-level corroboration helps when additional documents are syndicated copies rather than independent sources.

### COVE, NAACL 2025

- **Contribution:** predicts an image's original context before predicting caption veracity. Its context schema covers source, time, place, motivation, entities, and event.
- **Strongest lesson:** intermediate context is useful to both models and human reviewers, and retrieval should recover what an image originally represented rather than only measure image-caption similarity.
- **Boundary:** context labels are partly weak or generated, retrieval can return irrelevant entities, and the pipeline is expensive. It does not enforce a claim-specific decision cutoff or model dependence among web sources.
- **Implication here:** retain original-context retrieval as a baseline and an interpretable feature group, but make availability and source dependence explicit before aggregation.

### ChronoFact, IJCAI 2025

- **Contribution:** represents claim and evidence timelines and applies chronological reasoning to complex temporal fact verification.
- **Strongest lesson:** temporal structure should be represented in the model, not reduced to an undifferentiated text embedding.
- **Boundary:** the task is textual and focuses on the semantics of time within claims and evidence. It does not test whether a document was publicly available at a historical decision time.
- **Implication here:** distinguish **event time** from **evidence availability time**. The principal temporal contribution is a decision-time eligibility rule, with temporal graph features as a secondary mechanism.

### Resolving Conflicting Evidence in Automated Fact-Checking, IJCAI 2025

- **Contribution:** studies RAG decisions when sources disagree and tests source background and credibility interventions.
- **Strongest lesson:** aggressive credibility filtering or reranking can remove useful opposing evidence; supplying source background to the reasoner is safer than treating credibility as a hard relevance score.
- **Boundary:** it is not a multimodal future-event setting and does not model syndicated copies or publication cutoffs.
- **Implication here:** the primary graph models **dependence**, not normative credibility. Credibility is excluded from the main weighting function and may appear only in a disclosed sensitivity analysis.

### MDF-FND, Knowledge-Based Systems 2025

- **Contribution:** models modality uncertainty and dynamically fuses text and image signals with graph attention and Dempster-Shafer evidence combination.
- **Strongest lesson:** weak or conflicting modalities should not be forced into a fixed fusion weight.
- **Boundary:** it is content-only, evaluated on social-media datasets, and does not retrieve external evidence or enforce decision-time validity. Several thresholds and fusion assumptions create sensitivity that requires careful validation.
- **Implication here:** include modality-quality features or an uncertainty-aware fusion comparator, but do not present uncertainty-aware fusion as the main novelty.

### Early Detection via Reinforced Propagation Path Generation, IEEE TKDE 2025

- **Contribution:** generates early social-propagation paths using heterogeneous graph learning, contrastive objectives, and reinforcement learning.
- **Strongest lesson:** deployment timing changes the information available to a detector and deserves explicit early-stage evaluation.
- **Boundary:** social diffusion paths are different from web-evidence provenance, and the method assumes propagation information not guaranteed for ClaimReview cases.
- **Implication here:** cite it to delimit propagation-based early detection. Do not add a propagation module unless the feasibility pilot shows reliable public interaction traces.

### SNIFFER, CVPR 2024

- **Contribution:** combines internal image-text consistency with externally retrieved image-related text in an instruction-tuned multimodal model.
- **Strongest lesson:** internal mismatch and external context are complementary signals.
- **Boundary:** external material is not constrained to what was available at decision time, and source copies are treated as separate retrieved items.
- **Implication here:** use SNIFFER-style internal/external consistency as a strong comparison, then isolate the effect of cutoff filtering and independent-source aggregation.

### Event-Radar, ACL 2024

- **Contribution:** builds an event graph over visual manipulation, textual affect, and event-level inconsistency, with robustness analysis under input noise.
- **Strongest lesson:** relational event structure can improve content-only detection and should be represented in a graph baseline.
- **Boundary:** the graph does not encode external evidence origin, availability, or duplication.
- **Implication here:** include an event-aware no-retrieval graph baseline so gains are not attributed to provenance when they come from graph structure alone.

### MetaSumPerceiver, ACL 2024

- **Contribution:** summarizes multiple multimodal evidence documents with an entailment- and language-model-guided objective.
- **Strongest lesson:** evidence aggregation must control redundancy and produce a compact decision representation.
- **Boundary:** the method assumes a retrieved document set and does not determine whether those documents are time-valid or independent.
- **Implication here:** compare cluster-aware aggregation with a strong document-level summarization baseline, especially under duplicate injection.

### 5Pils, EMNLP 2024

- **Contribution:** retrieves and predicts original image meta-context for visual misinformation.
- **Strongest lesson:** fact-check pages and post-verdict evidence should be excluded from the evidence pool; the paper provides direct precedent for this control.
- **Boundary:** the dataset is modest in size, several context fields rely on weak or automatic labels, and retrieval failures remain substantial.
- **Implication here:** adopt the exclusion principle but make it reproducible through archived pre-cutoff observations and timestamp-confidence tiers.

### MOCHEG, SIGIR 2023

- **Contribution:** provides a large multimodal fact-checking dataset and an end-to-end retrieval, verification, and explanation task.
- **Strongest lesson:** retrieval and verdict prediction should be evaluated separately, with text and image evidence both available.
- **Boundary:** evidence is drawn from fact-check material, which can expose verdict-bearing information; document-level top-k retrieval can also introduce noisy images.
- **Implication here:** use MOCHEG for transfer or controlled baselines, not as evidence that the proposed temporal protocol is already solved.

## Supporting and lower-priority references

### MultiReflect, MAGMaR 2025

- **Contribution:** a staged self-reflective web-search pipeline that rates evidence for authority, timeliness, relevance, support, and usefulness.
- **Value:** demonstrates that iterative retrieval and reflection can improve binary verification.
- **Boundary:** the study uses closed multimodal models, does not impose a hard historical cutoff, and performs weakly on the harder VERITE multiclass setting.
- **Use here:** an optional reasoning baseline; not the methodological anchor because reproducibility and information-access controls are weaker.

### RAGAR, FEVER 2024

- **Contribution:** chain- and tree-structured multimodal RAG for political fact-checking.
- **Value:** supplies a direct vanilla/structured RAG comparison.
- **Boundary:** only 300 binary cases, live-search variability, closed-model dependence, and no insufficient-evidence label or source-dependence model.
- **Use here:** comparison baseline and prompt-design reference, with frozen retrieval results required for reproducibility.

### Learning from Synthetic Data, Findings of EMNLP 2024

- **Contribution:** selects useful synthetic multimodal misinformation examples for transfer to real data.
- **Value:** addresses distribution mismatch and sample selection.
- **Boundary:** it does not retrieve evidence or model provenance and is therefore orthogonal to the main causal question.
- **Use here:** possible training augmentation after the real-data pilot, never a substitute for the future-event test.

### Temporal Generalizability, GenBench 2023

- **Contribution:** shows substantial degradation under chronological multimodal misinformation splits.
- **Value:** motivates future-event evaluation and warns against random splits.
- **Boundary:** the model is content-only and the study does not isolate evidence leakage.
- **Use here:** justifies the chronological protocol and the separate reporting of 2025 and partial-2026 results.

### VERITE, IJMIR 2024

- **Contribution:** a 1,000-item real-world benchmark balanced to reduce unimodal shortcuts across true, miscaptioned, and out-of-context cases.
- **Value:** a strong out-of-distribution test for content and context models.
- **Boundary:** no retrieved-evidence timestamps, dependency clusters, or insufficiency label.
- **Use here:** an external stress test, with claims limited by the available labels and evidence metadata.

### NewsCLIPpings, EMNLP 2021

- **Contribution:** a foundational large-scale benchmark of genuine images paired with synthetic out-of-context captions.
- **Value:** supports controlled pretraining and measures image-caption mismatch.
- **Boundary:** synthetic construction and lack of web-evidence provenance limit real-world decision-time conclusions.
- **Use here:** pretraining or content-only baseline only.

## Resulting novelty position

The review does not support claiming novelty for multimodal RAG, graph encoding, image-context prediction, temporal reasoning, uncertainty-aware fusion, or evidence reranking on their own. Strong prior work already covers each element.

The defensible contribution is the **joint evaluation and implementation of decision-time evidence validity and independent corroboration** in future-event multimodal fact-checking:

1. Evidence enters the primary model only when a pre-cutoff archive or crawl observation supports its availability. Self-reported dates alone enter a sensitivity analysis, not the primary test.
2. Copied, linked, and syndicated documents are grouped into source-dependence clusters. Each cluster contributes bounded mass, so ten copies of one report cannot imitate ten independent sources.
3. Evidence sufficiency is supervised separately from the professional verdict on an audited subset. The system can predict `insufficient_evidence`, while calibrated selective deferral remains a distinct operational action.
4. Counterfactual interventions add copies, inject post-cutoff verdicts, remove the earliest source, and vary dependent versus independent conflicts. These tests measure the claimed mechanism directly.
5. Evaluation is event-disjoint and chronological. The 2026 portion is explicitly a partial-year snapshot frozen on 2026-10-21 and is reported separately from 2025.

The central research question is therefore:

> Does decision-time-valid, independently corroborated evidence improve future-event multimodal fact-checking and selective risk relative to matched document-level RAG?

The graph encoder is one implementation of this hypothesis. If a simpler cluster-aware aggregator performs equally well, the paper should report that result and narrow the method claim rather than protect the graph architecture.

## Required changes before a journal-level empirical claim

- Freeze a professional fact-checker allowlist and a reproducible collection snapshot.
- Audit timestamp confidence and evidence sufficiency on the 200-case pilot.
- Validate source-dependence clusters against a manual sample; report pairwise precision, recall, and F1.
- Run matched reranking, original-context, event-graph, and document-level RAG baselines.
- Separate retrieval, cluster quality, classification, calibration, and selective-risk results.
- Report 2025 and partial-2026 tests separately and together.
- Preserve negative results and remove any contribution whose preregistered falsifier is met.

