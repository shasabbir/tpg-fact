# Novelty audit

## Candidate retained

**Evidence-Traceable Multimodal Claim Verification with Counterfactual Citation Grounding**

The paper asks whether the evidence cited by a multimodal verifier actually causes and supports its decision. The model must predict a verdict, select evidence identifiers, and generate citation-constrained explanations. Attribution is tested through gold evidence and controlled interventions.

## Closest work and non-overlap

| Work | Closest overlap | Remaining distinction |
|---|---|---|
| M-RAV (IPM, 2026) | multimodal retrieval, evidence verbalization, zero-shot verification | no explicit citations, no causal attribution test, fixed evidence budget |
| MEVER (EACL, 2026) | graph retrieval, verdict, explanation generation | label consistency rather than evidence attribution; mean-pooled evidence; no citation constraints |
| Evaluating Evidence Attribution in Generated Fact Checking Explanations (NAACL, 2025) | citation masking and recovery | text-only evaluation protocol; TRACE-SCV trains a multimodal verifier and tests verdict dependence |
| SciVer (ACL, 2025) | expert multimodal evidence and rationales | benchmark/evaluation paper; TRACE-SCV proposes a grounded model and counterfactual protocol |
| MuRGAt (2026 preprint) | multimodal fact-level citations | broad audio/video reasoning benchmark; not fact-checking-specific and not the same training objective |
| SAFE-MR (2026 preprint) | evidence sufficiency, interventions, selective prediction | rumor detection and abstention; no citation-constrained explanations or scientific document evidence |
| Utility-Oriented Visual Evidence Selection (ACL, 2026) | relevance is not utility | general multimodal RAG evidence ranking; no verdict/explanation citation grounding |
| AMuFC (2026 preprint) | adaptive visual evidence use | binary modality necessity rather than evidence-item attribution |
| RW-Post/AgentFact (CVPRW, 2026) | auditable evidence-grounded fact checking | dataset and agent baseline; TRACE-SCV tests sparse causal citations on existing public benchmarks |

## Rejected directions

- **Temporal provenance + calibrated abstention:** too close to SAFE-MR and several 2026 provenance/agent systems; the former private benchmark also created avoidable data burden.
- **Utility-aware evidence selection alone:** directly overlaps ACL 2026 work.
- **Adaptive use of images alone:** overlaps AMuFC.
- **Generic graph fusion:** MEVER and many earlier systems already occupy this space.
- **New public benchmark collection:** violates the user's preference to avoid dataset collection and adds a weak point that is unnecessary for this hypothesis.

## Novelty risk

The proposal remains a hypothesis until the complete 2026 literature sweep and data feasibility audit finish. The manuscript must avoid first-ever language. The safest contribution wording is:

> We study counterfactually validated evidence citations for multimodal claim verification and introduce a model in which verdict prediction and explanation generation share a sparse, auditable evidence bottleneck.

## Publication bar

For a credible Q1/Q2 journal submission, the final study needs all of the following:

- at least two public datasets in main experiments and one external-transfer dataset;
- locally reproduced matched baselines;
- verdict and attribution endpoints;
- five-seed uncertainty and paired tests;
- complete ablations;
- robustness interventions;
- resource accounting;
- no unsupported superiority or “faithful” claim when deletion tests fail.
