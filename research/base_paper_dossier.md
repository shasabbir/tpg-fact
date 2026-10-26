# Base-paper dossier

Reviewed from the two PDFs in `Base papers/` on 2026-10-26. Page references below are PDF pages.

## M-RAV

**Citation.** Son T. Luu, Trung Vo, and Le-Minh Nguyen. “M-RAV: Multimodal retrieve-augment-verify framework for boosting zero-shot fact verification system with large language models.” *Information Processing & Management*, 64, 104988. DOI: 10.1016/j.ipm.2026.104988. Available online 25 June 2026.

### Method

1. Retrieve text with BGE-M3 and images with CLIP from the MOCHEG evidence collection.
2. Form text--image candidate pairs.
3. Use Llama-3.2-90B-Vision-Instruct to verbalize image--text consistency.
4. Score claim--evidence relevance with LLM2Vec.
5. Prompt a text or vision-language LLM to predict supported, refuted, or not enough information.

The paper evaluates MOCHEG, FACTIFY, and Fin-Fact with gold and, for MOCHEG, system-retrieved evidence.

### Strong points

- clean zero-shot framing and broad open-model comparison;
- explicit gold-versus-system evidence analysis;
- public code and public dataset references;
- useful retrieval, top-k, ablation, and error analyses;
- a practical bridge between multimodal retrieval and LLM verification.

### Empirical warning signs that motivate this study

- Image retrieval is weak: at the selected top-5 setting, the reported image F1@K is 6.31 and MAP is 12.32.
- The relevance stage is not consistently helpful. On MOCHEG the macro-F1 difference between full M-RAV and removing relevance is only 0.10 points; on Fin-Fact removing relevance improves macro-F1 from 48.60 to 50.46.
- More aligned pairs reduce verification performance after the top-5 point, indicating evidence overload.
- Several model/dataset combinations deteriorate after M-RAV augmentation.
- NEI remains a major failure mode; the system often converts ambiguity into support or refutation.
- Generated consistency descriptions are free text. Examples in the appendix repeat unsupported contextual assertions and do not expose which image region or textual proposition justified them.

### Improvement taken forward

Replace relevance-only selection and free-form alignment descriptions with an explicit evidence-ID bottleneck. Test whether cited evidence is sufficient, whether deleting it changes the verdict, and whether irrelevant or duplicated evidence leaves the prediction stable.

## MEVER

**Citation.** Delvin Ce Zhang, Suhan Cui, Zhelin Chu, Xianren Zhang, and Dongwon Lee. “MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval.” *Proceedings of EACL 2026*, pages 5236–5255. DOI: 10.18653/v1/2026.eacl-long.242.

### Method

1. Build a two-layer graph of evidence texts and their associated images.
2. Perform image-to-text and text-to-image reasoning inside Transformer and ViT steps.
3. Retrieve evidence with a contrastive objective.
4. Fuse claim and evidence at token and evidence levels for verdict prediction.
5. Generate explanations with multimodal Fusion-in-Decoder.
6. Use a consistency regularizer so the explanation-derived label agrees with the verifier.

MEVER evaluates retrieval, verification, and explanation generation on AIChartClaim, ChartCheck, MOCHEG, and MR2.

### Strong points

- end-to-end treatment of retrieval, verification, and explanation;
- multiple public multimodal datasets and both gold/retrieved settings;
- a strong baseline suite;
- three-run reporting and useful architectural ablations;
- public implementation and release of AIChartClaim.

### Empirical and methodological gaps

- The explanation consistency loss is label consistency, not evidence attribution. An explanation may agree with a wrong or shortcut-driven label.
- Evidence items are mean-pooled in several places, obscuring which item caused the verdict.
- Explanation quality is mostly measured with lexical metrics; the VLM judge does not establish citation correctness.
- No explanation contains a required evidence identifier, so invented or weakly grounded statements cannot be deterministically audited.
- The primary model uses four A100-80GB GPUs, which impairs reproducibility for smaller labs.
- Its maximum text length is 128 tokens, and the paper reports long-evidence failures.
- AIChartClaim is small (1,200 claims), binary, AI-only, and partly generated; it is valuable as an auxiliary set but not sufficient as the sole basis for a broad claim.

### Improvement taken forward

Retain graph-structured multimodal evidence but replace opaque mean aggregation with sparse evidence selection. Generate constrained evidence citations and optimize/evaluate them with counterfactual deletion, addition, duplication, and modality-swap tests.

## Joint conclusion

The defensible novelty is not “another graph,” “another RAG system,” or generic explainability. It is the combination of:

1. a fact-checking-specific evidence citation interface;
2. a sparse bottleneck shared by verdict and explanation;
3. counterfactual training and evaluation of citation faithfulness;
4. public, expert-annotated multimodal scientific data plus general-domain transfer;
5. a resource-conscious implementation.
