I've reviewed the manuscript, **“Retrieval-Conditioned Calibration and Adaptive Evidence Selection for Multimodal Fact-Checking.”**

My initial assessment is that this is a promising research paper with a meaningful methodological contribution. However, I would **not recommend submitting the current version directly to a Q1 journal yet.**

The good news is that I don't think the research needs to be abandoned or completely redesigned. The main opportunity is to strengthen the experimental evidence, establish the novelty more convincingly, and improve the positioning against recent multimodal fact-checking research.

There is also something important about this paper: its negative findings are scientifically meaningful. You do not claim that every adaptive selection method works. Instead, you show that retrieval reachability, probability calibration, and downstream verification affect whether adaptive evidence selection actually helps. That is a potentially valuable research contribution.

## **1. Overall assessment**

**Q1 readiness**

Major revision required

# **Not ready yet**

My qualitative assessment of the submitted 18-page manuscript, not an editorial decision.

| Evaluation area          | Assessment        |
| ------------------------ | ----------------- |
| Research problem         | Strong            |
| Research motivation      | Strong            |
| Experimental design      | Good foundation   |
| Methodological novelty   | Moderate          |
| Retrieval performance    | Weak              |
| Calibration contribution | Promising         |
| External validation      | Insufficient      |
| Statistical analysis     | Good foundation   |
| Reproducibility          | Needs improvement |
| Writing and presentation | Needs polishing   |

### **What is already good?**

There are several aspects I would preserve:

1. **The research question:** Separating retrieval reachability from evidence-selection quality is important.
2. **The calibration experiments:** Comparing six probability maps is a useful component.
3. **The matched comparisons:** Fixed top-k, thresholding, learned count, expected-F1 and completeness-aware switching are evaluated on shared candidate packets.
4. **Statistical discipline:** Five deterministic partitions, 10,000 bootstrap draws, multiplicity correction and explicit superiority criteria are positive features.
5. **Negative findings:** You report when an approach fails rather than presenting only favorable results.

The manuscript's pages 8–10 and 13–15 provide particularly useful evidence for these strengths.

### **The five most important weaknesses**

**1. Retrieval performance is too limited.**

Your repaired MOCHEG retrieval achieves:

- Packet-oracle F1: 0.3637
- Complete-gold coverage: 4.74%
- Text Recall\@40: 0.3393
- Image Recall\@20: 0.3287

This means the selector is operating with a severely restricted evidence pool. The retrieval repair is also relatively small compared with the original packet.

A Q1 reviewer will likely ask:

- Why should this retrieval configuration be considered sufficient?
- How does it compare with recent multimodal retrieval systems?
- Can better retrieval change the conclusions about adaptive selection?

**What I recommend:** Add stronger retrieval baselines, increase retrieval depth, and evaluate the effect of retrieval quality on selection performance. Report the retrieval ceiling separately from selector performance.

**2. The external validation is not sufficiently convincing.**

This is probably the biggest weakness.

Your AVerImaTeC experiment uses exact normalized source identity rather than the official semantic evaluator.

The development data contains:

- 152 claims.
- 141 Refuted claims.
- 6 Not Enough Evidence claims.
- 4 Supported claims.
- Only 1 Conflicting Evidence claim.

Additionally, 59.87% of claims have no reachable annotated target.

The paper itself acknowledges that the official Gemma-3-27B evaluator was unavailable and that the external experiment is a diagnostic nonreplication, not an official shared-task result.

This is scientifically honest, but it considerably limits the strength of your generalization claims.

**What I recommend:** Obtain an official semantic evaluation or conduct an independently annotated human evaluation on a sufficiently large sample. Ideally, add another independent benchmark.&#x20;

paper-2.pdf



**3. The methodological novelty needs strengthening.**

Currently, most of the individual components are established techniques:

- BGE-M3 retrieval.
- BM25.
- Reciprocal rank fusion.
- SigLIP2.
- Logistic regression.
- Platt, beta and isotonic calibration.
- Expected-F1 decoding.

Your contribution is primarily their controlled integration and evaluation.

That can be publishable, but a high-ranking journal will need a clearer answer to:

**What fundamentally new knowledge or methodology does this paper introduce that existing calibration and evidence-retrieval studies do not?**

I would develop a more explicit retrieval-conditioned decision framework, rather than presenting the contribution as simply comparing several existing selection methods.

**4. The downstream verifier is relatively weak.**

You use a small MLP as the shared verifier.

On MOCHEG, the switch's aggregate verifier improvement is not statistically superior, and the per-seed safeguard fails.

On AVerImaTeC, verifier macro-F1 decreases from 0.1573 to 0.1444 when using the switch.

I recommend introducing at least one stronger evidence-aware verifier that actually processes the retrieved textual and visual evidence. Compare the selected evidence policies using the same verifier architecture and training conditions.

**5. The validation history and reproducibility need attention.**

Your limitations section acknowledges that the MOCHEG validation resource had already been examined in the larger project. This weakens its status as completely independent confirmation.

Also, although the paper describes hashed artifacts and reproducibility records, I could not identify a public code and data repository link in the manuscript.

I recommend:

- A publicly accessible GitHub repository.
- Frozen configurations and experiment manifests.
- Dataset preparation scripts.
- Reproducible result-generation scripts.
- A clear record of which datasets and annotations are publicly available.

paper-2.pdf

## **2. What I would change before Q1 submission**

I would organize the revision into three levels.

### **Priority A: Essential experiments**

| Experiment                   | What to add                                                     | Why it matters                       |
| ---------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| Independent evaluation       | Official MOCHEG test or another genuinely independent benchmark | Establish generalization             |
| Semantic evidence evaluation | Official AVerImaTeC evaluator or expert human annotation        | Validate actual evidence correctness |
| Retrieval comparison         | Recent state-of-the-art text and image retrievers               | Establish retrieval competitiveness  |
| Retrieval-depth analysis     | Multiple candidate-packet sizes                                 | Understand the retrieval ceiling     |
| Stronger verifier            | Evidence-aware transformer or multimodal LLM                    | Validate downstream usefulness       |
| External calibration         | Brier, NLL, ECE and reliability diagrams                        | Establish calibration transfer       |

One particularly important issue: your external experiment reports the difference between calibrated and uncalibrated expected-F1, but does not report external Brier, NLL or ECE. I would add those metrics before claiming that probability calibration itself generalizes across benchmarks.

### **Priority B: Improve the actual research contribution**

I see an opportunity to develop a more substantial method.

I would investigate a new **Reachability-Aware Calibration and Evidence Selection Framework**.

The framework could contain:

**Multimodal evidence retrieval**

Text and image candidate generation

**Evidence reachability estimation**

Estimate whether sufficient evidence exists in the retrieved packet

**Prior-aware probability calibration**

Estimate reliable candidate membership probabilities

**Dependence-aware evidence selection**

Account for duplicates, source relationships and complementary modalities

**Evidence-aware verification**

Evaluate evidence quality and downstream verdict preservation

The most interesting methodological opportunity is addressing the conditional-independence assumption in expected-F1 decoding. Your manuscript already acknowledges that evidence from the same source, duplicate images and complementary cross-modal evidence can violate that assumption.

A structured or dependence-aware selection method could be a genuine contribution, provided it demonstrates measurable improvements over the existing policies.

### **Priority C: Improve the manuscript**

I would also:

1. Rewrite the abstract around the strongest scientifically supported findings.
2. Strengthen the novelty and contribution statements.
3. Add a dedicated comparison against recent 2025–2026 literature.
4. Improve the explanation of the statistical methodology.
5. Publish the complete reproducibility package.
6. Clearly distinguish development evidence, independent validation and official test results.

One important positioning decision: I would stop presenting adaptive-selector superiority as the central achievement. Your strongest finding is that retrieval reachability and probability calibration are prerequisites for trustworthy adaptive evidence selection.

That is a much more defensible research story.

## **3. Where should you submit?**

I investigated journals that are relevant to this research area and checked their latest verifiable 2025 journal metrics.

These are my recommendations.

[Science China Information Sciences_影响因子8.1 - Book学术](https://images.openai.com/static-rsc-4/_Evxt8sGSGuipE91-MDbDzQJdc9kYl-ff1kci2OkYyDeTe8dTmBnMU3XoQ6Nu9I5n2-p6LEpnTL5flgrVXV2uXD9XY_ma3DB_1TpNEclw2SOoUUfRs091b0dMCSkn6mGWj1zYdf9PJnWICGNAeBpZYvlQsmnmCTzWsyyjhzDu_w?purpose=inline)

My first recommendation

**1. Information Processing & Management**

[Journal website](https://www.sciencedirect.com/journal/information-processing-and-management?utm_source=chatgpt.com)

Elsevier · 2025 JCR Q1 · 2025 SJR Q1

- 2025 Impact Factor: 8.1
- 2025 SJR: 1.845

**Why it fits:** Its scope includes information retrieval, multimedia processing, information systems, novel methods and critical applications. It has also published work on NLP-based fact-checking.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.iit.comillas.edu\&sz=32)

iit.comillas.edu+2



Your paper's retrieval-conditioned evaluation and calibration framework align particularly well with this journal.

[Academic Service – The S²↔︎AM Lab](https://images.openai.com/static-rsc-4/HxQvfU5hjWaZjAsbo5T377_ANLRqL5Oen6pD-1-WJUw3hdJTV_Y0wqWrnN_W9CnM-gkxubcr2v3aLcvuG3icKXTQLcglHmCX4BkmDd43gOGoQfoBGjv2ekI4VB7l7OMXLZoDaMRlWhUvOnk_-U-xhtE-sM8BLiS9oLxWK0XQAo8?purpose=inline)

**2. Knowledge-Based Systems**

[Journal website](https://www.sciencedirect.com/journal/knowledge-based-systems?utm_source=chatgpt.com)

Elsevier · 2025 JCR Q1 · 2025 SJR Q1

- 2025 Impact Factor: 8.0
- 2025 SJR: 1.753

**Why it fits:** AI-based decision support, machine learning methodologies, intelligent systems and uncertainty-aware information processing are within its scope.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.iit.comillas.edu\&sz=32)

iit.comillas.edu+1



However, I would target this journal only after developing a stronger original selection or calibration methodology.

[Expert Systems With Applications (ESWA) (Q1 Journal)](https://images.openai.com/static-rsc-4/vlXmjJkXcKhHhSI88obA4t6YM3CO0wKBTQV_b2Bp_O6hdJiFGnSvgYT_c3vQNWZfV4MJCW26p2bAP77SnGLsPx4ks77eFTOF_6a4qIoo8oPTjT3jhxgwxUXdQ_GezpSd0jnh5GwZlmZeNfjlvEZFOkx4XbqCWGBtJZ8HK3iy_-4?purpose=inline)

**3. Expert Systems with Applications**

[Journal website](https://www.sciencedirect.com/journal/expert-systems-with-applications?utm_source=chatgpt.com)

Elsevier · 2025 JCR Q1 · 2025 SJR Q1

- 2025 Impact Factor: 9.4
- 2025 SJR: 1.939

**Why it fits:** Its scope explicitly includes intelligent systems, information retrieval, multimedia mining, machine learning and practical AI applications.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.iit.comillas.edu\&sz=32)

iit.comillas.edu+1



A stronger implemented fact-checking system with convincing experimental validation would improve your chances of fitting this journal.

*The quartiles above refer to the 2025 JCR and SJR data I could verify. Quartiles can differ by subject category and ranking system, so check the exact category required by your institution before submission.*

### **What about Information Fusion?**

[Information Fusion容易中吗](https://images.openai.com/static-rsc-4/AsqXA7hZGA19j0fb7ZHKFOGFW8KU-jhjvDt2p0JqJ5mPEd7ltoTNOuDxmE3YQXmEQE8ADzhc3-beaKfX73b0qee4POVZdOzH5nDrxijk65-h4EGLqAFe9N1GINyOvgU9SjJ3-Id_pYQ_4VzUUcuBMJspNId8rvLFeoAm9HbBOBk?purpose=inline)

**Information Fusion**

[Journal website](https://www.sciencedirect.com/journal/information-fusion?utm_source=chatgpt.com)

Not my recommendation currently

2025 SJR Q1 · SJR 4.197

Its central scope is multi-source, multi-sensor and multi-process information fusion. Your current manuscript uses multimodal retrieval and rank fusion, but does not introduce a sufficiently novel information-fusion architecture.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.scimagojr.com\&sz=32)

SJR, Quartile, H-index



## **4. My recommended publication strategy**

This is how I would proceed:

1. **Do not submit the current manuscript immediately.**

   Address the evaluation and novelty limitations first.
2. **Strengthen the experiments.**

   Prioritize independent semantic evaluation, retrieval baselines, stronger verification and external calibration.
3. **Develop the methodological contribution.**

   Investigate dependence-aware, reachability-conditioned evidence selection.
4. **Reorganize the manuscript.**

   Make calibration, reachability and reliable evidence selection the central research contribution.
5. **Submit to Information Processing & Management first.**

   Keep Knowledge-Based Systems and Expert Systems with Applications as alternatives depending on the final methodological contribution.

## **5. Final verdict**

I want to emphasize one thing: **this is not a weak research idea. It is an underdeveloped Q1 submission, not a research project that needs to be discarded.**

The manuscript has a good experimental foundation, thoughtful statistical controls and scientifically honest negative results.

But there is a difference between demonstrating that existing methods behave differently under controlled conditions and introducing a sufficiently novel, independently validated contribution for a competitive Q1 journal.

My recommendation is to invest in the additional experiments and methodological development rather than submitting it prematurely.

**I would like to take this paper toward Information Processing & Management, with a major-revision strategy.**

The next useful step would be to prepare a detailed, section-by-section revision plan for all 18 pages, identifying exactly what to retain, remove, rewrite and experimentally add before submission.