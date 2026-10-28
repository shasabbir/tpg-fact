# Search log

Search date: 2026-10-26. Sources were restricted to primary papers, official proceedings, official dataset pages, and author repositories for technical claims.

## Queries

- `2025 2026 multimodal fact verification adaptive evidence selection evidence utility public dataset`
- `site:aclanthology.org 2025 multimodal fact checking evidence retrieval verification`
- `SAFE-MR multimodal fact checking evidence sufficiency selective`
- `evidence utility multimodal fact verification retrieval`
- `2025 2026 multimodal fact checking explanation faithfulness citations evidence attribution public dataset`
- `multimodal fact checking visual region attribution explanation citation counterfactual evidence`
- `multimodal fact checking train test leakage event disjoint image duplicates evidence shortcut`
- `SciVer dataset GitHub 3000 multimodal scientific claim verification 2025`
- `MuSciClaims dataset github 2025`

## Key verified sources

- M-RAV version of record: https://doi.org/10.1016/j.ipm.2026.104988
- MEVER proceedings paper: https://aclanthology.org/2026.eacl-long.242/
- MEVER repository and public data link: https://github.com/cezhang01/mever
- MOCHEG repository: https://github.com/PLUM-Lab/Mocheg
- SciVer paper: https://aclanthology.org/2025.acl-long.420/
- SciVer public dataset: https://huggingface.co/datasets/chengyewang/SciVer
- MuSciClaims paper: https://aclanthology.org/2025.ijcnlp-long.175/
- MuSciClaims public dataset: https://huggingface.co/datasets/StonyBrookNLP/MuSciClaims
- Evidence attribution evaluation: https://aclanthology.org/2025.naacl-long.282/
- DEFAME: https://proceedings.mlr.press/v267/braun25b.html
- Utility-Oriented Visual Evidence Selection: https://aclanthology.org/2026.acl-long.1620/
- Selective Multimodal Retrieval: https://aclanthology.org/2026.fever-1.10/
- RW-Post/AgentFact: https://openaccess.thecvf.com/content/CVPR2026W/PPMisDet/html/Xu_RW-Post_Auditable_Evidence-Grounded_Multimodal_Fact-Checking_in_the_Wild_CVPRW_2026_paper.html

## Access checks

- `chengyewang/SciVer`: Hugging Face API reports public, ungated, revision `5695ac384d247ae5df2a7b73568dc63d189296b0`, CC BY 4.0.
- `StonyBrookNLP/MuSciClaims`: Hugging Face API reports public, ungated, revision `9134142d05fb473e0a10c4c05f0c2d5a2af51099`; the API card does not declare a license, so terms remain a gate.
- SciClaimEval has public development/test resources, but its full release is scheduled for 2026-12-11. It is not a core dataset in the current plan.

## Peer-reviewed expansion audit (2026-10-26)

The literature search was expanded across Crossref DOI records, ACL Anthology, ACM Digital Library, IEEE/CVF proceedings, SpringerLink, and Elsevier journal landing pages. Queries combined `multimodal misinformation`, `multimodal fake news`, `out-of-context image`, `evidence retrieval`, `scientific claim verification`, `fact-check explanation`, `citation attribution`, and `faithful rationale` with publication years 2020--2027. Backward and forward chaining began from M-RAV, MEVER, MOCHEG, SciVer, MuSciClaims, SNIFFER, NewsCLIPpings, and the 2025 evidence-attribution paper.

Inclusion required a peer-reviewed journal, conference, or archival workshop version; a resolvable DOI; and direct relevance to task definition, multimodal fusion, evidence retrieval, scientific verification, or faithful attribution. arXiv-only manuscripts, duplicate preprints, DOI-less candidate papers, and works without a supportable role in the review were excluded. The final manuscript bibliography contains 50 DOI-identified peer-reviewed sources: 27 journal articles and 23 conference/workshop papers. Crossref relation fields exposed no `updated-by` or `update-to` notices for the retained records at the time of the audit. Claim-level support is recorded in `reference_ledger.csv`; paywalled papers are used only for claims supportable from authoritative titles/abstracts and bibliographic records.

The supplied M-RAV version of record states that it was available online on 25 June 2026 and assigned to volume 64 (2027). The bibliography therefore cites the 2026 online publication and records the 2027 issue in a note. The Crossref cache used by the helper splits the first author's name for Shah and Ganatra (2022), while the live Crossref record and inspected publisher identity give Minal Nirav Shah and Amit Ganatra; that automated name warning remains documented rather than silently ignored.

## Positive-study pivot audit (2026-10-28)

The completed SciVer experiment was treated as a closed negative study rather than retuned after its official test was consumed. A new benchmark and method were selected so that any later confirmatory claim can be evaluated on untouched data.

Verified benchmark sources:

- MOCHEG version of record: https://doi.org/10.1145/3539618.3591879
- MOCHEG official repository: https://github.com/PLUM-Lab/Mocheg
- MOCHEG official access form: https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform?usp=sf_link

The paper and repository establish a public CC BY 4.0 benchmark of 15,601 claims with official train/development/test partitions, multimodal evidence, and supported/refuted/not-enough-information labels. The release archive is not yet local, so schema inspection and test-identity sealing remain mandatory preflight gates.

Focused novelty queries covered `adaptive evidence set size fact checking`, `expected F1 decoding evidence selection`, `minimal evidence claim verification`, `utility-oriented visual evidence selection`, `MOCHEG top k evidence`, and `calibrated set prediction information retrieval`. Closest verified works were:

- Utility-Oriented Visual Evidence Selection: https://doi.org/10.18653/v1/2026.acl-long.1620
- Learning to mine all minimal evidences for unverified claims: https://doi.org/10.1016/j.ins.2025.122505
- M-RAV: https://doi.org/10.1016/j.ipm.2026.104988

The retained novelty boundary is narrow: CEFES selects a variable multimodal evidence set by maximizing exact expected set-F1 under calibrated candidate probabilities. It is not presented as the first adaptive retriever, the first helpfulness ranker, or the first minimal-evidence method. The primary comparison is against a development-selected fixed top-k selector with identical candidates, ranking features, and verifier. The MOCHEG official test remains sealed until the development gate in `plan.md` passes and the protocol is frozen.
