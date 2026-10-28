# DACS-SCV Local Pilot Audit

Audit date: 2026-10-27  
Preserved archive: `dacs-scv-local-pilot-outputs.zip`  
Archive size: 17,549 bytes  
SHA-256: `c17c12299235f361fa96e55c56cb049bc3ba15a03f1d6a66ca07bd4d51f32713`

## Decision

**The decoupled adaptive citation-set pilot passes its predeclared validation gate and justifies freezing a final protocol. It is not a final scientific result.**

The official SciVer test remains untouched. The pilot reuses the independently audited corrected v9 train/validation feature stores and the frozen graph-selector checkpoint. It does not change the verifier, evidence ranking, verdict probabilities, or verdict predictions. A small head observes the sorted frozen selector-score profile and available text/visual item counts, predicts a citation-set size, and returns that many top-ranked evidence identifiers.

## Why the design changed

The joint adaptive model predicted evidence-set size accurately (87.3% exact; MAE 0.17) but changed the ranking and verdict representations. Its citation F1 remained effectively unchanged at 0.4852 and verdict F1 fell to 0.5320, so it failed the safety gate. Error analysis showed that applying its predicted counts to the original frozen ranking would raise citation F1 to 0.5158. This motivated a single, predeclared decoupled design rather than further causal-loss tuning.

## Integrity checks

- 150 unique validation claims align exactly with the audited fixed-top-three baseline.
- Source-paper groups, gold citations, and complete ranked citation lists match the baseline.
- Every stored verdict-probability vector matches the frozen baseline within `1e-7`.
- The cardinality-head checkpoint records hashes of the base checkpoint and train/validation feature stores; all recompute exactly.
- Citation and cardinality metrics recompute exactly from the exported comparison records.
- No official-test file or record is referenced.
- All 33 unit/integration tests pass after the decoupled final-path integration.

## Validation-only measurements

| Selector output | Precision | Recall | Citation macro-F1 | Verdict macro-F1 |
|---|---:|---:|---:|---:|
| Frozen graph ranking, fixed top 3 | 0.3956 | 0.6914 | 0.4851 | 0.5441 |
| DACS-SCV, learned citation count | 0.6933 | 0.4537 | 0.5303 | 0.5441 |

The citation-F1 difference from fixed top three is +0.0452. A 10,000-sample paired source-paper bootstrap gives a mean difference of +0.0450 with a 95% interval from -0.0089 to +0.0991 and an unadjusted two-sided bootstrap p-value of 0.1058. The interval includes zero, so the pilot is a feasibility signal, not evidence of final superiority.

A final pre-freeze diagnostic cut the same frozen ranking at one and two citations. Fixed top one obtains 0.5236 citation macro-F1 and fixed top two obtains 0.5047. DACS-SCV therefore exceeds the best fixed budget by only 0.0067. Its paired grouped-bootstrap mean difference from top one is +0.0066 with a 95% interval from -0.0077 to +0.0219. This stricter comparison is now required in the formal analysis; beating top three alone will not support an adaptive-sizing superiority claim.

Citation-count MAE is 0.5867; exact count match is 0.5067; the predicted mean is 1.14 versus a gold mean of 1.73. The gain appears in both text-only (+0.0403 F1) and visually grounded (+0.0491 F1) subsets. It is positive for the common one- and two-gold-item strata, but negative for the rare three- and four-item strata. These subgroup findings are diagnostic and must be retested on the untouched official test.

## Freeze decision

The final study is renamed **DACS-SCV: Decoupled Adaptive Citation-Set Prediction for Multimodal Scientific Claim Verification**. Counterfactual causal grounding is no longer a contribution. The formal run will compare fixed relevance, flat learned selection, fixed top-one/top-two/top-three graph selection, and DACS-SCV under identical grouped SciVer splits. DACS-SCV is declared successful only if its five-seed official-test citation effect is positive against both its matched fixed-top-three verifier and the best fixed-budget cutoff, with reported group-bootstrap uncertainty, while verdict predictions remain identical.
