# Retrieval-Conditioned CEFES Phase-2 Protocol

Frozen: 2026-11-02

Status: development protocol. It does not authorize access to MOCHEG formal-test annotations.

## Objective

Measure how retrieval quality changes the ordering of fixed and adaptive evidence selectors, and test whether a retrieval-conditioned, precision-aware CEFES variant improves complete-gold evidence-set F1 without relying on very large selected sets.

## Confirmatory boundary

The completed negative study in `experiments/outputs/mocheg/` remains immutable. Phase-2 artifacts are written only under `experiments/outputs/q1_phase2/`. The official MOCHEG test remains sealed. The already-inspected MOCHEG development split is a locked benchmark check, not fresh confirmation.

## Data use

All released MOCHEG training claims are partitioned by a deterministic SHA-256 claim-group rule into fit, calibration, model-selection, and internal-holdout groups. Retrieval indexes contain the released train/development public corpus and train/development image paths, but relevance labels are joined only after rankings are frozen. The complete released gold set remains in metric denominators.

## Experiment sequence

1. Reproduce the frozen dense retrieval metrics through the new interface.
2. Compare dense sentence retrieval, BM25 sentence retrieval, reciprocal-rank fusion, and frozen cross-encoder reranking.
3. Compare SigLIP image retrieval, filename-text dense retrieval, and their reciprocal-rank fusion.
4. Apply the retrieval gate before interpreting any new selector as an improvement.
5. Fit shared and modality-specific logistic membership scorers. Add the frozen text cross-encoder score only in the declared augmented scorer.
6. Calibrate on claim-disjoint training partitions and select the calibrator by training-only Brier score.
7. Compare the strongest fixed budget, threshold, count, original CEFES, size-penalized CEFES, and size-plus-redundancy-penalized RC-CEFES on identical packets.
8. Run the shared verifier only after the primary evidence analysis is frozen.
9. Audit all identities, metrics, intervals, and gates independently.
10. Update the manuscript with the checked outcome, including negative or mixed results.

## Stop rules

- If text and image Recall@50 do not each improve by at least 0.10 absolute without reducing the oracle ceiling, no decoder-superiority claim is pursued.
- If the augmented scorer does not improve ranking in every seed while preserving calibration, it remains a negative ablation.
- If the selector superiority gate fails, the official test remains sealed.
- A favorable verifier result cannot replace a failed evidence-set endpoint.

## Deviations from the expansion plan

OCR and generated image captions are not included in the initial frozen matrix because they introduce an additional model family and a large unmeasured compute budget. Filename-text retrieval is the declared low-cost image-text branch. OCR/caption retrieval may only be added under a separately frozen extension after the initial matrix is complete; it cannot replace an unfavorable result from this protocol.

## Source records

The machine-readable source of truth is `experiments/configs/q1_phase2/protocol.yaml`. Every run must record its configuration hash, source commit, model revisions, input hashes, candidate-ranking hashes, runtime, hardware, and formal-test seal state.

