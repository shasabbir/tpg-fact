# Publication Confirmation Experiment Ledger

This record is append-only. Failed and repaired runs remain visible.

| Date | Run | Status | Decision / evidence |
|---|---|---|---|
| 2026-11-09 | PC-PROTOCOL-001 | complete | Protocol frozen at `ae68972`; development seal passed; formal-test annotations absent and locked. |
| 2026-11-09 | PC-RETR-FAIL-001 | failed during reporting | All weighted rankings were generated, but CSV reporting rejected the image-only `recall_at_20` column because the header was inferred from a text row containing `recall_at_40`. No final result file was written. Partial artifacts were preserved under `experiments/outputs/publication_confirmation/failed_retrieval_csv_20261109/`. The repair changes only the CSV field union and adds a regression test; rankings, selection rules, protocol, and metrics are unchanged. |
| 2026-11-09 | PC-RETR-001 | complete; advancement gate passed, engineering target failed | Train-only model selection chose equal RRF/cross rank fusion for text and equal SigLIP2/filename fusion for images. On the locked validation population, text Recall@40 increased by 0.00714, image Recall@20 was unchanged, packet-oracle F1 increased by 0.00612, complete-gold coverage increased from 5.30% to 5.91%, and zero-reachable claims fell from 211 to 207. The independent artifact audit passed. The stronger +0.05 oracle-F1 engineering target failed, so the result is a modest retrieval repair rather than a major retrieval contribution. The optional OCR/context branch is not opened because the advancement gate passed. |
| 2026-11-09 | PC-PACKET-FAIL-001 | interrupted before completion | The candidate builder repeatedly decompressed arrays from the frozen NPZ archive inside the per-claim loop. The run was stopped after 25.6 MB of partial JSON and before any manifest, feature archive, audit, or scientific result was produced. The partial directory is preserved under `experiments/outputs/publication_confirmation/failed_candidate_build_repeated_decompression_20261109/`. The repair materializes every immutable ranking array once before iteration and does not change values or protocol choices. |

