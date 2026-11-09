# Publication Confirmation Experiment Ledger

This record is append-only. Failed and repaired runs remain visible.

| Date | Run | Status | Decision / evidence |
|---|---|---|---|
| 2026-11-09 | PC-PROTOCOL-001 | complete | Protocol frozen at `ae68972`; development seal passed; formal-test annotations absent and locked. |
| 2026-11-09 | PC-RETR-FAIL-001 | failed during reporting | All weighted rankings were generated, but CSV reporting rejected the image-only `recall_at_20` column because the header was inferred from a text row containing `recall_at_40`. No final result file was written. Partial artifacts were preserved under `experiments/outputs/publication_confirmation/failed_retrieval_csv_20261109/`. The repair changes only the CSV field union and adds a regression test; rankings, selection rules, protocol, and metrics are unchanged. |

