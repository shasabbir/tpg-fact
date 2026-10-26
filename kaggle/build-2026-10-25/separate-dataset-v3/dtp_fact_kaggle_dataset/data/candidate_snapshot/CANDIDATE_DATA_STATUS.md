# Candidate snapshot status

- Source: Google Fact Check Tools API `claims.search` metadata.
- Collection timestamp: 2026-10-24T07:53:33.754563Z.
- Deduplicated candidates: 32,898.
- Raw response pages: 440.
- Deterministic screening queue: 1,200 rows, 400 per suggested label.
- API key stored in package: no.

This is a screening snapshot, not the US-TPMFC benchmark. Suggested labels are
automatic mappings from publisher rating text and are not gold labels. The
target image, United States relevance, rating interpretation, decision cutoff,
event identity, and evidence must be audited before any row enters a scientific
experiment. Do not train the final model directly on `candidates.jsonl` or
`screening_queue.csv`.

