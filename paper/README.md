# Temporal and Evidence-Source Provenance-Aware Multimodal Misinformation Detection

This directory contains the editable IEEE TCSS-style manuscript and a reproducible experiment package for the proposed US-TPMFC study.

The living source of truth for scope, milestones, gates, decisions, deviations,
and next actions is [`plan.md`](plan.md). Read and update it before and after
substantive study work.

## Current evidence status

- **Implemented:** project structure, LaTeX manuscript, record schemas, explicit rating normalization, temporal filtering, event-disjoint split checks, lexical retrieval, reciprocal-rank fusion, duplicate/source clustering, calibration and abstention utilities, evaluation metrics, a reference ledger, notebooks, and automated tests.
- **Checked:** the included synthetic smoke test runs end to end and generates machine-readable results plus a LaTeX table.
- **Not yet measured:** the real ClaimReview/GDELT/Common Crawl corpus, multi-GPU training, external benchmark results, and five-seed comparisons.
- **Not claimed:** benchmark scale, superiority over prior work, journal quartile, or any empirical result not produced by the included code.

## Layout

- `paper.tex` and `supplement.tex`: main LaTeX entry points.
- `sections/`: manuscript prose organized by section.
- `references.bib`: peer-reviewed 2020+ references with DOI records.
- `research/`: literature, reference-integrity, and claim-evidence registers.
- `experiments/`: reusable Python modules, configurations, notebooks, tests, and generated outputs.
- `output/pdf/`: compiled deliverables.

## Quick start

From `paper/experiments`, expose `src` and run the dependency-light smoke test:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_smoke_test.py --config configs/base.yaml
python -m unittest discover -s tests -v
```

The smoke test uses only synthetic records. It is a software check, not scientific evidence.

Compile the LaTeX sources with the Codex built-in compiler or an installed IEEE-compatible LaTeX distribution. BibTeX is required for the final bibliography.

## Reproducibility rule

Notebooks may inspect outputs but must import scientific logic from `experiments/src`. Main tables and figures must be generated from preserved outputs rather than copied manually into the manuscript.

