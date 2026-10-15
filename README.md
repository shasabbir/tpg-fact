# Temporal and Evidence-Source Provenance-Aware Multimodal Misinformation Detection

This repository contains the editable IEEE TCSS-style manuscript and a reproducible experiment package for the proposed US-TPMFC study.

The living source of truth for scope, milestones, gates, decisions, deviations,
and next actions is [`plan.md`](plan.md). Read and update it before and after
substantive study work.

## Current evidence status

- **Implemented:** project structure, LaTeX manuscript, record schemas, explicit rating normalization, temporal filtering, event-disjoint split checks, lexical retrieval, reciprocal-rank fusion, duplicate/source clustering, calibration and abstention utilities, evaluation metrics, a reference ledger, notebooks, and automated tests.
- **Checked:** the included synthetic smoke test runs end to end and generates machine-readable diagnostic outputs.
- **Not yet measured:** the real ClaimReview/GDELT/Common Crawl corpus, multi-GPU training, external benchmark results, and five-seed comparisons.
- **Not claimed:** benchmark scale, superiority over prior work, journal quartile, or any empirical result not produced by the included code.

## Layout

- `paper/paper.tex` and `paper/supplement.tex`: main LaTeX entry points.
- `paper/sections/`: manuscript prose organized by section.
- `paper/references.bib`: peer-reviewed 2020+ references with DOI records.
- `paper/figures/`, `paper/tables/`, and `paper/output/pdf/`: manuscript assets and compiled deliverables.
- `research/`: literature, reference-integrity, and claim-evidence registers.
- `experiments/`: reusable Python modules, configurations, notebooks, tests, and generated outputs.

## Quick start

The complete Windows setup, pilot-data, five-seed, evaluation, notebook, and LaTeX commands are documented in [`experiments/README.md`](experiments/README.md).

For an already configured environment, run the dependency-light smoke test from `experiments`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_smoke_test.py --config configs/base.yaml
python -m unittest discover -s tests -v
```

The smoke test uses only synthetic records. It is a software check, not scientific evidence.

Compile the LaTeX sources with the Codex built-in compiler or an installed IEEE-compatible LaTeX distribution. BibTeX is required for the final bibliography.

## Reproducibility rule

Notebooks may inspect outputs but must import scientific logic from `experiments/src`. Main tables and figures must be generated from preserved outputs rather than copied manually into the manuscript.

