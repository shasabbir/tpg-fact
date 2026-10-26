# Decision-Time Provenance for Multimodal Fact-Checking

This repository contains the editable IEEE TCSS-style manuscript and a reproducible experiment package for the proposed US-TPMFC study. The central test concerns decision-time-valid evidence and independent corroboration, not graph architecture by itself.

The living source of truth for scope, milestones, gates, decisions, deviations,
and next actions is [`plan.md`](plan.md). Read and update it before and after
substantive study work.

## Current evidence status

- **Implemented:** the data/audit pipeline plus dataset and feature freezing, fused BM25/BGE-M3/SigLIP2 retrieval with pinned BGE reranking, reusable graph and matched feature models, validation-only hyperparameter selection, five-seed baseline/ablation execution, calibration and abstention, paired event bootstrap, Holm adjustment, generated LaTeX tables, and a resumable final-run orchestrator.
- **Prepared data:** the redesigned evidence-first pilot contains 200 frozen image--text claims and 682 time-valid evidence records. All 200 claims have evidence, 141 have at least two source-domain clusters, and the 50/39/111 temporal splits pass exact event-ID and pHash leakage checks. It remains an AI-assisted pilot, not the final benchmark.
- **Checked:** 44 unit/integration tests, Python compilation, the frozen pilot, a tiny CPU training/checkpoint/calibration cycle, synthetic smoke workflow, the pilot validation notebook, and page-by-page manuscript PDF inspection pass.
- **Not yet measured:** dense real-pilot features, CUDA pilot training, the final 2,000-case corpus or fallback benchmark, robustness effects, and final five-seed comparisons.
- **Not claimed:** benchmark scale, superiority over prior work, journal quartile, or any empirical result not produced by the included code.

## Layout

- `paper/paper.tex` and `paper/supplement.tex`: main LaTeX entry points.
- `paper/sections/`: manuscript prose organized by section.
- `paper/references.bib`: peer-reviewed 2020+ references with DOI records.
- `paper/figures/`, `paper/tables/`, and `paper/output/pdf/`: manuscript assets and compiled deliverables; `paper/output/overleaf-main-2026-10-24-v9.zip` is the clean-tested main-paper upload package.
- `research/`: literature, reference-integrity, and claim-evidence registers.
- `research/reference_paper_dossiers.md`: detailed audit of all 17 supplied papers and its consequences for novelty, baselines, and evaluation.
- `experiments/`: reusable Python modules, configurations, notebooks, tests, and generated outputs.

## Quick start

The complete Windows setup, pilot-data, five-seed, evaluation, notebook, and LaTeX commands are documented in [`experiments/README.md`](experiments/README.md).

For an already configured environment, run the dependency-light checks from `experiments`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_smoke_test.py --config configs/base.yaml
python scripts/audit_feasibility.py --claims outputs/smoke/claims.jsonl --evidence outputs/smoke/evidence.jsonl --output outputs/smoke/feasibility.json
python -m pytest -q
```

The smoke test and tiny CPU training test use synthetic records. They are software checks, not scientific evidence. The exact frozen-data and resumable Kaggle commands are in [`experiments/README.md`](experiments/README.md).

For the simplest current Kaggle path, upload [`dtp_fact_background_2026-10-25_v4.zip`](kaggle/output/dtp_fact_background_2026-10-25_v4.zip) and [`dtp_fact_pilot_ready_2026-10-25.zip`](kaggle/output/dtp_fact_pilot_ready_2026-10-25.zip) as private datasets, import [`01_run_pilot_experiments.ipynb`](kaggle/notebooks/01_run_pilot_experiments.ipynb), enable GPU and Internet, and run all cells. This produces pilot-only outputs; the full final-study notebooks remain gated on the final corpus. Full instructions are in [`kaggle/README.md`](kaggle/README.md).

Compile the LaTeX sources with the Codex built-in compiler or an installed IEEE-compatible LaTeX distribution. BibTeX is required for the final bibliography.

## Reproducibility rule

Notebooks may inspect outputs but must import scientific logic from `experiments/src`. Main tables and figures must be generated from preserved outputs rather than copied manually into the manuscript.

