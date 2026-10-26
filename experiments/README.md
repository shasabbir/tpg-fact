# Experiment package

This package contains reusable code for TRACE-SCV. Notebooks orchestrate and inspect runs; scientific logic remains in `src/trace_scv`.

## What is implemented now

- immutable claim/evidence/prediction schemas;
- citation and verdict metrics;
- deterministic evidence interventions;
- official public-release accessibility checks;
- public Hugging Face snapshot download helper;
- synthetic end-to-end smoke run;
- unit tests for metrics and interventions;
- Kaggle-safe install path (regular wheel install, not editable install from `/kaggle/input`).

The deep multimodal model is intentionally not represented as finished. It is milestone M3 in `../plan.md` and must be implemented after dataset schema and license gates pass.

## Local setup

```powershell
cd D:\1.msc\ET2\experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
```

Run checks:

```powershell
python scripts/check_public_datasets.py --output outputs/data_preflight.json
python scripts/run_smoke.py --output outputs/smoke
python -m pytest -q
```

Download a public Hugging Face snapshot after preflight:

```powershell
python scripts/download_public_data.py --dataset sciver --root data/public
python scripts/download_public_data.py --dataset musciclaims --root data/public
```

The downloader pins revisions from `src/trace_scv/data/registry.py`. Update a revision only through a documented study-plan decision.

## Output rule

`outputs/`, `data/`, `cache/`, and `checkpoints/` are ignored. Preserve formal run outputs separately with their config, environment, code commit, and dataset manifest before writing results into the paper.
