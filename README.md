# TRACE-SCV

This repository develops **Evidence-Traceable Multimodal Claim Verification with Counterfactual Citation Grounding**. It is a complete reset of the former private-data temporal-provenance project.

The study asks whether a verifier's cited text, table, figure, or image actually supports its verdict. It uses public datasets only and evaluates verdict quality, citation attribution, counterfactual faithfulness, robustness, and compute cost.

The living source of truth is [`plan.md`](plan.md). It records the scope, novelty boundary, dataset gates, hypotheses, stop rules, milestones, and next action.

## Layout

- `Base papers/`: the two supplied anchor papers (M-RAV and MEVER).
- `Reference_good_papers/`: the earlier PDF archive, retained as source material.
- `research/`: base-paper dossier, novelty audit, dataset registry, literature matrix, and search log.
- `experiments/`: reusable Python package, tests, scripts, configurations, and analysis notebooks.
- `kaggle/`: the Kaggle entry notebook and short run instructions.
- `paper/`: IEEE-style LaTeX manuscript and direct supporting files only.

## Local smoke test

From `D:\1.msc\ET2\experiments`:

```powershell
python -m pip install -e ".[dev]"
python scripts/run_smoke.py --output outputs/smoke
python -m pytest -q
```

The smoke run uses synthetic evidence packets and validates software behavior only. It is not a scientific result.

## Public-data preflight

```powershell
python scripts/check_public_datasets.py --output outputs/data_preflight.json
```

For Kaggle, upload this repository or clone it in the notebook, enable Internet for the first run, and execute [`kaggle/01_public_data_preflight.ipynb`](kaggle/01_public_data_preflight.ipynb). The notebook installs the package non-editably to avoid Kaggle read-only dataset errors.

## Evidence policy

Result claims may enter the manuscript only after they are linked to a saved output, frozen configuration, dataset revision, and audit row. Published baseline values and locally reproduced values are never merged into the same statistical comparison.
