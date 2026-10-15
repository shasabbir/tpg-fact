# TPG-Fact experiments

The experiment package keeps reusable scientific logic in `src/` and uses notebooks only for audit and presentation. Run every command below from the `experiments` directory. The included smoke workflow uses synthetic records to test the software path; it is not a scientific experiment.

## 1. Create the environment

The code requires Python 3.10 or newer. In Windows PowerShell:

```powershell
cd D:\1.msc\ET2\experiments
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,test]"
$env:PYTHONPATH = "$PWD\src"
```

Install the optional PyTorch dependency when working on the trainable graph model:

```powershell
python -m pip install -e ".[ml]"
```

If PowerShell blocks activation, enable scripts for the current process only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Register the environment once for clean notebook execution:

```powershell
python -m ipykernel install --user --name tpg-fact --display-name "Python (tpg-fact)"
```

For a new terminal, reactivate `.venv` and set `PYTHONPATH` again before running scripts.

## 2. Run the complete software check

```powershell
python scripts/run_smoke_test.py --config configs/base.yaml
python scripts/validate_manifests.py --claims outputs/smoke/claims.jsonl
python scripts/evaluate_predictions.py `
  --claims outputs/smoke/claims.jsonl `
  --predictions outputs/smoke/predictions.jsonl `
  --bootstrap-replicates 10000 `
  --output outputs/smoke/evaluation.json
python -m pytest -q
python scripts/execute_notebooks.py
```

Successful execution produces JSON/JSONL artifacts under `outputs/smoke/`, four executed notebooks under `outputs/executed_notebooks/`, and a passing test summary. Delete or archive old output directories before a protocol-frozen run so artifacts from different configurations are not mixed.

## 3. Begin the real-data feasibility pilot

Place exported ClaimReview JSON-LD in a project-controlled location that is excluded from public release when licensing requires it. The following example normalizes only explicit ratings and writes a rejection report next to the accepted manifest:

```powershell
python scripts/ingest_claimreview.py `
  --input data/raw/claimreview.json `
  --output data/processed/claims.jsonl

python scripts/validate_manifests.py `
  --claims data/processed/claims.jsonl
```

Collect GDELT candidate metadata for a pilot query:

```powershell
python scripts/collect_open_evidence.py `
  --query 'United States election image claim' `
  --max-records 100 `
  --output data/interim/gdelt-election.json
```

Archive resolution, image recovery, timestamp adjudication, and claim-to-evidence linking must be completed and audited before these candidates become benchmark evidence.

## 4. Run seeded training commands

`run_seed_matrix.py` invokes a command once for each seed in `configs/base.yaml`. Replace the example trainer and its arguments with the frozen training entry point:

```powershell
python scripts/run_seed_matrix.py `
  --config configs/base.yaml `
  --output-directory outputs/main_model `
  -- python path/to/train.py --config configs/base.yaml --seed "{seed}" --output-dir "{run_dir}"
```

Each seed receives its own output directory and log. The runner writes `manifest.json` incrementally and stops on the first failed run unless `--keep-going` is supplied.

## 5. Evaluate five frozen prediction files

Pass one prediction file per seed. The evaluator checks claim IDs and probability normalization before computing aggregate metrics and event-level bootstrap intervals.

```powershell
python scripts/evaluate_predictions.py `
  --claims data/frozen/test_claims.jsonl `
  --predictions outputs/main_model/seed_11/predictions.jsonl `
  --predictions outputs/main_model/seed_23/predictions.jsonl `
  --predictions outputs/main_model/seed_37/predictions.jsonl `
  --predictions outputs/main_model/seed_53/predictions.jsonl `
  --predictions outputs/main_model/seed_71/predictions.jsonl `
  --expected-seeds 5 `
  --bootstrap-replicates 10000 `
  --target-coverage 0.80 `
  --output outputs/main_model/evaluation.json
```

## 6. Compile the manuscript

Compile from the `paper` directory so LaTeX resolves the section, figure, and table paths:

```powershell
cd D:\1.msc\ET2\paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf paper.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf supplement.tex
```

## Current implementation boundary

The repository implements the schemas, temporal checks, lexical retrieval, rank fusion, provenance graph utilities, calibration, evaluation, tests, and a synthetic integration path. Dense BGE-M3/SigLIP2 extraction, audited Common Crawl recovery, and the full batched multi-GPU trainer still require implementation before the main experiments can run. Do not use the synthetic smoke classifier or its outputs as scientific evidence.

Raw copyrighted page content and images must remain outside the public release package unless their licenses permit redistribution.

