# TPG-Fact experiments

The experiment code separates reusable logic from notebooks. The included smoke run uses synthetic records and validates the software path only.

## Commands

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_smoke_test.py --config configs/base.yaml
python scripts/validate_manifests.py --claims outputs/smoke/claims.jsonl
python -m unittest discover -s tests -v
```

Evaluate a frozen prediction artifact (repeat `--predictions` five times for the
main experiment and add `--expected-seeds 5`):

```powershell
python scripts/evaluate_predictions.py `
  --claims outputs/smoke/claims.jsonl `
  --predictions outputs/smoke/predictions.jsonl `
  --bootstrap-replicates 10000 `
  --output outputs/smoke/evaluation.json
```

`scripts/run_seed_matrix.py` executes any configuration-driven training command
once for each frozen seed. It passes `{seed}` and `{run_dir}` placeholders as
separate command arguments and writes a run manifest without invoking a shell.

Install the package for development if desired:

```powershell
python -m pip install -e ".[analysis,test]"
```

## Real-data workflow

1. Export eligible ClaimReview JSON-LD records into a project-controlled raw-data location.
2. Run `scripts/ingest_claimreview.py` to normalize explicit ratings and reject ambiguous records.
3. Discover evidence with the configured GDELT client and resolve archived copies through Common Crawl.
4. Freeze hashes and split manifests before fitting retrieval indexes or model parameters.
5. Run baselines and the proposed model through configuration-driven entry points.

The trainable graph module expects precomputed node features and typed graph
edges. Dense BGE-M3/SigLIP2 feature extraction and the full multi-GPU trainer
remain real-experiment work; the repository does not silently substitute the
synthetic smoke classifier for those components.

Raw copyrighted page content and images must remain outside the public release package unless their licenses permit redistribution.

