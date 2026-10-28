# DACS-SCV experiment package

Reusable scientific logic lives in `src/trace_scv`; the Kaggle notebook only installs and orchestrates it.

## Implemented

- pinned SciVer download and deterministic source-paper-grouped development splits;
- frozen BGE-small semantic features for every evidence item and masked CLIP visual features for recoverable media;
- batched, deduplicated feature extraction reused across all seeds;
- claim-only, all-pool, relevance, flat-selector, and graph-selector verifiers;
- fixed top-one, top-two, and top-three citation outputs from one frozen graph ranking;
- a decoupled cardinality head trained after the matched graph verifier is frozen;
- exact safeguards for base-checkpoint hash, feature width, claim order, evidence ranking, and verdict-record identity;
- verdict, ranking, citation, cardinality, calibration, modality, runtime, and grouped-bootstrap outputs;
- input/code/configuration hashes, fingerprint-bound checkpoints, per-claim records, manifests, and artifact inventories;
- cardinality-head ablations that remove the valid-candidate or visual-candidate count;
- 34 unit/integration tests, including a complete clean smoke run and checkpoint-reuse rerun.

The former causal counterfactual objective remains in historical code for auditability but is not part of the frozen formal matrix or manuscript contribution.

## Local setup and checks

```powershell
cd D:\1.msc\ET2\experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[formal,analysis,dev]"
python -m pytest -q
python scripts/run_study.py --profile smoke --device cpu
```

Smoke data are synthetic and cannot support scientific claims. The corrected validation pilot is already frozen and audited; repeating it is unnecessary.

## Frozen formal run

The only remaining scientific execution is:

```powershell
python scripts/run_study.py --profile formal --download --device auto
```

Formal mode downloads the pinned SciVer release, uses the official test once, trains five predetermined seeds for the controlled learned variants, derives fixed-budget and DACS-SCV outputs from the matched graph ranking, and runs 10,000 source-paper-grouped bootstrap samples. MuSciClaims is disabled because the inspected main release lacks authoritative reuse-license metadata.

Use Kaggle rather than this CPU-only workstation for the formal run. Preserve the returned output archive unchanged and audit it before editing the results section.

## Output policy

`outputs/`, `data/`, `cache/`, and checkpoints are ignored by Git. A result claim is admissible only when linked to a saved prediction/comparison record, exact configuration, dataset revision, checkpoint fingerprint, and successful artifact audit. Published numbers from other papers are context, not entries in the paired formal comparison.
