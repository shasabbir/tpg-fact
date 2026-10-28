# Multimodal evidence-selection experiment package

Reusable scientific logic lives in `src/trace_scv`; notebooks only install and orchestrate it. The historical SciVer implementation remains reproducible, while active development targets CEFES on MOCHEG.

## Active CEFES/MOCHEG implementation

- pinned train, development, shared-corpus, and sealed-test transport identities;
- a development-only downloader that never fetches formal-test annotations;
- normalization of both aggregated-train and evidence-row development JSON schemas;
- count, label, qrel-topic, split-overlap, corpus, file-hash, and test-seal checks;
- reconstruction of every released train/development sentence candidate from
  `Corpus3.csv` with the official NLTK preprocessing and duplicate policy;
- bounded deterministic image acquisition with SHA-256 and Pillow readability audits;
- exact expected set-F1 decoding under the declared independent-Bernoulli approximation;
- deterministic binary temperature scaling with Brier, NLL, and ECE diagnostics;
- empty-set policy, maximum-budget handling, stable probability ordering, and smaller-set tie-breaking;
- brute-force equivalence tests for the expected-F1 dynamic program;
- 76 passing project tests after the Phase II additions.

The 400-claim development packet gate passes. All 378,864 distinct train/development sentence-qrel IDs resolve to public corpus text; the pilot contains 15,823 candidates, 15,289 available candidates, and no unavailable gold candidate. The candidate feature store, fitted selector, matched verifier, and development study have not yet been completed, so no positive scientific result exists.

Prepare or re-audit the currently downloaded metadata with:

```powershell
cd experiments
python -m pip install --no-deps -e .
python scripts/prepare_mocheg.py
```

Reconstruct and audit all released train/development sentence candidates with:

```powershell
python scripts/prepare_mocheg.py --build-sentence-text
```

The command pins NLTK 3.9.2 through the package data extra, records the Punkt
resource hash, writes `sentence_text.jsonl`, and fails if even one qrel sentence
cannot be resolved.

Download a bounded, resumable image subset for a development smoke/pilot with:

```powershell
python scripts/prepare_mocheg.py --download-pilot-media `
  --max-train-media-claims 200 `
  --max-validation-media-claims 200 `
  --seed 13
```

This command does not access test annotations. A free read-only `HF_TOKEN` is optional and only improves rate limits.

Build the matched pilot packets after the media and sentence gates pass:

```powershell
python scripts/prepare_mocheg.py --build-pilot-packets
```

## RTX 4090 workstation

The primary execution profile is `configs/workstation_4090.yaml`. Transfer the
source bundle, extract it onto a drive with at least 300 GiB free, then run:

```powershell
cd experiments
powershell -ExecutionPolicy Bypass -File scripts/setup_workstation.ps1
```

If ordinary `pip install torch torchvision` does not expose CUDA, obtain the
current Windows/Pip/CUDA index URL from the official PyTorch selector and pass
it as `-TorchIndexUrl`. The setup stops unless the machine has at least 48 GiB
RAM, 300 GiB free storage, a CUDA GPU with at least 20 GiB VRAM, and CUDA-enabled
PyTorch. Public MOCHEG data should be downloaded directly on the workstation;
do not copy the current partial cache.

After setup passes, execute the complete real development pilot with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_workstation_pilot.ps1
```

The CPU hash backend has also completed end to end as a software-only smoke
test. It cannot satisfy the development gate or support a paper claim.

## Retained SciVer implementation

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
- checked unit/integration coverage, including a complete clean smoke run and checkpoint-reuse rerun.

The former causal counterfactual objective remains in historical code for auditability but is not part of the frozen formal matrix or manuscript contribution.

## Local setup and checks

```powershell
cd experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[formal,analysis,dev]"
python -m pytest -q
python scripts/run_study.py --profile smoke --device cpu
```

Smoke data are synthetic and cannot support scientific claims. The corrected validation pilot is already frozen and audited; repeating it is unnecessary.

## Historical SciVer formal runner

The retained reproduction command is:

```powershell
python scripts/run_study.py --profile formal --download --device auto
```

Formal mode downloads the pinned SciVer release, uses the official test once, trains five predetermined seeds for the controlled learned variants, derives fixed-budget and DACS-SCV outputs from the matched graph ranking, and runs 10,000 source-paper-grouped bootstrap samples. MuSciClaims is disabled because the inspected main release lacks authoritative reuse-license metadata.

The SciVer official test is already consumed and the result is negative. This command is retained for reproduction only; it is not a route to a fresh confirmatory claim.

## Output policy

`outputs/`, `data/`, `cache/`, and checkpoints are ignored by Git. A result claim is admissible only when linked to a saved prediction/comparison record, exact configuration, dataset revision, checkpoint fingerprint, and successful artifact audit. Published numbers from other papers are context, not entries in the paired formal comparison.
