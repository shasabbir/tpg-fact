# DTP-Fact experiments

The experiment package keeps reusable scientific logic in `src/` and uses notebooks only for audit and presentation. Run every command below from the `experiments` directory. The included smoke workflow uses synthetic records to test the software path; it is not a scientific experiment.

## Current frozen pilot (2026-10-25)

The redesigned evidence-first pilot is frozen at
`data/private/pilot_evidence_first_final_2026-10-25/frozen`: 200 claims, 682
time-valid evidence records, and 141 claims with at least two source-domain
clusters. The automated pilot gate passes, but the data remain AI-assisted and
are not the final benchmark.

The easiest GPU execution path is documented in `../kaggle/README.md`. It uses
the v4 background ZIP, the gate-passing pilot ZIP, and
`01_run_pilot_experiments.ipynb`. The equivalent command on a configured CUDA
machine is:

```powershell
python scripts/run_pilot_experiments.py `
  --config configs/base.yaml `
  --pilot-directory <extracted-dtp_fact_pilot_data> `
  --output-directory outputs/pilot_cuda_2026-10-25 `
  --device cuda `
  --resume
```

## 1. Create the environment

The code requires Python 3.10 or newer. In Windows PowerShell:

```powershell
cd D:\1.msc\ET2\experiments
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,test,collection]"
$env:PYTHONPATH = "$PWD\src"
```

Install the optional PyTorch dependency when working on the trainable graph model:

```powershell
python -m pip install -e ".[ml]"
```

Install the retrieval extra for BGE-M3, the BGE reranker, and SigLIP2:

```powershell
python -m pip install -e ".[retrieval]"
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
python scripts/audit_feasibility.py `
  --claims outputs/smoke/claims.jsonl `
  --evidence outputs/smoke/evidence.jsonl `
  --output outputs/smoke/feasibility.json
python scripts/build_intervention_sets.py `
  --claims outputs/smoke/claims.jsonl `
  --evidence outputs/smoke/evidence.jsonl `
  --output-directory outputs/smoke/interventions
python -m pytest -q
python scripts/execute_notebooks.py
```

Successful execution produces JSON/JSONL artifacts under `outputs/smoke/`, four executed notebooks under `outputs/executed_notebooks/`, and a passing test summary. Delete or archive old output directories before a protocol-frozen run so artifacts from different configurations are not mixed.

## 3. Begin the real-data feasibility pilot

### 3.1 Store the Google API key locally

The collector reads `GOOGLE_FACTCHECK_API_KEY` or the ignored local file
`.secrets/google_factcheck_api_key.txt`. Never paste the key into a command,
notebook, CSV, tracked configuration, or chat message. To use the local file:

```powershell
New-Item -ItemType Directory -Force .secrets
notepad .secrets\google_factcheck_api_key.txt
```

Paste only the key into Notepad and save the file. The `.secrets/` directory is
excluded by `.gitignore`.

Enable the **Fact Check Tools API** in the same Google Cloud project that owns
the key, wait a few minutes for propagation, and run the safe preflight:

```powershell
python scripts/check_factcheck_api.py
```

The preflight never prints the key. A `PERMISSION_DENIED`/HTTP 403 response
stating that the API is disabled means the key file was found but the service
still needs to be enabled for its project.

### 3.2 Collect fact-check candidates

The command below paginates the official Google Fact Check Tools API, preserves
each raw response, deduplicates review URLs, filters the configured year range,
and creates a human-audit CSV. It does not guess the target image or US relevance.

```powershell
python scripts/collect_factcheck_candidates.py `
  --output-directory data/private/pilot_claims `
  --publisher-site politifact.com `
  --publisher-site factcheck.org `
  --publisher-site reuters.com `
  --publisher-site apnews.com `
  --publisher-site factcheck.afp.com `
  --publisher-site snopes.com `
  --publisher-site leadstories.com `
  --language-code en `
  --start-year 2020 `
  --end-year 2026
```

Review `data/private/pilot_claims_2026-10-24/screening_queue.csv`. For every row, set
`decision` to `keep` or `reject`. Rejections require `exclusion_reason`. Kept
rows require confirmation of US relevance, multimodality, the fact-checked
image, rating, dates, label, and event group. The prefilled `suggested_label`
is only a suggestion and must not replace the audit.

For the frozen 2026-10-24 snapshot, begin with the deterministic balanced queue
instead of opening all 32,898 rows:

```powershell
python scripts/build_screening_queue.py `
  --input data/private/pilot_claims_2026-10-24/candidate_review.csv `
  --output data/private/pilot_claims_2026-10-24/screening_queue.csv `
  --manifest data/private/pilot_claims_2026-10-24/screening_queue_manifest.json `
  --target-size 1200 `
  --seed 2026
```

The checked queue already contains 1,200 pending rows: 400 per suggested label,
with 652 train-period, 233 validation-period, and 315 test-period candidates.
These counts describe screening priorities, not accepted benchmark cases.

```powershell
python scripts/validate_candidate_review.py `
  --input data/private/pilot_claims_2026-10-24/screening_queue.csv
```

If review was performed in the smaller `screening_queue.csv`, merge only its
audited fields into a new full-size CSV. The command verifies candidate identity,
refuses unknown/duplicate IDs, keeps both source files unchanged, and records
SHA-256 hashes:

```powershell
python scripts/merge_screening_review.py `
  --master data/private/pilot_claims_2026-10-24/candidate_review.csv `
  --queue data/private/pilot_claims_2026-10-24/screening_queue.csv `
  --output data/private/pilot_claims_2026-10-24/candidate_review_merged.csv `
  --manifest data/private/pilot_claims_2026-10-24/screening_merge_manifest.json

python scripts/validate_candidate_review.py `
  --input data/private/pilot_claims_2026-10-24/candidate_review_merged.csv
```

### 3.3 Download only verified target images

After `claim_image_url` and `target_image_verified=yes` are audited, download
the images into the private data directory and calculate exact SHA-256 and
perceptual hashes:

```powershell
python scripts/fetch_reviewed_images.py `
  --review-csv data/private/pilot_claims_2026-10-24/candidate_review_merged.csv `
  --output-directory data/private/pilot_images `
  --manifest data/private/pilot_images/image_manifest.csv

python scripts/finalize_pilot_claims.py `
  --review-csv data/private/pilot_claims_2026-10-24/candidate_review_merged.csv `
  --image-manifest data/private/pilot_images/image_manifest.csv `
  --output data/processed/pilot_claims.jsonl
```

Image files remain private unless their source licenses permit redistribution.

### 3.4 Collect and audit evidence candidates

The GDELT command uses each kept row's `evidence_query`; when blank, it creates
a transparent keyword query from the claim text. Returned pages are candidates,
not accepted evidence. The command preserves raw responses and failures.

```powershell
python scripts/collect_pilot_evidence.py `
  --review-csv data/private/pilot_claims_2026-10-24/candidate_review_merged.csv `
  --output-directory data/private/pilot_evidence `
  --exclude-domain politifact.com `
  --exclude-domain factcheck.org `
  --exclude-domain factcheck.afp.com `
  --exclude-domain snopes.com `
  --exclude-domain leadstories.com
```

Review `data/private/pilot_evidence/evidence_review.csv`. Accepted rows need an
article publication time, relevance decision, relation to the claim, and a
reviewed source-dependence cluster. GDELT's observation time is not silently
treated as the article's publication time. Reuters and AP host both ordinary
reporting and fact-check articles, so their domains are not excluded wholesale;
their fact-check URLs must still be rejected during review.

After publication times are populated, verify historical availability in
Common Crawl:

```powershell
python scripts/verify_archive_observations.py `
  --input data/private/pilot_evidence/evidence_review.csv `
  --output data/private/pilot_evidence/archive_audit.csv
```

The index checker records every queried Common Crawl collection, including
negative results. Respect the configured delay and do not overload the service.

After relevance, publication time, and source clusters have been explicitly
confirmed in the evidence CSV, finalize the evidence and attach its identifiers
to a new claim manifest:

```powershell
python scripts/finalize_pilot_evidence.py `
  --claims data/processed/pilot_claims.jsonl `
  --review-csv data/private/pilot_evidence/evidence_review.csv `
  --archive-audit data/private/pilot_evidence/archive_audit.csv `
  --evidence-output data/processed/pilot_evidence.jsonl `
  --claims-output data/processed/pilot_claims_with_evidence.jsonl `
  --exclude-domain politifact.com `
  --exclude-domain factcheck.org `
  --exclude-domain factcheck.afp.com `
  --exclude-domain snopes.com `
  --exclude-domain leadstories.com
```

The finalizer refuses evidence when its relevance, publication time, or source
cluster has not been confirmed, when a fact-check domain slipped through, or
when the publication/archive chronology fails the primary temporal rule.

### 3.5 Run the pilot-specific audit

The pilot report is deliberately separate from the final 2,000-case benchmark
gate. It measures screening yield, projected collection burden, field
completeness, and independent-evidence coverage without pretending that the
final corpus already exists.

```powershell
python scripts/audit_pilot.py `
  --review-csv data/private/pilot_claims_2026-10-24/candidate_review_merged.csv `
  --claims data/processed/pilot_claims_with_evidence.jsonl `
  --evidence data/processed/pilot_evidence.jsonl `
  --output outputs/pilot/pilot_assessment.json
```

### Legacy JSON-LD intake

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

For the primary analysis, an evidence item must have both a publication date and an independent archive or crawl observation no later than the claim's decision cutoff. Mark dates as `confirmed`, `observed`, or `uncertain`; uncertain items belong only in the named sensitivity analysis. Do not infer a gold sufficiency label from the verdict string alone.

After a full corpus—not merely the 200-case pilot—has source-cluster identifiers,
run the preregistered final benchmark gate:

```powershell
python scripts/audit_feasibility.py `
  --claims data/processed/claims.jsonl `
  --evidence data/processed/evidence.jsonl `
  --config configs/base.yaml `
  --output outputs/pilot/feasibility.json
```

Create the deterministic duplicate, post-cutoff, and earliest-source test sets before model selection:

```powershell
python scripts/build_intervention_sets.py `
  --claims data/processed/claims.jsonl `
  --evidence data/processed/evidence.jsonl `
  --output-directory outputs/pilot/interventions `
  --copies 1 5 10
```

Encode a license-compliant local evidence-text manifest with BGE-M3. The input JSONL must contain `evidence_id` and either `text` or `title`:

```powershell
python scripts/embed_evidence_text.py `
  --input data/processed/evidence_text.jsonl `
  --output data/features/evidence_bge_m3.npz `
  --model BAAI/bge-m3 `
  --batch-size 32
```

Exact public model revisions are frozen in `configs/base.yaml` from the official
Hugging Face model records retrieved on 2026-10-24. Recheck availability before
the frozen run, but do not silently advance a revision after results exist.

## 4. Freeze the audited data

Do not freeze `candidate_review.csv` or the 1,200-row screening queue directly.
They are acquisition aids, not experimental data. Freeze only the finalized claim
and evidence manifests after the 200-case pilot gate passes:

```powershell
python scripts/freeze_dataset.py `
  --claims data/processed/claims_with_evidence.jsonl `
  --evidence data/processed/evidence.jsonl `
  --output-directory data/frozen/final_2026-10-24
```

The command rejects duplicate identifiers, missing evidence links, empty temporal
partitions, and event/image leakage. It writes SHA-256 hashes for the immutable
full and split manifests. Keep `dataset_manifest.json` with every run.

## 5. Run the final experiment matrix

### 5.1 Preflight

The API key is not needed on Kaggle because the raw candidate snapshot is already
frozen. Use its key-free collection manifest:

```powershell
python scripts/preflight_full_run.py `
  --collection-manifest data/private/pilot_claims_2026-10-24/collection_manifest.json `
  --conflict-manifest data/private/conflict_donors.jsonl
```

The command exits with code 2 until the 200-case pilot manifests, pilot gate, and
CUDA runtime are present. This is intentional. `ready` means the run prerequisites
exist; it does not mean that the study's claims are already supported.

### 5.2 One resumable final command

After copying the frozen data and verified image manifests to a CUDA machine, run:

```powershell
python scripts/run_full_study.py `
  --config configs/base.yaml `
  --frozen-data-directory data/frozen/final_2026-10-24 `
  --claim-image-manifest data/private/pilot_images/image_manifest.csv `
  --evidence-image-manifest data/private/pilot_evidence_images/image_manifest.csv `
  --evidence-text data/private/evidence_text.jsonl `
  --conflict-manifest data/private/conflict_donors.jsonl `
  --output-directory outputs/final_2026-10-24 `
  --paper-table-directory ..\paper\tables `
  --device cuda `
  --resume
```

Omit `--evidence-image-manifest` if no audited evidence images are available, and
omit `--evidence-text` if only evidence titles can legally be retained. Do not use
`--allow-cpu` for a scientific run; that switch is only for tiny diagnostics.
`conflict_donors.jsonl` must contain one audited row for every test claim, with
`claim_id`, `donor_evidence_id`, and the JSON boolean
`"semantic_conflict_verified": true`. The exporter refuses to create paper tables
from automatic opposite-label pairings.

The orchestrator performs these stages in order:

1. builds frozen BGE-M3 and SigLIP2 feature archives at the pinned revisions;
2. runs BM25, BGE-M3, and available SigLIP2 rankings, reciprocal-rank fusion,
   pinned BGE reranking, decision-time filtering, and top-20 selection;
3. executes 20 validation-only trials for every configured trainable method;
4. freezes each method's selected hyperparameters;
5. trains all configured baselines and ablations with seeds 11, 23, 37, 53, and 71;
6. evaluates duplicate-copy, post-cutoff-verdict, earliest-source, dependent and
   independent conflict, caption-swap, and historical-image-reuse interventions
   with the frozen full, vanilla-RAG, time-filtered-RAG, and provenance-pooling checkpoints;
7. evaluates the untouched test split with calibration and selective-risk metrics;
8. computes paired event-level bootstrap comparisons and Holm adjustments; and
9. writes dataset, retrieval, main-result, ablation, and intervention LaTeX tables from checked artifacts.

Every subprocess has a log. `orchestration_manifest.json` is updated after each
stage, and `--resume` skips completed feature and selection stages. Preserve the
entire output directory; it contains the audit trail for the paper.

### 5.3 Focused commands for debugging

Use these only when a stage needs to be inspected separately:

```powershell
python scripts/build_feature_store.py --config configs/base.yaml --claims data/frozen/final_2026-10-24/claims.jsonl --evidence data/frozen/final_2026-10-24/evidence.jsonl --claim-image-manifest data/private/pilot_images/image_manifest.csv --output-directory outputs/final_2026-10-24/features --device cuda

python scripts/retrieve_evidence.py --config configs/base.yaml --claims data/frozen/final_2026-10-24/claims.jsonl --evidence data/frozen/final_2026-10-24/evidence.jsonl --text-features outputs/final_2026-10-24/features/text_embeddings.npz --image-features outputs/final_2026-10-24/features/image_embeddings.npz --output-directory outputs/final_2026-10-24/retrieval

python scripts/tune.py --config configs/base.yaml --claims outputs/final_2026-10-24/retrieval/retrieved_claims.jsonl --evidence data/frozen/final_2026-10-24/evidence.jsonl --text-features outputs/final_2026-10-24/features/text_embeddings.npz --image-features outputs/final_2026-10-24/features/image_embeddings.npz --variant full --output-directory outputs/final_2026-10-24/selection/full --device cuda

python scripts/run_final_matrix.py --config configs/base.yaml --claims outputs/final_2026-10-24/retrieval/retrieved_claims.jsonl --evidence data/frozen/final_2026-10-24/evidence.jsonl --text-features outputs/final_2026-10-24/features/text_embeddings.npz --image-features outputs/final_2026-10-24/features/image_embeddings.npz --selection-root outputs/final_2026-10-24/selection --output-directory outputs/final_2026-10-24/matrix --device cuda

python scripts/evaluate_interventions.py --config configs/base.yaml --claims outputs/final_2026-10-24/retrieval/retrieved_claims.jsonl --evidence data/frozen/final_2026-10-24/evidence.jsonl --text-features outputs/final_2026-10-24/features/text_embeddings.npz --image-features outputs/final_2026-10-24/features/image_embeddings.npz --matrix-root outputs/final_2026-10-24/matrix --conflict-manifest data/private/conflict_donors.jsonl --output-directory outputs/final_2026-10-24/interventions --device cuda
```

`train.py --selection-only` never constructs test examples. Calibration,
temperature, and the 80% coverage threshold are fitted from validation data. The
matrix runner will refuse to start a method without a completed selection record.

## 6. Kaggle GPU instructions

The ready-to-import clients are split into three notebooks under
`../kaggle/notebooks/`. Attach `dtp_fact_kaggle_dataset_2026-10-25.zip` as a
private Kaggle Dataset and run them in numeric order. The first validates setup
and candidate integrity, the second runs the final CUDA matrix when audited data
exist, and the third independently refuses or validates the resulting study
artifacts. See `../kaggle/README.md` for the shortest instructions.

This is a Python package with notebook clients, not a notebook-only study. Upload
the repository and the audited frozen inputs as **private Kaggle datasets**. Do
not upload `.secrets/`, raw copyrighted pages, or unlicensed images. In a Kaggle
GPU notebook, the cells may call the command-line package as follows:

```python
%cd /kaggle/working/ET2/experiments
!python -m pip install -e ".[analysis,test,retrieval,ml]"
%env PYTHONPATH=/kaggle/working/ET2/experiments/src
!python -m pytest -q
!python scripts/preflight_full_run.py --collection-manifest /kaggle/input/us-tpmfc-private/collection_manifest.json --conflict-manifest /kaggle/input/us-tpmfc-private/conflict_donors.jsonl
!python scripts/run_full_study.py --config configs/base.yaml --frozen-data-directory /kaggle/input/us-tpmfc-private/frozen --claim-image-manifest /kaggle/input/us-tpmfc-private/image_manifest.csv --conflict-manifest /kaggle/input/us-tpmfc-private/conflict_donors.jsonl --output-directory /kaggle/working/final_run --paper-table-directory /kaggle/working/generated_tables --device cuda --resume
```

Enable a GPU accelerator before running the last command. If the session stops,
save `/kaggle/working/final_run` as a private dataset or download it, restore it in
the next session, and rerun with `--resume`. The four `.ipynb` files are for audit,
error analysis, and presentation; no training or metric definition exists only in
a notebook.

After a final run, the strict command-line equivalent of notebook 03 is:

```powershell
python scripts/validate_study_results.py `
  --config configs/base.yaml `
  --frozen-data-directory data/frozen/final_2026-10-24 `
  --results-directory outputs/final_2026-10-24 `
  --paper-table-directory ..\paper\tables `
  --output outputs/final_2026-10-24/study_validation_report.json
```

It returns code 2 unless every scientific-artifact check passes. Use
`--allow-incomplete` only to write a diagnostic blocked report; it never changes
`scientific_claims_allowed` to true.

## 7. Verify and compile the manuscript

Compile from the `paper` directory so LaTeX resolves the section, figure, and table paths:

```powershell
cd D:\1.msc\ET2\paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf paper.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf supplement.tex
```

## Current implementation boundary

The reusable trainer, graph and matched feature models, validation-only tuner,
five-seed matrix runner, calibration/abstention path, paired event bootstrap,
Holm adjustment, feature-store builder, dataset freezer, generated LaTeX tables,
and resumable final orchestrator are implemented and mechanically tested. The
frozen candidate snapshot and balanced 1,200-row screening queue also exist.

The scientific run is **not yet ready** because the 200 retained claims and their
decision-time evidence have not been manually audited, the pilot gate has not
passed, and this local host has no CUDA GPU. No real-data metric has been measured.
The CPU smoke outputs and tiny training-cycle test are software diagnostics only.
Do not put them in the paper as results.

Raw copyrighted page content and images must remain outside the public release package unless their licenses permit redistribution.

