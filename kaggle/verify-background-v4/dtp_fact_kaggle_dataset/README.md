# Kaggle files: separate notebooks and one background dataset

Upload `output/dtp_fact_kaggle_dataset_2026-10-25_v3.zip` once as a **private
Kaggle Dataset**. Import the notebooks separately; they are intentionally not
inside that ZIP.

For the newly collected 200-case pilot, upload
`output/dtp_fact_pilot_data_2026-10-25.zip` as a **private Kaggle Dataset** and
run `notebooks/00_validate_audited_pilot.ipynb`. It needs no GPU and no API key.
The expected status is `pilot_integrity_validated_evidence_gate_failed`: all
artifact-integrity checks pass, but only 108/200 claims have independently
archived evidence and 32/200 have two source clusters. This is a real failed
feasibility result, not a software error.

For the code and eventual final experiment, run in this order:

1. `notebooks/01_setup_and_data_validation.ipynb`
   - installs the background Python modules;
   - runs the packaged tests and the synthetic integration diagnostic;
   - validates the 32,898-candidate snapshot and 1,200-row screening queue;
   - does **not** call those candidates scientific data.
2. `notebooks/02_run_final_experiments.ipynb`
   - also requires a private audited `dtp_fact_final_data_<date>.zip`;
   - requires a Kaggle GPU;
   - runs the resumable final pipeline and exports results/tables ZIPs;
   - blocks safely when audited final data are absent.
3. `notebooks/03_validate_study_results.ipynb`
   - attach the exact final-data input and saved output from notebook 02;
   - independently checks hashes, leakage controls, CUDA execution,
     validation-only tuning, five seeds, retrieval, interventions, statistical
     comparisons, and generated tables;
   - displays metrics only after every required check passes.

The background ZIP contains `configs/`, `src/`, `scripts/`, `tests/`, and the
key-free candidate snapshot. It excludes `.secrets`, caches, local virtual
environments, paper files, and unlicensed final images/articles.

Each notebook treats `/kaggle/input` as read-only: it copies only the small
background package to `/kaggle/working/dtp_fact_runtime/background` before the
editable installation. The candidate data remain mounted read-only.

## Important current boundary

The supplied candidate snapshot is sufficient to test setup and begin
screening. The separate 200-case pilot ZIP is audited by an AI-assisted visual
workflow but fails the evidence-coverage gate; neither artifact is the final
benchmark. Consequently, notebooks
02 and 03 have been locally executed only through their honest blocked path.
They will run/validate the scientific study after the audited final-data ZIP is
created using `FINAL_DATA_LAYOUT.md`.

