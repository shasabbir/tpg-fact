# DTP-Fact on Kaggle

## Run the completed 200-case pilot

Upload these two ZIP files as **private Kaggle Datasets**:

1. `output/dtp_fact_background_2026-10-25_v4.zip`
2. `output/dtp_fact_pilot_ready_2026-10-25.zip`

Import `notebooks/01_run_pilot_experiments.ipynb`, enable **GPU** and
**Internet**, and click **Run all**. No API key is needed. The notebook copies
the Python package out of Kaggle's read-only input mount before installing it,
so it avoids the earlier editable-install error.

The run performs pinned feature extraction, frozen retrieval, validation-only
tuning, all configured baselines and ablations, five seeds, 10,000 event-level
bootstrap comparisons, and output validation. When it finishes, download
`dtp_fact_pilot_results.zip` from the Kaggle Output pane.

For a fast data-only check, run `notebooks/00_validate_audited_pilot.ipynb`
with only the pilot ZIP attached. Its expected status is
`pilot_integrity_validated_evidence_gate_passed`: 200/200 claims have
decision-time-valid evidence and 141/200 have at least two source-domain
clusters.

## Scientific boundary

This package is a 200-case AI-assisted feasibility pilot. Its images, labels,
timestamps, and source-domain clusters pass the automated pilot gate, but the
dataset has not received independent human annotation and is not the planned
2,000-case final benchmark. Pilot model values are for debugging and method
selection; they must not be presented as final paper results.

The older `01_setup_and_data_validation.ipynb`,
`02_run_final_experiments.ipynb`, and `03_validate_study_results.ipynb` remain
for the eventual full benchmark. They are not needed for the pilot run above.

