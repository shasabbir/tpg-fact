# DACS-SCV

This repository contains the study **Decoupling Citation-Set Size from Verification: A Controlled Study of Multimodal Scientific Claims**.

The study asks whether a learned per-claim citation count can improve evidence-item citation quality while leaving the verifier and its evidence ranking exactly unchanged. It uses the public SciVer benchmark, frozen BGE/CLIP features, grouped development splits, five formal seeds, and source-paper-grouped uncertainty estimates.

The living source of truth is [`plan.md`](plan.md). It records the novelty boundary, public-data gates, hypotheses, retired ideas, pilot audits, frozen protocol, and next action.

## Layout

- `Base papers/`: the supplied M-RAV and MEVER anchor papers.
- `Reference_good_papers/`: the retained reference-PDF archive.
- `research/`: literature, dataset, reference, claim-evidence, and pilot-audit records.
- `experiments/`: reusable Python package, tests, scripts, configuration, and generated outputs.
- `kaggle/`: the single final-run notebook and concise instructions.
- `paper/`: the IEEE-style manuscript and its direct LaTeX support files only.

## Current status

The frozen formal study is complete:

- corrected SciVer parsing and cross-split data audits pass;
- the original counterfactual causal-grounding idea failed its prespecified pilot gate and was retired;
- 37 unit/integration tests and a complete resumable synthetic rerun pass;
- the five-seed official-test archive passes all artifact, code, record-level metric, grouped-bootstrap, verdict-identity, and ranking-identity audits;
- DACS-SCV improves citation-count accuracy but obtains 0.4702 citation macro-F1, below fixed top-two at 0.4810;
- the prespecified adaptive-superiority hypothesis is therefore rejected and retained as a controlled negative result;
- manuscript tables and the formal analysis figure are generated directly from the frozen archive.
- the final ten-page IEEE manuscript compiles without warnings, has passed page-level visual inspection, and is packaged in `dacs-scv-paper-source-2026-10-28.zip` for a clean rebuild.

The SciVer official test has been consumed. Do not tune a revised adaptive method on it and describe that revision as confirmatory.

## Local verification

Use the bundled or project Python environment from `D:\1.msc\ET2\experiments`:

```powershell
python -m pip install -e ".[formal,analysis,dev]"
python -m pytest -q
python scripts/run_study.py --profile smoke --device cpu
```

The smoke profile is synthetic software validation only.

Audit the returned formal output and regenerate paper assets with:

```powershell
python research/audit_formal_outputs.py <extracted-output-directory> --project-root . --output research/formal_results_audit_2026-10-28.json
python research/build_formal_paper_assets.py dacs-scv-formal-outputs.zip --paper-root paper --audit-output research/formal_paper_values_2026-10-28.json
```

## Frozen Kaggle runner

Create a free Hugging Face read-only token and add it to the Kaggle notebook as an enabled private secret named `HF_TOKEN`; never paste it into a cell. Upload `dacs-scv-kaggle-v3.zip` as a private Kaggle dataset, import [`kaggle/TRACE_SCV_Kaggle.ipynb`](kaggle/TRACE_SCV_Kaggle.ipynb), enable Internet and a GPU, and choose **Run All**. The notebook requires package version `0.3.2`, defaults to the frozen `formal` profile on Kaggle, composites transparent figures onto white before visual auditing and CLIP encoding, keeps downloads/features/checkpoints in resumable `/kaggle/working` locations, downloads only SciVer files referenced by the pinned 3,000-row release, and exports `dacs-scv-formal-outputs.zip`.

The retained runner reproduces the completed protocol. The audited formal archive has SHA-256 `06FE5026B871197B9768DDF31027A7FD47E3F050882084166374229A6B567F7C`.
