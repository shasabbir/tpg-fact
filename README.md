# Multimodal Fact-Checking Study Workspace

This workspace contains two scientifically separate studies:

1. the completed SciVer study, **Decoupling Citation-Set Size from Verification: A Controlled Study of Multimodal Scientific Claims**; and
2. the active Phase II protocol, **Optimizing the Evidence Set: Calibrated Adaptive Selection for Multimodal Fact-Checking**.

The completed study asked whether a learned per-claim citation count could improve evidence-item citation quality while leaving the verifier and its evidence ranking unchanged. The active study instead tests CEFES, a calibrated selector that chooses a variable evidence set by maximizing exact expected set-F1 on the public MOCHEG benchmark.

The living source of truth is [`plan.md`](plan.md). It records the novelty boundary, public-data gates, hypotheses, retired ideas, pilot audits, frozen protocol, and next action.

## Layout

- `Base papers/`: the supplied M-RAV and MEVER anchor papers.
- `Reference_good_papers/`: the retained reference-PDF archive.
- `research/`: literature, dataset, reference, claim-evidence, and pilot-audit records.
- `experiments/`: reusable Python package, tests, scripts, configuration, and generated outputs.
- `kaggle/`: the single final-run notebook and concise instructions.
- paper/: the active CEFES/MOCHEG IEEE pre-results manuscript and its direct LaTeX support files only.

## Current status

The frozen SciVer study is complete:

- corrected SciVer parsing and cross-split data audits pass;
- the original counterfactual causal-grounding idea failed its prespecified pilot gate and was retired;
- 37 unit/integration tests and a complete resumable synthetic rerun pass;
- the five-seed official-test archive passes all artifact, code, record-level metric, grouped-bootstrap, verdict-identity, and ranking-identity audits;
- DACS-SCV improves citation-count accuracy but obtains 0.4702 citation macro-F1, below fixed top-two at 0.4810;
- the prespecified adaptive-superiority hypothesis is therefore rejected and retained as a controlled negative result;
- the old manuscript and generated SciVer result assets remain recoverable from Git history and the frozen paper archive.

The SciVer official test has been consumed. Do not tune a revised adaptive method on it and describe that revision as confirmatory.

Phase II implementation is ready for its pinned semantic development pilot but has not produced scientific results:

- MOCHEG is the selected primary confirmatory benchmark;
- its official test partition must remain sealed throughout development;
- CEFES and all baselines will share candidates, ranking features, and verifier so that selection policy is the controlled variable;
- a three-seed development pilot must pass every gate in `plan.md` before the five-seed formal test run is unlocked;
- the active paper is a conventional pre-results CEFES manuscript; it records
  implemented methods and checked readiness facts while explicitly marking GPU,
  verifier, and formal results as pending.

The MOCHEG metadata, sentence-text, media, and matched-packet gates now pass on pinned train/development mirrors. Both observed JSON schemas are normalized, all 378,864 referenced sentence IDs resolve, and the deterministic 400-claim packet contains 15,289 available candidates with no unavailable gold item. The exact CEFES decoder, temperature calibration, bounded candidate pool, three-seed scorer, and paired bootstrap pipeline pass 76 tests. A CPU hash run completed only as software validation and failed the scientific gate by construction.

The official MOCHEG archive has been located, but its single 69.2 GiB gzip stream is not required for routine development. Pinned public transport mirrors reproduce the official train/development claim and evidence counts, and the 91,822-row shared text corpus is present locally with a recorded hash. The next operation is the pinned BGE-M3/SigLIP2 pilot on the RTX 4090 workstation. Formal-test annotations remain absent until the complete development gate and protocol freeze pass.

## Next run: RTX 4090 workstation

Read plan.md Section 0, then use
[RUN_ON_4090_WORKSTATION.md](RUN_ON_4090_WORKSTATION.md). The complete transfer
artifact is dist/cefes-study-handoff.bundle after the handoff commit. The
source-only alternative is dist/cefes-workstation-source.zip; it excludes public
data, outputs, virtual environments, credentials, and old archives. The
workstation downloads pinned public data directly and runs the real 400-claim
development pilot with one PowerShell command.

## Retained SciVer verification

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
