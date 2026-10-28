# Formal-run incident audit — 2026-10-28

## Evidence inspected

- Executed notebook: `kaggle/TRACE_SCV_Kaggle_after_run.ipynb`
- Frozen package requested by that notebook: DACS-SCV `0.3.0`
- Dataset revision: SciVer `5695ac384d247ae5df2a7b73568dc63d189296b0`
- Second-run evidence: the user-supplied release-gate traceback and the complete local reconstruction of the same pinned release

## Finding

Cells 1--4 completed code discovery, installation, import verification, and public-release preflight. Cell 5 failed during the initial Hugging Face snapshot download with HTTP 429 at the SciVer Xet read-token endpoint. Later result and export cells have no execution count or output. Therefore the run did not reach SciVer preparation, feature construction, training, checkpoint selection, official-test inference, or metric aggregation. The failed attempt contributes no scientific result and does not consume the frozen official test.

## Corrective implementation

Runner v2 / package `0.3.1`:

1. requires a Kaggle secret named `HF_TOKEN` before formal execution;
2. disables the Xet download route before importing `huggingface_hub`;
3. downloads only the 1,191 paper records and 1,399 images referenced by the official 1,000-row development and 2,000-row test files, rather than every repository file;
4. limits Hub concurrency and applies bounded retry/backoff only to HTTP 429 and transient server/network failures;
5. keeps downloads, processed data, feature stores, checkpoints, and outputs in stable `/kaggle/working` roots, outside the source checkout that is replaced during setup.

These are operational changes. They do not alter the pinned data revision, data rows, split logic, model, seeds, hyperparameters, comparison family, metrics, or formal success rule.

## Second attempt: false perceptual-overlap gate

The authenticated v2 attempt completed the SciVer download and deterministic packet construction, then stopped at the release gate before feature extraction or training. A complete local reconstruction of the same pinned 3,000-row release reproduced the failure:

- source-paper group overlap: zero;
- exact content overlaps involving gold evidence: zero;
- exact raw-media SHA-256 overlap: zero;
- exact 64-bit perceptual-hash overlap: one fingerprint across each split pair.

All three reported overlaps were the same fingerprint, `0000000000000000`, assigned to three distinct gold figures from papers `2411.18243v1`, `2410.22785v1`, and `2409.03165v1`. The files have different raw hashes, dimensions, captions, and scientific content. Each PNG uses luminance-plus-alpha (`LA`) storage. The previous audit converted `LA` directly to grayscale, discarding alpha and exposing the zero-valued hidden luminance of transparent pixels; this collapsed each figure to an all-zero thumbnail. The formal CLIP loader used the analogous direct RGB conversion and was therefore also affected.

Package `0.3.2` composites `LA`, `RGBA`, and palette-transparency images onto white before dHash calculation and CLIP encoding. The unchanged full-corpus release gate then passes:

- train/validation/test counts: 804/196/2,000;
- group intersections: 0/0/0;
- exact raw-media SHA-256 overlaps: 0/0/0;
- exact corrected perceptual-hash overlaps: 0/0/0;
- gold-involving exact content overlaps: 0/0/0.

The correction changes visual preprocessing but does not change the dataset rows, split assignment, labels, model architecture, seeds, comparison family, metrics, or success rule. Because both failed attempts stopped before feature extraction, no model output or official-test prediction from an earlier preprocessing version exists.

The final v3 upload archive was then tested from an isolated temporary workspace containing only a renamed copy of the ZIP in a nested read-only-style input directory. The notebook discovered the archive, installed package `0.3.2`, completed all 11 smoke variants, and produced a 56-file output inventory. The frozen upload is `dacs-scv-kaggle-v3.zip`, 98,208 bytes, with SHA-256 `8B3A3B36F7803EC13E0A91ED669663D7279FAF87C34D149D5226F48F8CBE8EE8`.
