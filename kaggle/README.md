# Kaggle final run

Use [`TRACE_SCV_Kaggle.ipynb`](TRACE_SCV_Kaggle.ipynb) as the single entry point.

1. Create a free Hugging Face **read-only** token.
2. In Kaggle, add that token as an enabled private notebook secret named `HF_TOKEN`. Never paste or print it in a cell.
3. Create a new notebook and enable **Internet** and a **GPU**.
4. Upload `dacs-scv-kaggle-v3.zip` as a private Kaggle dataset and attach it to the notebook. Remove all v1/v2 dataset attachments.
5. Import `TRACE_SCV_Kaggle.ipynb`.
6. Choose **Run All**. Do not change the automatic `formal` profile.
7. When the run finishes, download `dacs-scv-formal-outputs.zip` from `/kaggle/working` and return it unchanged for audit.

The notebook copies the read-only upload to `/kaggle/working`, installs it non-editably, clears cached modules, verifies DACS-SCV version `0.3.2`, checks public-data availability, runs the frozen five-seed matrix, and exports the complete audit bundle. It never runs editable installation against `/kaggle/input`.

Remove older TRACE-SCV/DACS-SCV ZIP attachments before running. The new notebook rejects an incomplete source tree or a version other than `0.3.2`. Its first cell must report `hf_authenticated: True`; otherwise it stops before touching the formal experiment. Its install cell must print a path ending in `experiments/src/trace_scv` after `DACS-SCV import verified:`. A Kaggle warning about `dopamine-rl` and `gym` is unrelated; this project imports neither package.

Version 3 retains the resumable authenticated download path from version 2 and fixes alpha-aware visual preprocessing. Transparent `LA`, `RGBA`, and palette images are composited onto white before perceptual hashing and CLIP input construction. The corrected full-corpus local audit has zero paper-group overlap, zero exact visual overlap, and zero exact gold-content overlap.

Profiles:

- `smoke`: synthetic software validation only;
- `pilot`: retained for reproducibility of development work, not needed again;
- `formal`: all eligible SciVer records, frozen BGE/CLIP features, five seeds, fixed-budget and adaptive citation outputs, and 10,000 group-bootstrap samples on the official test.

Do not change settings and rerun after seeing the official-test output. The returned archive must pass the independent audit before any value is copied into the paper.
