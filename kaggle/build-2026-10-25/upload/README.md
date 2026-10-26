# DTP-Fact Kaggle package

Use `DTP_Fact_Kaggle.ipynb` in Kaggle and attach
`dtp_fact_kaggle_upload_bundle_2026-10-25.zip` as a **private** Kaggle dataset.
Then select **Run all**. The notebook discovers the attached ZIPs, installs the
package, runs the tests and synthetic diagnostic, and displays the real
candidate-screening snapshot.

The included data are a key-free ClaimReview candidate snapshot, not an audited
training benchmark. The notebook therefore does not start scientific training
from those candidates. It starts the full CUDA experiment automatically only
when a separate `dtp_fact_final_data*.zip` is attached and all required audited
files in `FINAL_DATA_LAYOUT.md` are present.

Kaggle settings:

1. Accelerator: GPU.
2. Internet: on, so the pinned Hugging Face revisions can be downloaded.
3. Add the upload bundle as a private dataset.
4. Import the notebook and select Run all.

The final run writes to `/kaggle/working/final_run` and generated manuscript
tables to `/kaggle/working/generated_tables`. Save or download those directories
before the Kaggle session ends.

