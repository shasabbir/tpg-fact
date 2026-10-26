# Kaggle quick start

Use [`01_public_data_preflight.ipynb`](01_public_data_preflight.ipynb) first.

1. Create a Kaggle notebook and enable Internet for the initial public-release check/download.
2. Add this repository as a Kaggle dataset, or upload a ZIP of the repository and extract it under `/kaggle/working`.
3. Import the notebook and run all cells.
4. Leave `DOWNLOAD_DATA = False` for the fast access/revision/software check.
5. Set it to `True` only when you are ready to download the pinned SciVer and MuSciClaims snapshots.

The notebook copies code from `/kaggle/input` to `/kaggle/working` and performs a normal install. It never uses `pip install -e` against the read-only input mount, which avoids the editable-build error from the former project.

This notebook is a preflight, not the final model experiment. The final training notebook will be added only after dataset schemas and usage terms pass milestone M1.
