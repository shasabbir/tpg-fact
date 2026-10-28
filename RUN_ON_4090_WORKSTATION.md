# Run the CEFES development pilot on the RTX 4090 PC

This workflow uses only MOCHEG train and development annotations. The 400-claim
semantic packet and shared verifier have been checked; the full-corpus
development comparison and clean reproduction are the current stage. The
official test remains locked.

## 1. Move the complete project

Clone the repository into any folder you choose. That cloned folder is the
study root. It is identified by containing plan.md, paper/, experiments/, and
research/. No drive letter or folder name is required by the code.

Preferred offline transfer: copy dist/cefes-study-handoff.bundle to the new PC
and run from the parent directory where you want the study folder created:

    git clone X:\Transfer\cefes-study-handoff.bundle my-study
    cd my-study
    git status
    git log -1 --oneline

Replace the bundle path and my-study with your actual choices. If the handoff
commit has been pushed to GitHub, an ordinary clone is also valid:

    git clone https://github.com/shasabbir/tpg-fact.git my-study
    cd my-study

Alternative: extract dist/cefes-workstation-source.zip into any folder. That
extracted folder becomes the study root but does not contain Git history. Do
not copy the old experiments/data or experiments/outputs directories; the
runner downloads pinned public files and creates fresh outputs.

Open the cloned study root in ChatGPT/Codex and say: "Continue this study. Treat
this repository folder as the study root and read plan.md first." The complete
continuation instructions are in plan.md Section 0.7.

## 2. Create and check the environment

Open PowerShell anywhere inside the cloned repository. Normalize to its root,
then enter experiments/:

    Set-Location -LiteralPath (git rev-parse --show-toplevel)
    Set-Location experiments
    powershell -ExecutionPolicy Bypass -File scripts\setup_workstation.ps1

If the final hardware check reports pytorch_cuda_available as false, use the
current official PyTorch Windows/Pip/CUDA index URL:

    powershell -ExecutionPolicy Bypass -File scripts\setup_workstation.ps1 -TorchIndexUrl "OFFICIAL_PYTORCH_INDEX_URL"

The setup runs the complete test suite and stops unless RAM, storage, CUDA, and
VRAM requirements pass. No paid API is required. A read-only Hugging Face token
is optional for download rate limits and must remain outside the repository.

## 3. Run the real 400-claim development pilot

    powershell -ExecutionPolicy Bypass -File scripts\run_workstation_pilot.ps1

The command is resumable. It downloads and audits public train/development data,
reconstructs sentence candidates, downloads the bounded image packet, builds
pinned BGE-M3 and SigLIP2 features on the GPU, trains three selector seeds,
calibrates them, runs 10,000 paired bootstrap samples, and writes:

    experiments\outputs\mocheg\selector_pilot\summary.json

## 4. Audit the packet result

Keep the complete experiments/outputs/mocheg/selector_pilot folder and the three
pilot_features files. Audit scientific_feature_backend, encoder revisions,
feature hashes, bounded candidate counts, every seed, calibration values, and
claim records.

Do not run or download formal-test annotations. The checked Trial 2 selector
and verifier outputs are preserved separately; they do not unlock the test.

## 5. Run the full-corpus development comparison

The portable downloader is resumable and never lists the test mirror:

    .venv-workstation\Scripts\python.exe scripts\run_mocheg_full_corpus.py --download-media --max-download-workers 64
    .venv-workstation\Scripts\python.exe scripts\run_mocheg_full_corpus.py --build-features
    .venv-workstation\Scripts\python.exe scripts\audit_mocheg_full_corpus.py
    .venv-workstation\Scripts\python.exe scripts\run_mocheg_selector_pilot.py --features outputs\mocheg\full_corpus_features.npz --output-dir outputs\mocheg\full_corpus_selector_trial_002 --calibration-safety-guard
    .venv-workstation\Scripts\python.exe scripts\audit_mocheg_selector_pilot.py --features outputs\mocheg\full_corpus_features.npz --output-dir outputs\mocheg\full_corpus_selector_trial_002 --audit-output outputs\mocheg\full_corpus_selector_trial_002\audit.json
    .venv-workstation\Scripts\python.exe scripts\run_mocheg_matched_verifier.py --features outputs\mocheg\full_corpus_features.npz --selector-dir outputs\mocheg\full_corpus_selector_trial_002 --output-dir outputs\mocheg\full_corpus_verifier_trial_002
    .venv-workstation\Scripts\python.exe scripts\audit_mocheg_matched_verifier.py --output-dir outputs\mocheg\full_corpus_verifier_trial_002

If the full-corpus selector gate fails, the verifier command may add
`--diagnostic-after-failed-selector-gate`; that diagnostic cannot rescue the
failed condition. A second end-to-end run writes under
`outputs\mocheg\clean_rerun`, after which
`compare_mocheg_clean_reproduction.py` and
`evaluate_mocheg_development_gate.py` apply every condition in plan.md Section
13. Only a passing aggregate gate can authorize a later protocol freeze. It
does not itself download or evaluate formal-test annotations.
