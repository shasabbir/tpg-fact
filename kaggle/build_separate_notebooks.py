"""Generate the three thin Kaggle client notebooks.

Scientific logic remains in the packaged Python modules.  The notebooks only
discover inputs, run commands, and present checked artifacts.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks"
OUTPUT.mkdir(parents=True, exist_ok=True)


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


BOOTSTRAP = r'''
from pathlib import Path
import json, os, shutil, subprocess, sys, zipfile

ON_KAGGLE = Path("/kaggle/input").exists()
INPUT_ROOT = Path("/kaggle/input") if ON_KAGGLE else Path.cwd()
WORK_ROOT = Path("/kaggle/working/dtp_fact_runtime") if ON_KAGGLE else Path.cwd() / "_dtp_fact_runtime"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

def find_background():
    candidates = [
        path for path in sorted(INPUT_ROOT.rglob("background/pyproject.toml"))
        if (path.parent.parent / "data").is_dir()
    ]
    candidates += [
        path for path in sorted(WORK_ROOT.rglob("background/pyproject.toml"))
        if (path.parent.parent / "data").is_dir()
    ]
    if not candidates:
        archives = sorted(INPUT_ROOT.rglob("dtp_fact_kaggle_dataset*.zip"))
        if not archives:
            raise FileNotFoundError(
                "Attach dtp_fact_kaggle_dataset_2026-10-25_v3.zip as a Kaggle Dataset."
            )
        extract_root = WORK_ROOT / "dataset"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(extract_root)
        candidates = [
            path for path in sorted(extract_root.rglob("background/pyproject.toml"))
            if (path.parent.parent / "data").is_dir()
        ]
    if not candidates:
        raise FileNotFoundError("The attached dataset has no background/pyproject.toml")
    return candidates[0].parent, candidates[0].parent.parent

SOURCE_CODE_ROOT, BUNDLE_ROOT = find_background()
DATA_ROOT = BUNDLE_ROOT / "data"
CODE_ROOT = WORK_ROOT / "background"
if SOURCE_CODE_ROOT.resolve() != CODE_ROOT.resolve():
    shutil.copytree(
        SOURCE_CODE_ROOT,
        CODE_ROOT,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv", ".secrets"),
    )
sys.path.insert(0, str(CODE_ROOT / "src"))

def install_background():
    install_command = [
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-build-isolation",
        "-e", f"{CODE_ROOT}[analysis,test,retrieval,ml]",
    ]
    subprocess.run(install_command, check=True)
    print("Background package installed from", CODE_ROOT)

print(json.dumps({
    "on_kaggle": ON_KAGGLE,
    "source_code_root": str(SOURCE_CODE_ROOT),
    "code_root": str(CODE_ROOT),
    "data_root": str(DATA_ROOT),
    "work_root": str(WORK_ROOT),
}, indent=2))
'''


INSTALL = r'''
install_background()
'''


nb1 = nbf.v4.new_notebook(
    cells=[
        markdown(r'''
        # 01 — Setup and data validation

        Run this notebook first with `dtp_fact_kaggle_dataset_2026-10-25_v3.zip`
        attached as a private Kaggle Dataset. It installs the background package,
        runs every software test, executes the synthetic integration diagnostic,
        and checks the collected candidate snapshot.

        **Boundary:** the included candidates are screening inputs, not audited gold
        data. Passing this notebook validates software and collection integrity only;
        it does not validate the paper's scientific findings.
        '''),
        code(BOOTSTRAP),
        code(INSTALL),
        code(r'''
        test_run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=CODE_ROOT,
            text=True,
            capture_output=True,
        )
        print(test_run.stdout)
        if test_run.stderr:
            print(test_run.stderr)
        if test_run.returncode != 0:
            raise RuntimeError("The background package tests failed.")

        smoke_run = subprocess.run(
            [sys.executable, "scripts/run_smoke_test.py", "--config", "configs/base.yaml"],
            cwd=CODE_ROOT,
            text=True,
            capture_output=True,
        )
        print(smoke_run.stdout)
        if smoke_run.stderr:
            print(smoke_run.stderr)
        if smoke_run.returncode != 0:
            raise RuntimeError("The synthetic integration diagnostic failed.")

        smoke_metrics = json.loads((CODE_ROOT / "outputs/smoke/metrics.json").read_text(encoding="utf-8"))
        assert smoke_metrics["status"] == "synthetic_software_diagnostic_only"
        print("Software checks passed; smoke values remain non-scientific diagnostics.")
        '''),
        code(r'''
        import csv
        from collections import Counter

        snapshots = sorted(DATA_ROOT.rglob("collection_manifest.json"))
        if not snapshots:
            raise FileNotFoundError("Candidate collection_manifest.json is absent.")
        candidate_root = snapshots[0].parent
        collection = json.loads((candidate_root / "collection_manifest.json").read_text(encoding="utf-8"))
        queue_manifest = json.loads((candidate_root / "screening_queue_manifest.json").read_text(encoding="utf-8"))

        with (candidate_root / "candidates.jsonl").open("r", encoding="utf-8") as stream:
            candidate_count = sum(bool(line.strip()) for line in stream)
        with (candidate_root / "screening_queue.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            queue_rows = list(csv.DictReader(stream))
        queue_labels = Counter(row["suggested_label"] for row in queue_rows)
        raw_pages = len(list((candidate_root / "raw").glob("*.json")))

        checks = {
            "candidate_count_32898": candidate_count == 32898,
            "manifest_count_matches": candidate_count == collection["candidate_count_after_deduplication"],
            "raw_pages_440": raw_pages == 440,
            "queue_size_1200": len(queue_rows) == 1200 == queue_manifest["actual_size"],
            "queue_balanced": queue_labels == Counter({
                "supported": 400, "refuted": 400, "insufficient_evidence": 400
            }),
            "all_rows_pending": all(row["decision"] == "pending" for row in queue_rows),
            "api_key_not_recorded": collection.get("api_key_recorded") is False,
            "no_secret_directory": not any(path.name == ".secrets" for path in BUNDLE_ROOT.rglob(".secrets")),
        }
        report = {
            "status": "candidate_snapshot_validated_for_screening_only" if all(checks.values()) else "candidate_snapshot_invalid",
            "scientific_results_validated": False,
            "candidate_count": candidate_count,
            "raw_response_pages": raw_pages,
            "screening_queue_rows": len(queue_rows),
            "suggested_label_counts": dict(queue_labels),
            "checks": checks,
            "interpretation": "Automatic suggested labels are not gold labels; do not train the final study on this snapshot.",
        }
        report_path = WORK_ROOT / "01_data_validation_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        if not all(checks.values()):
            raise RuntimeError("Candidate snapshot validation failed.")
        '''),
        markdown(r'''
        If the final line says `candidate_snapshot_validated_for_screening_only`,
        the code and acquisition snapshot are intact. Continue to notebook 02 only
        after an audited final-data dataset is attached. Notebook 02 will block
        rather than silently treat suggested labels as scientific ground truth.
        '''),
    ],
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
)


FINAL_DISCOVERY = r'''
def extract_named_archives(pattern, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(INPUT_ROOT.rglob(pattern)):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination / archive_path.stem)

def find_final_data_root():
    search_roots = [INPUT_ROOT, WORK_ROOT]
    manifests = []
    for root in search_roots:
        manifests.extend(root.rglob("frozen/dataset_manifest.json"))
    if not manifests:
        extract_named_archives("dtp_fact_final_data*.zip", WORK_ROOT / "final_inputs")
        manifests = list((WORK_ROOT / "final_inputs").rglob("frozen/dataset_manifest.json"))
    return manifests[0].parent.parent if manifests else None

FINAL_DATA_ROOT = find_final_data_root()
print("Final data root:", FINAL_DATA_ROOT or "NOT FOUND")
'''


nb2 = nbf.v4.new_notebook(
    cells=[
        markdown(r'''
        # 02 — Run the final experiments

        Attach two private Kaggle Datasets:

        1. `dtp_fact_kaggle_dataset_2026-10-25_v3.zip` (background Python package), and
        2. an audited `dtp_fact_final_data_<date>.zip` following the included layout.

        Enable a Kaggle GPU before running. This notebook invokes the reusable
        command-line pipeline; no trainer or metric is implemented only here.
        It writes resumable outputs to `/kaggle/working/final_run`.
        '''),
        code(BOOTSTRAP),
        code(FINAL_DISCOVERY),
        code(r'''
        readiness = {"status": "checking", "blockers": []}
        if FINAL_DATA_ROOT is None:
            readiness["blockers"].append("Audited dtp_fact_final_data_<date>.zip is not attached.")
        else:
            required = [
                FINAL_DATA_ROOT / "frozen/dataset_manifest.json",
                FINAL_DATA_ROOT / "frozen/claims.jsonl",
                FINAL_DATA_ROOT / "frozen/evidence.jsonl",
                FINAL_DATA_ROOT / "image_manifest.csv",
                FINAL_DATA_ROOT / "conflict_donors.jsonl",
            ]
            readiness["blockers"].extend(f"Missing {path}" for path in required if not path.is_file())
            manifest_path = FINAL_DATA_ROOT / "frozen/dataset_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "frozen_dataset":
                    readiness["blockers"].append("dataset_manifest.json is not a frozen_dataset manifest.")
        try:
            import torch
            if not torch.cuda.is_available():
                readiness["blockers"].append("No CUDA GPU is enabled in this Kaggle session.")
            readiness["cuda_devices"] = torch.cuda.device_count()
            readiness["cuda_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            readiness["blockers"].append("PyTorch is unavailable.")
        readiness["status"] = "ready_for_final_run" if not readiness["blockers"] else "blocked"
        readiness_path = WORK_ROOT / "02_final_run_readiness.json"
        readiness_path.write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        '''),
        code(r'''
        if readiness["status"] == "ready_for_final_run":
            install_background()
        else:
            print("Installation and training skipped because final-run prerequisites are incomplete.")
        '''),
        code(r'''
        RESULTS_ROOT = Path("/kaggle/working/final_run") if ON_KAGGLE else WORK_ROOT / "final_run"
        TABLE_ROOT = Path("/kaggle/working/generated_tables") if ON_KAGGLE else WORK_ROOT / "generated_tables"
        validation_report = None

        if readiness["status"] == "ready_for_final_run":
            command = [
                sys.executable, "scripts/run_full_study.py",
                "--config", "configs/base.yaml",
                "--frozen-data-directory", str(FINAL_DATA_ROOT / "frozen"),
                "--claim-image-manifest", str(FINAL_DATA_ROOT / "image_manifest.csv"),
                "--conflict-manifest", str(FINAL_DATA_ROOT / "conflict_donors.jsonl"),
                "--output-directory", str(RESULTS_ROOT),
                "--paper-table-directory", str(TABLE_ROOT),
                "--device", "cuda",
                "--resume",
            ]
            optional = {
                "--evidence-image-manifest": FINAL_DATA_ROOT / "evidence_image_manifest.csv",
                "--evidence-text": FINAL_DATA_ROOT / "evidence_text.jsonl",
            }
            for flag, path in optional.items():
                if path.is_file():
                    command.extend([flag, str(path)])
            print("Running:", " ".join(command))
            subprocess.run(command, cwd=CODE_ROOT, check=True)

            from evaluation.study_validation import validate_study_artifacts
            validation_report = validate_study_artifacts(
                CODE_ROOT / "configs/base.yaml",
                FINAL_DATA_ROOT / "frozen",
                RESULTS_ROOT,
                TABLE_ROOT,
            )
            validation_path = RESULTS_ROOT / "study_validation_report.json"
            validation_path.write_text(json.dumps(validation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({
                "status": validation_report["status"],
                "scientific_claims_allowed": validation_report["scientific_claims_allowed"],
                "blockers": validation_report["blockers"],
            }, indent=2))
            shutil.make_archive(str(Path("/kaggle/working/dtp_fact_final_results") if ON_KAGGLE else WORK_ROOT / "dtp_fact_final_results"), "zip", RESULTS_ROOT.parent, RESULTS_ROOT.name)
            shutil.make_archive(str(Path("/kaggle/working/dtp_fact_generated_tables") if ON_KAGGLE else WORK_ROOT / "dtp_fact_generated_tables"), "zip", TABLE_ROOT.parent, TABLE_ROOT.name)
        else:
            blocked = {
                "status": "blocked_missing_audited_data_or_gpu",
                "scientific_claims_allowed": False,
                "blockers": readiness["blockers"],
            }
            (WORK_ROOT / "02_experiment_status.json").write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
            print("No scientific experiment was run.")
        '''),
        markdown(r'''
        A completed run creates `dtp_fact_final_results.zip` and
        `dtp_fact_generated_tables.zip` in Kaggle's Output pane. Save a Kaggle
        version so notebook 03 can attach those outputs. If this notebook reports
        `blocked`, do not interpret that as a negative study result—the final data
        or GPU prerequisite is simply absent.
        '''),
    ],
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
)


nb3 = nbf.v4.new_notebook(
    cells=[
        markdown(r'''
        # 03 — Independently validate the study results

        Attach the background dataset, the exact audited final-data dataset used
        in notebook 02, and the saved output of notebook 02. This notebook does
        not accept screenshots or copied metric tables. It validates the frozen
        hashes, temporal grouping, CUDA run, validation-only tuning, all configured
        five-seed runs, retrieval, interventions, bootstrap comparisons, and
        generated paper tables before displaying any result.
        '''),
        code(BOOTSTRAP),
        code(FINAL_DISCOVERY),
        code(r'''
        def extract_result_archives():
            destination = WORK_ROOT / "result_inputs"
            destination.mkdir(parents=True, exist_ok=True)
            patterns = ("dtp_fact_final_results*.zip", "dtp_fact_generated_tables*.zip")
            for pattern in patterns:
                for archive_path in sorted(INPUT_ROOT.rglob(pattern)):
                    with zipfile.ZipFile(archive_path) as archive:
                        archive.extractall(destination / archive_path.stem)

        extract_result_archives()

        result_candidates = []
        table_candidates = []
        for root in (INPUT_ROOT, WORK_ROOT):
            result_candidates.extend(path.parent for path in root.rglob("orchestration_manifest.json"))
            table_candidates.extend(path.parent for path in root.rglob("generated_main_results.tex"))
        RESULTS_ROOT = result_candidates[0] if result_candidates else None
        TABLE_ROOT = table_candidates[0] if table_candidates else None
        print(json.dumps({
            "final_data_root": str(FINAL_DATA_ROOT) if FINAL_DATA_ROOT else None,
            "results_root": str(RESULTS_ROOT) if RESULTS_ROOT else None,
            "table_root": str(TABLE_ROOT) if TABLE_ROOT else None,
        }, indent=2))
        '''),
        code(r'''
        if FINAL_DATA_ROOT is not None and RESULTS_ROOT is not None and TABLE_ROOT is not None:
            install_background()
            from evaluation.study_validation import validate_study_artifacts
            report = validate_study_artifacts(
                CODE_ROOT / "configs/base.yaml",
                FINAL_DATA_ROOT / "frozen",
                RESULTS_ROOT,
                TABLE_ROOT,
            )
        else:
            missing = []
            if FINAL_DATA_ROOT is None:
                missing.append("audited final-data artifact")
            if RESULTS_ROOT is None:
                missing.append("final experiment output")
            if TABLE_ROOT is None:
                missing.append("generated paper tables")
            report = {
                "status": "blocked_missing_final_results",
                "scientific_claims_allowed": False,
                "blockers": missing,
                "checks": [],
            }

        output_path = Path("/kaggle/working/study_validation_report.json") if ON_KAGGLE else WORK_ROOT / "study_validation_report.json"
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "scientific_claims_allowed": report["scientific_claims_allowed"],
            "blockers": report["blockers"],
            "report": str(output_path),
        }, indent=2))
        '''),
        code(r'''
        if not report["scientific_claims_allowed"]:
            print("RESULT DISPLAY BLOCKED: the study artifacts have not passed strict validation.")
        else:
            import pandas as pd
            summary = report["validated_result_summary"]
            rows = []
            for item in summary["evaluations"]:
                row = {"variant": item["variant"], "seed_count": item["seed_count"]}
                for metric in ("macro_f1", "balanced_accuracy", "brier", "ece", "aurc"):
                    row[f"{metric}_mean"] = item["aggregate"][metric]["mean"]
                    row[f"{metric}_std"] = item["aggregate"][metric]["sample_std"]
                rows.append(row)
            frame = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
            print(f"Validated test set: {summary['test_claims']} claims, {summary['test_events']} events")
            display(frame)
            display(pd.DataFrame(summary["paired_event_bootstrap"]))
        '''),
        markdown(r'''
        Only `scientific_artifacts_validated` authorizes copying the displayed
        values into the manuscript. Any blocked or invalid status names the exact
        missing checks in `study_validation_report.json`.
        '''),
    ],
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
)


for name, notebook in (
    ("01_setup_and_data_validation.ipynb", nb1),
    ("02_run_final_experiments.ipynb", nb2),
    ("03_validate_study_results.ipynb", nb3),
):
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    nbf.write(notebook, OUTPUT / name)
    print(OUTPUT / name)
