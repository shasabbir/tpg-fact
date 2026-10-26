from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "01_run_pilot_experiments.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


notebook = nbf.v4.new_notebook(
    cells=[
        markdown(
            """
            # Run the complete DTP-Fact pilot experiment

            In Kaggle, attach these two private datasets:

            1. `dtp_fact_background_2026-10-25_v4.zip`
            2. `dtp_fact_pilot_ready_2026-10-25.zip`

            Enable a GPU and Internet, then choose **Run all**. The notebook copies
            the read-only code to `/kaggle/working`, verifies the pilot gate, builds
            pinned BGE-M3 and SigLIP2 features, tunes every configured variant,
            runs five seeds, and validates the output matrix.

            These are pilot results for debugging and feasibility. They are not the
            final 2,000-case study and must not be reported as the paper's final results.
            """
        ),
        code(
            r'''
            from pathlib import Path
            import json, shutil, subprocess, sys, zipfile

            ON_KAGGLE = Path("/kaggle/input").exists()
            INPUT = Path("/kaggle/input") if ON_KAGGLE else Path.cwd()
            WORK = Path("/kaggle/working/dtp_fact_pilot_run") if ON_KAGGLE else Path.cwd() / "_pilot_run"
            WORK.mkdir(parents=True, exist_ok=True)

            def extract_first(pattern, destination):
                archives = sorted(INPUT.rglob(pattern))
                if not archives and not ON_KAGGLE:
                    archives = sorted(INPUT.parent.rglob(pattern))
                if not archives:
                    return False
                destination.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archives[0]) as archive:
                    archive.extractall(destination)
                return True

            background_candidates = list(INPUT.rglob("background/pyproject.toml"))
            if not background_candidates:
                extract_first("dtp_fact_background*.zip", WORK / "background_input") or extract_first(
                    "dtp_fact_kaggle_dataset*.zip", WORK / "background_input"
                )
                background_candidates = list((WORK / "background_input").rglob("background/pyproject.toml"))
            if not background_candidates:
                raise FileNotFoundError("Attach dtp_fact_background_2026-10-25_v4.zip")
            SOURCE_CODE = background_candidates[0].parent
            CODE_ROOT = WORK / "background"
            shutil.copytree(
                SOURCE_CODE,
                CODE_ROOT,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv", ".secrets"),
            )

            pilot_manifests = [
                path for path in INPUT.rglob("pilot_manifest.json")
                if json.loads(path.read_text(encoding="utf-8")).get("status")
                == "audited_feasibility_pilot_evidence_gate_passed"
            ]
            if not pilot_manifests:
                if not extract_first("dtp_fact_pilot_ready*.zip", WORK / "pilot_input"):
                    raise FileNotFoundError("Attach dtp_fact_pilot_ready_2026-10-25.zip")
                pilot_manifests = list((WORK / "pilot_input").rglob("pilot_manifest.json"))
            PILOT_ROOT = pilot_manifests[0].parent
            RESULTS_ROOT = Path("/kaggle/working/dtp_fact_pilot_results") if ON_KAGGLE else WORK / "results"
            sys.path.insert(0, str(CODE_ROOT / "src"))
            print(json.dumps({
                "code_root": str(CODE_ROOT),
                "pilot_root": str(PILOT_ROOT),
                "results_root": str(RESULTS_ROOT),
            }, indent=2))
            '''
        ),
        code(
            r'''
            install = [
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--no-build-isolation", "-e", f"{CODE_ROOT}[analysis,test,retrieval,ml]",
            ]
            subprocess.run(install, check=True)
            print("Background package installed from writable storage.")
            '''
        ),
        code(
            r'''
            manifest = json.loads((PILOT_ROOT / "pilot_manifest.json").read_text(encoding="utf-8"))
            frozen = json.loads((PILOT_ROOT / "frozen/dataset_manifest.json").read_text(encoding="utf-8"))
            checks = {
                "pilot_gate_passed": manifest["status"] == "audited_feasibility_pilot_evidence_gate_passed",
                "claim_count_200": manifest["claim_count"] == frozen["claim_count"] == 200,
                "all_claims_have_evidence": manifest["claims_with_time_valid_evidence"] == 200,
                "two_source_gate": manifest["independent_cluster_fraction"] >= 0.70,
                "frozen_dataset": frozen["status"] == "frozen_dataset",
                "no_group_leakage": frozen["group_leakage_checked"] == ["event_id", "image_phash"],
                "all_splits_present": all(frozen["splits"][name]["claims"] > 0 for name in ("train", "validation", "test")),
            }
            print(json.dumps(checks, indent=2))
            if not all(checks.values()):
                raise RuntimeError("The attached pilot package failed its frozen integrity gate")

            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Enable a Kaggle GPU in Notebook settings, then rerun all cells")
            print("GPU:", torch.cuda.get_device_name(0))
            '''
        ),
        code(
            r'''
            command = [
                sys.executable,
                "scripts/run_pilot_experiments.py",
                "--config", "configs/base.yaml",
                "--pilot-directory", str(PILOT_ROOT),
                "--output-directory", str(RESULTS_ROOT),
                "--device", "cuda",
                "--resume",
            ]
            print("Starting the full pilot matrix. This can take several hours.")
            subprocess.run(command, cwd=CODE_ROOT, check=True)
            '''
        ),
        code(
            r'''
            import yaml

            config = yaml.safe_load((CODE_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
            orchestration = json.loads((RESULTS_ROOT / "pilot_orchestration_manifest.json").read_text(encoding="utf-8"))
            matrix = json.loads((RESULTS_ROOT / "matrix/matrix_manifest.json").read_text(encoding="utf-8"))
            comparison = json.loads((RESULTS_ROOT / "comparison.json").read_text(encoding="utf-8"))
            expected_variants = set(config["training"]["variants"])
            successful_runs = [row for row in matrix["runs"] if row["return_code"] == 0]
            completed_variants = {
                row["variant"] for row in matrix["evaluations"]
                if row["return_code"] == 0 and row["variant"] != "majority"
            }
            checks = {
                "orchestration_complete": orchestration["status"] == "pilot_experiment_complete",
                "matrix_complete": matrix["status"] == "complete",
                "all_variants_evaluated": completed_variants == expected_variants,
                "five_seeds_per_variant": all(
                    sum(row["variant"] == variant and row["seed"] is not None for row in successful_runs) == 5
                    for variant in expected_variants
                ),
                "comparison_written": bool(comparison),
            }
            report = {
                "status": "pilot_model_matrix_validated_not_final_study" if all(checks.values()) else "pilot_model_matrix_invalid",
                "scientific_results_validated": False,
                "pilot_results_validated": all(checks.values()),
                "checks": checks,
                "warning": "Pilot-only evidence; do not copy these values into the final-paper Results section.",
            }
            report_path = RESULTS_ROOT / "pilot_result_validation.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, sort_keys=True))
            if not all(checks.values()):
                raise RuntimeError("The pilot result matrix is incomplete")

            archive_base = Path("/kaggle/working/dtp_fact_pilot_results") if ON_KAGGLE else WORK / "dtp_fact_pilot_results"
            shutil.make_archive(str(archive_base), "zip", RESULTS_ROOT.parent, RESULTS_ROOT.name)
            print("Saved:", str(archive_base) + ".zip")
            '''
        ),
    ],
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
)
notebook["nbformat"] = 4
notebook["nbformat_minor"] = 5
nbf.write(notebook, OUTPUT)
print(OUTPUT)
