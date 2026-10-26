from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "00_validate_audited_pilot.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


notebook = nbf.v4.new_notebook(
    cells=[
        markdown(
            """
            # Validate the collected DTP-Fact pilot

            Attach `dtp_fact_pilot_ready_2026-10-25.zip` as a private Kaggle Dataset,
            then run all cells. This notebook checks the 200 claims, image hashes,
            evidence chronology, references, and preregistered feasibility gate.

            Passing this pilot gate authorizes scaling and pilot experiments. It is not
            the same as validating the final 2,000-case study or its scientific claims.
            """
        ),
        code(
            r'''
            from pathlib import Path
            from collections import Counter
            from datetime import datetime
            import csv, hashlib, json, zipfile

            ON_KAGGLE = Path("/kaggle/input").exists()
            INPUT = Path("/kaggle/input") if ON_KAGGLE else Path.cwd()
            WORK = Path("/kaggle/working/dtp_fact_pilot_ready") if ON_KAGGLE else Path.cwd() / "_pilot_validation_ready"
            WORK.mkdir(parents=True, exist_ok=True)

            search_roots = [INPUT] if ON_KAGGLE else [INPUT, INPUT.parent]
            manifest_paths = [
                path
                for search_root in search_roots
                for path in search_root.rglob("dtp_fact_pilot_data/pilot_manifest.json")
            ]
            roots = [
                path for path in manifest_paths
                if json.loads(path.read_text(encoding="utf-8")).get("status")
                == "audited_feasibility_pilot_evidence_gate_passed"
            ]
            if not roots:
                archives = sorted(
                    path
                    for search_root in search_roots
                    for path in search_root.rglob("dtp_fact_pilot_ready*.zip")
                )
                if not archives:
                    archives = sorted(
                        path
                        for search_root in search_roots
                        for path in search_root.rglob("dtp_fact_pilot_data*.zip")
                    )
                if not archives:
                    raise FileNotFoundError("Attach dtp_fact_pilot_ready_2026-10-25.zip")
                with zipfile.ZipFile(archives[0]) as archive:
                    archive.extractall(WORK)
                roots = list(WORK.rglob("dtp_fact_pilot_data/pilot_manifest.json"))
            if not roots:
                roots = manifest_paths
            if not roots:
                raise FileNotFoundError("No pilot_manifest.json was found")
            ROOT = roots[0].parent
            print("Pilot root:", ROOT)
            '''
        ),
        code(
            r'''
            def load_jsonl(path):
                with path.open("r", encoding="utf-8") as stream:
                    return [json.loads(line) for line in stream if line.strip()]

            def sha256(path):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            manifest = json.loads((ROOT / "pilot_manifest.json").read_text(encoding="utf-8"))
            assessment = json.loads((ROOT / "pilot_assessment.json").read_text(encoding="utf-8"))
            claims_path = ROOT / "pilot_claims_with_evidence.jsonl"
            evidence_path = ROOT / "pilot_evidence.jsonl"
            claims = load_jsonl(claims_path)
            evidence = load_jsonl(evidence_path)
            evidence_by_id = {row["evidence_id"]: row for row in evidence}
            with (ROOT / "image_manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                images = list(csv.DictReader(stream))

            chronology_ok = True
            referenced_ids = set()
            two_cluster_claims = 0
            for claim in claims:
                cutoff = datetime.fromisoformat(claim["decision_cutoff"].replace("Z", "+00:00"))
                clusters = set()
                for evidence_id in claim["evidence_ids"]:
                    referenced_ids.add(evidence_id)
                    row = evidence_by_id[evidence_id]
                    published = datetime.fromisoformat(row["published_at"].replace("Z", "+00:00"))
                    archived = datetime.fromisoformat(row["archived_at"].replace("Z", "+00:00"))
                    chronology_ok &= published <= archived <= cutoff
                    clusters.add(row["source_cluster_id"])
                two_cluster_claims += len(clusters) >= 2

            image_hashes_ok = all(
                (ROOT / row["local_path"]).is_file()
                and sha256(ROOT / row["local_path"]) == row["image_sha256"]
                for row in images
            )
            checks = {
                "claim_count_200": len(claims) == manifest["claim_count"] == 200,
                "evidence_count_matches": len(evidence) == manifest["evidence_count"],
                "image_count_200": len(images) == manifest["image_count"] == 200,
                "claim_hash_matches": sha256(claims_path) == manifest["claims_sha256"],
                "evidence_hash_matches": sha256(evidence_path) == manifest["evidence_sha256"],
                "all_image_hashes_match": image_hashes_ok,
                "all_evidence_references_resolve": referenced_ids <= set(evidence_by_id),
                "all_evidence_is_time_valid": chronology_ok,
                "two_cluster_count_matches": two_cluster_claims == manifest["claims_with_two_source_clusters"],
                "pilot_assessment_matches": assessment["evidence"]["record_count"] == len(evidence),
            }
            integrity_ok = all(checks.values())
            gate_passed = (
                assessment["evidence"]["claims_with_time_valid_evidence"] == len(claims)
                and assessment["evidence"]["independent_cluster_fraction"] >= 0.70
            )
            report = {
                "status": (
                    "pilot_integrity_validated_evidence_gate_passed" if integrity_ok and gate_passed
                    else "pilot_integrity_validated_evidence_gate_failed" if integrity_ok
                    else "pilot_integrity_invalid"
                ),
                "scientific_results_validated": False,
                "label_counts": dict(Counter(row["normalized_label"] for row in claims)),
                "claims_with_evidence": assessment["evidence"]["claims_with_time_valid_evidence"],
                "claims_with_two_clusters": two_cluster_claims,
                "independent_cluster_fraction": assessment["evidence"]["independent_cluster_fraction"],
                "checks": checks,
                "next_action": (
                    "Proceed to pilot experiments and scale-up preparation; do not report final-study conclusions."
                    if integrity_ok and gate_passed
                    else "Correct the pilot evidence or integrity failure before experiments."
                ),
            }
            output = Path("/kaggle/working/pilot_validation_report.json") if ON_KAGGLE else WORK / "pilot_validation_report.json"
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, sort_keys=True))
            if not integrity_ok:
                raise RuntimeError("Pilot artifact integrity validation failed")
            '''
        ),
    ],
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
)
notebook["nbformat"] = 4
notebook["nbformat_minor"] = 5
nbf.write(notebook, OUTPUT)
print(OUTPUT)
