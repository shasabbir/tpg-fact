from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the audited feasibility pilot for private Kaggle use")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--assessment", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--visual-audit-ledger", required=True)
    parser.add_argument("--evidence-review", required=True)
    parser.add_argument("--archive-audit", required=True)
    parser.add_argument("--frozen-directory")
    parser.add_argument("--selection-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    claims_path = Path(args.claims)
    evidence_path = Path(args.evidence)
    assessment_path = Path(args.assessment)
    image_manifest_path = Path(args.image_manifest)
    claims = read_jsonl(claims_path)
    evidence = read_jsonl(evidence_path)
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    claim_ids = {str(row["claim_id"]) for row in claims}
    with image_manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        images = list(csv.DictReader(stream))
    if len(claims) != 200 or len(images) != 200:
        raise SystemExit("The pilot package requires exactly 200 claims and 200 image rows")
    if {row["candidate_id"] for row in images} != claim_ids:
        raise SystemExit("Image manifest IDs do not match the pilot claim IDs")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dtp-pilot-") as temporary:
        root = Path(temporary) / "dtp_fact_pilot_data"
        image_root = root / "images"
        audit_root = root / "audit"
        image_root.mkdir(parents=True)
        audit_root.mkdir(parents=True)
        rewritten: list[dict[str, str]] = []
        for row in images:
            source = Path(row["local_path"])
            if not source.is_file() or sha256(source) != row["image_sha256"]:
                raise SystemExit(f"Missing or hash-mismatched image: {source}")
            destination = image_root / source.name
            shutil.copy2(source, destination)
            copied = dict(row)
            copied["local_path"] = f"images/{source.name}"
            rewritten.append(copied)

        shutil.copy2(claims_path, root / "pilot_claims_with_evidence.jsonl")
        shutil.copy2(evidence_path, root / "pilot_evidence.jsonl")
        shutil.copy2(assessment_path, root / "pilot_assessment.json")
        shutil.copy2(args.visual_audit_ledger, audit_root / "visual_audit_ledger.csv")
        shutil.copy2(args.evidence_review, audit_root / "evidence_review_audited.csv")
        shutil.copy2(args.archive_audit, audit_root / "archive_audit.csv")
        if args.selection_manifest:
            shutil.copy2(args.selection_manifest, audit_root / "selection_manifest.json")
        if args.frozen_directory:
            shutil.copytree(Path(args.frozen_directory), root / "frozen")
        with (root / "image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rewritten[0]))
            writer.writeheader()
            writer.writerows(rewritten)

        evidence_claims = {str(row["claim_id"]) for row in claims if row.get("evidence_ids")}
        cluster_rich = 0
        evidence_by_id = {str(row["evidence_id"]): row for row in evidence}
        for claim in claims:
            clusters = {
                str(evidence_by_id[item]["source_cluster_id"])
                for item in claim.get("evidence_ids", [])
                if item in evidence_by_id
            }
            cluster_rich += len(clusters) >= 2
        required_fraction = float(
            assessment["evidence"].get("independent_cluster_target", 0.70)
        )
        gate_passed = (
            len(evidence_claims) == len(claims)
            and cluster_rich / len(claims) >= required_fraction
        )
        manifest = {
            "status": (
                "audited_feasibility_pilot_evidence_gate_passed"
                if gate_passed
                else "audited_feasibility_pilot_evidence_gate_failed"
            ),
            "scientific_results_validated": False,
            "claim_count": len(claims),
            "evidence_count": len(evidence),
            "image_count": len(rewritten),
            "claims_with_time_valid_evidence": len(evidence_claims),
            "claims_with_two_source_clusters": cluster_rich,
            "independent_cluster_fraction": cluster_rich / len(claims),
            "required_independent_cluster_fraction": required_fraction,
            "human_visual_verification_claimed": False,
            "claims_sha256": sha256(root / "pilot_claims_with_evidence.jsonl"),
            "evidence_sha256": sha256(root / "pilot_evidence.jsonl"),
            "assessment_sha256": sha256(root / "pilot_assessment.json"),
            "warning": (
                "This is an AI-assisted 200-case feasibility pilot. It is not the final "
                "2,000-case benchmark and must not be used to claim final study results."
            ),
        }
        (root / "pilot_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "README.md").write_text(
            "# DTP-Fact feasibility pilot\n\n"
            f"Private research bundle containing {len(claims)} image--text ClaimReview cases, "
            f"{len(evidence)} time-valid evidence records, image hashes, frozen temporal splits, "
            "and audit ledgers.\n\n"
            f"The pilot evidence gate {'passed' if gate_passed else 'failed'} "
            f"({len(evidence_claims)}/{len(claims)} claims have evidence; "
            f"{cluster_rich}/{len(claims)} have two source-domain clusters). "
            "The source-domain clusters are a pilot proxy pending the final source-dependency "
            "graph audit. This is not the final 2,000-case benchmark and it does not validate "
            "final paper results. Keep the Kaggle Dataset private.\n",
            encoding="utf-8",
        )

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent))
    print(json.dumps({"output": str(output), "sha256": sha256(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
