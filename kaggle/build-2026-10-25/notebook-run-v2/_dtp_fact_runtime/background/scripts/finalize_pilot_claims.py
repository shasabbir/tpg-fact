from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from data.io import write_json, write_jsonl
from data.pilot_review import read_review_csv, validate_review_rows
from data.schema import ClaimRecord


def _image_manifest(path: str) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            row["candidate_id"]: dict(row)
            for row in csv.DictReader(stream)
            if row.get("candidate_id")
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audited candidate rows and verified image hashes into ClaimRecord JSONL"
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_review_csv(args.review_csv)
    validation = validate_review_rows(rows)
    if validation["status"] != "valid":
        write_json(Path(args.output).with_suffix(".report.json"), validation)
        raise SystemExit(
            f"Review CSV has {len(validation['errors'])} error(s); see the report before finalizing"
        )
    images = _image_manifest(args.image_manifest)
    claims: list[ClaimRecord] = []
    errors: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        if row.get("decision", "").strip().casefold() != "keep":
            continue
        candidate_id = row["candidate_id"].strip()
        image = images.get(candidate_id, {})
        if image.get("status") != "ok":
            errors.append(
                {"row": str(row_number), "candidate_id": candidate_id, "error": "missing_verified_image"}
            )
            continue
        try:
            claims.append(
                ClaimRecord.from_dict(
                    {
                        "claim_id": candidate_id,
                        "claim_text": row["claim_text"].strip(),
                        "claim_image_url": row["claim_image_url"].strip(),
                        "image_sha256": image.get("image_sha256", ""),
                        "image_phash": image.get("image_phash", ""),
                        "claim_time": row["claim_time"].strip(),
                        "decision_cutoff": row["decision_cutoff"].strip(),
                        "fact_check_url": row["review_url"].strip(),
                        "source_rating": row["textual_rating"].strip(),
                        "normalized_label": row["normalized_label"].strip(),
                        "event_id": row["event_id"].strip(),
                        "evidence_ids": [],
                        "cutoff_basis": row.get("cutoff_basis", "fact_check_publication").strip(),
                        "cutoff_confidence": row.get("cutoff_confidence", "confirmed").strip(),
                        "sufficiency_target": row.get("sufficiency_target", "unlabeled").strip(),
                        "sufficiency_target_source": row.get("sufficiency_target_source", "").strip(),
                    }
                )
            )
        except (TypeError, ValueError) as exc:
            errors.append(
                {"row": str(row_number), "candidate_id": candidate_id, "error": str(exc)}
            )

    report = {
        "accepted_claims": len(claims),
        "errors": errors,
        "status": "valid" if not errors else "needs_correction",
    }
    write_json(Path(args.output).with_suffix(".report.json"), report)
    if errors:
        raise SystemExit(f"Could not finalize {len(errors)} kept row(s); see report")
    write_jsonl(args.output, (claim.to_dict() for claim in claims))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
