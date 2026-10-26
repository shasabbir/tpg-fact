from __future__ import annotations

import argparse
import csv
from pathlib import Path


# One-based positions in the deterministic contact-sheet/image-manifest order.
# These cases were removed after visual comparison of the downloaded article
# image against the structured ClaimReview statement.
VISUAL_REJECTS = {
    18, 26, 33, 48, 51, 64, 67, 75,
    81, 82, 85, 87, 89, 93, 97, 99,
    105, 111, 112, 118, 119,
    121, 122, 124, 136,
    143, 146, 149, 151,
    161, 163, 167, 169, 171, 172, 173, 178,
    181, 182, 183, 187, 193, 196, 199,
    202, 205, 213, 219,
    227, 228, 233,
    241, 242, 245, 248,
    261, 262, 263, 268, 273, 279,
    281, 283, 295, 298,
    303, 304, 306, 308, 309, 314, 317, 318,
}

DUPLICATE_OR_CONTRADICTORY = {146, 161, 182, 219}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the recorded AI visual audit to a pilot shortlist")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--target", type=int, default=200)
    args = parser.parse_args()

    review_rows = read_csv(Path(args.review_csv))
    manifest_rows = read_csv(Path(args.image_manifest))
    review_by_id = {row["candidate_id"]: row for row in review_rows}

    accepted_ids: list[str] = []
    ledger: list[dict[str, str]] = []
    for position, image in enumerate(manifest_rows, start=1):
        candidate_id = image["candidate_id"]
        if position in DUPLICATE_OR_CONTRADICTORY:
            decision = "reject"
            reason = "duplicate_or_contradictory_claim_instance"
        elif position in VISUAL_REJECTS:
            decision = "reject"
            reason = "article_image_does_not_show_claimed_visual_content"
        else:
            decision = "pass"
            reason = "visual_content_matches_claimreview_statement"
            accepted_ids.append(candidate_id)
        ledger.append(
            {
                "contact_sheet_position": str(position),
                "candidate_id": candidate_id,
                "audit_decision": decision,
                "audit_reason": reason,
                "audit_actor": "AI-assisted visual inspection",
                "human_verified": "no",
            }
        )

    if len(accepted_ids) < args.target:
        raise SystemExit(f"Only {len(accepted_ids)} visual matches remain; target is {args.target}")
    final_ids = set(accepted_ids[: args.target])
    manifest_by_id = {row["candidate_id"]: row for row in manifest_rows}

    output_reviews: list[dict[str, str]] = []
    for row in review_rows:
        candidate_id = row["candidate_id"]
        if candidate_id in final_ids:
            row["decision"] = "keep"
            row["exclusion_reason"] = ""
            row["target_image_verified"] = "yes"
            row["reviewer_notes"] = (
                "Structured ClaimReview metadata matched the candidate and the downloaded "
                "article image visibly matched the claim statement in an AI-assisted contact-sheet "
                "audit. No human visual-verification or inter-annotator agreement is claimed."
            )
        elif candidate_id in manifest_by_id:
            audit_row = next(item for item in ledger if item["candidate_id"] == candidate_id)
            row["decision"] = "reject"
            row["exclusion_reason"] = (
                audit_row["audit_reason"]
                if audit_row["audit_decision"] == "reject"
                else "outside_200_case_pilot_quota"
            )
            row["target_image_verified"] = "no"
            row["reviewer_notes"] = "Excluded during AI-assisted pilot visual audit."
        output_reviews.append(row)

    output_manifest = [manifest_by_id[candidate_id] for candidate_id in accepted_ids[: args.target]]
    output_root = Path(args.output_directory)
    write_csv(output_root / "pilot_review_ai_audited.csv", output_reviews, list(output_reviews[0]))
    write_csv(output_root / "image_manifest_ai_audited.csv", output_manifest, list(output_manifest[0]))
    write_csv(
        output_root / "visual_audit_ledger.csv",
        ledger,
        [
            "contact_sheet_position",
            "candidate_id",
            "audit_decision",
            "audit_reason",
            "audit_actor",
            "human_verified",
        ],
    )
    print(f"visual_passes={len(accepted_ids)}")
    print(f"final_pilot_rows={len(output_manifest)}")
    print(f"visual_rejects={len(VISUAL_REJECTS)}")
    print(f"output_directory={output_root.resolve()}")


if __name__ == "__main__":
    main()
