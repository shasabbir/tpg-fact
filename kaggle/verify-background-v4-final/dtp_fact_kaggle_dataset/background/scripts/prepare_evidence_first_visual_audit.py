from __future__ import annotations

import argparse
import csv
from pathlib import Path


LEDGER_FIELDS = [
    "contact_sheet_position",
    "candidate_id",
    "audit_decision",
    "audit_reason",
    "audit_actor",
    "human_verified",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def positions(value: str) -> set[int]:
    result: set[int] = set()
    for item in filter(None, (part.strip() for part in value.split(","))):
        if "-" in item:
            left, right = item.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(item))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record an evidence-first contact-sheet audit and emit unseen candidates"
    )
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--non-us-positions", default="")
    parser.add_argument("--mismatch-positions", default="")
    parser.add_argument("--duplicate-positions", default="")
    parser.add_argument("--ledger-output", required=True)
    parser.add_argument("--expanded-review-csv")
    parser.add_argument("--expanded-image-manifest")
    parser.add_argument("--pending-output-directory")
    args = parser.parse_args()

    manifest = read_csv(Path(args.image_manifest))
    non_us = positions(args.non_us_positions)
    mismatch = positions(args.mismatch_positions)
    duplicate = positions(args.duplicate_positions)
    rejected = non_us | mismatch | duplicate
    maximum = len(manifest)
    invalid = sorted(position for position in rejected if not 1 <= position <= maximum)
    if invalid:
        raise SystemExit(f"Audit positions outside 1..{maximum}: {invalid}")

    ledger: list[dict[str, str]] = []
    for position, row in enumerate(manifest, start=1):
        if position in duplicate:
            decision, reason = "reject", "duplicate_claim_or_image_instance"
        elif position in mismatch:
            decision, reason = "reject", "claim_image_mismatch"
        elif position in non_us:
            decision, reason = "reject", "not_us_focused_under_pilot_scope"
        else:
            decision, reason = "pass", "visual_claim_match_and_us_scope"
        ledger.append(
            {
                "contact_sheet_position": str(position),
                "candidate_id": row["candidate_id"],
                "audit_decision": decision,
                "audit_reason": reason,
                "audit_actor": "AI-assisted visual and scope inspection",
                "human_verified": "no",
            }
        )
    write_csv(Path(args.ledger_output), ledger, LEDGER_FIELDS)

    if any(
        value is not None
        for value in (
            args.expanded_review_csv,
            args.expanded_image_manifest,
            args.pending_output_directory,
        )
    ):
        if not all(
            (
                args.expanded_review_csv,
                args.expanded_image_manifest,
                args.pending_output_directory,
            )
        ):
            raise SystemExit("All three expanded/pending arguments are required together")
        audited_ids = {row["candidate_id"] for row in ledger}
        expanded_review = read_csv(Path(args.expanded_review_csv))
        expanded_manifest = read_csv(Path(args.expanded_image_manifest))
        pending_manifest = [
            row for row in expanded_manifest if row["candidate_id"] not in audited_ids
        ]
        pending_ids = {row["candidate_id"] for row in pending_manifest}
        pending_review = [
            row
            for row in expanded_review
            if row["candidate_id"] in pending_ids and row.get("decision") == "keep"
        ]
        output_root = Path(args.pending_output_directory)
        write_csv(
            output_root / "pending_review.csv",
            pending_review,
            list(expanded_review[0]),
        )
        write_csv(
            output_root / "pending_image_manifest.csv",
            pending_manifest,
            list(expanded_manifest[0]),
        )
        print(f"pending_candidates={len(pending_manifest)}")

    print(f"audited_candidates={len(ledger)}")
    print(f"passes={sum(row['audit_decision'] == 'pass' for row in ledger)}")
    print(f"rejects={sum(row['audit_decision'] == 'reject' for row in ledger)}")


if __name__ == "__main__":
    main()
