from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def temporal_split(value: str) -> str:
    year = int(value[:4])
    return "train" if year <= 2023 else "validation" if year == 2024 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the strongest 200 cases from the audited evidence-first buffer"
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--evidence-review", required=True)
    parser.add_argument("--page-audit", required=True)
    parser.add_argument("--visual-ledger", action="append", required=True)
    parser.add_argument("--supplemental-evidence-review", action="append", default=[])
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--train-target", type=int, default=50)
    parser.add_argument("--validation-target", type=int, default=39)
    args = parser.parse_args()

    review = read_csv(Path(args.review_csv))
    images = read_csv(Path(args.image_manifest))
    evidence = read_csv(Path(args.evidence_review))
    supplemental_evidence = [
        row
        for path in args.supplemental_evidence_review
        for row in read_csv(Path(path))
    ]
    page_rows = read_csv(Path(args.page_audit))
    ledger = [
        row
        for path in args.visual_ledger
        for row in read_csv(Path(path))
    ]
    audit_by_id = {row["candidate_id"]: row for row in ledger}
    page_by_id = {row["candidate_id"]: row for row in page_rows}
    image_by_id = {row["candidate_id"]: row for row in images}

    records: list[dict[str, Any]] = []
    for row in review:
        candidate_id = row["candidate_id"]
        audit = audit_by_id.get(candidate_id)
        page = page_by_id.get(candidate_id)
        if (
            audit is None
            or audit.get("audit_decision") != "pass"
            or page is None
            or candidate_id not in image_by_id
        ):
            continue
        records.append(
            {
                "candidate_id": candidate_id,
                "label": row["normalized_label"],
                "published_at": page["published_at"],
                "split": temporal_split(page["published_at"]),
                "domain_count": int(page["evidence_domain_count"]),
            }
        )

    two_domain = [row for row in records if row["domain_count"] >= 2]
    one_domain = [row for row in records if row["domain_count"] == 1]
    selected = list(two_domain)
    selected_ids = {row["candidate_id"] for row in selected}

    # Preserve every visually valid minority-label case.  This gives the pilot
    # substantially better three-class coverage than the earlier snapshot.
    minority_one = [
        row
        for row in one_domain
        if row["label"] in {"supported", "insufficient_evidence"}
    ]
    selected.extend(minority_one)
    selected_ids.update(row["candidate_id"] for row in minority_one)

    split_targets = {
        "train": args.train_target,
        "validation": args.validation_target,
        "test": args.target - args.train_target - args.validation_target,
    }
    remaining = [
        row
        for row in one_domain
        if row["candidate_id"] not in selected_ids and row["label"] == "refuted"
    ]
    remaining.sort(key=lambda row: (row["published_at"], row["candidate_id"]), reverse=True)
    for split_name in ("train", "validation", "test"):
        deficit = split_targets[split_name] - sum(
            row["split"] == split_name for row in selected
        )
        if deficit <= 0:
            continue
        candidates = [row for row in remaining if row["split"] == split_name]
        selected.extend(candidates[:deficit])
        selected_ids.update(row["candidate_id"] for row in candidates[:deficit])

    if len(selected) < args.target:
        fallback = [row for row in remaining if row["candidate_id"] not in selected_ids]
        selected.extend(fallback[: args.target - len(selected)])
    if len(selected) != args.target:
        raise SystemExit(
            f"Selection produced {len(selected)} rows instead of target {args.target}"
        )
    selected_ids = {row["candidate_id"] for row in selected}

    final_review: list[dict[str, str]] = []
    for row in review:
        candidate_id = row["candidate_id"]
        audit = audit_by_id.get(candidate_id)
        if candidate_id in selected_ids:
            row["decision"] = "keep"
            row["exclusion_reason"] = ""
            row["target_image_verified"] = "yes"
            row["reviewer_notes"] = (
                "Structured ClaimReview metadata and a pre-cutoff cited source were machine-checked; "
                "the claim image and US scope passed an AI-assisted contact-sheet audit. Human "
                "verification and inter-annotator agreement are not claimed."
            )
        elif candidate_id in image_by_id:
            row["decision"] = "reject"
            row["target_image_verified"] = "no"
            row["exclusion_reason"] = (
                audit["audit_reason"]
                if audit and audit.get("audit_decision") == "reject"
                else "outside_200_case_evidence_first_pilot_selection"
            )
        final_review.append(row)

    final_images = [row for row in images if row["candidate_id"] in selected_ids]
    final_evidence = [row for row in evidence if row["claim_id"] in selected_ids]
    final_evidence.extend(
        row for row in supplemental_evidence if row["claim_id"] in selected_ids
    )
    final_ledger = [row for row in ledger if row["candidate_id"] in image_by_id]
    final_records = sorted(selected, key=lambda row: row["candidate_id"])
    output_root = Path(args.output_directory)
    write_csv(output_root / "pilot_review_ai_audited.csv", final_review, list(review[0]))
    write_csv(output_root / "image_manifest_ai_audited.csv", final_images, list(images[0]))
    write_csv(output_root / "evidence_review_ai_audited.csv", final_evidence, list(evidence[0]))
    write_csv(output_root / "visual_audit_ledger.csv", final_ledger, list(ledger[0]))
    write_csv(
        output_root / "pilot_selection.csv",
        final_records,
        ["candidate_id", "label", "published_at", "split", "domain_count"],
    )

    label_counts = Counter(row["label"] for row in selected)
    split_counts = Counter(row["split"] for row in selected)
    kept_domains: dict[str, set[str]] = {candidate_id: set() for candidate_id in selected_ids}
    for row in final_evidence:
        if row.get("decision", "").casefold() == "keep":
            kept_domains[row["claim_id"]].add(row["source_cluster_id"])
    two_source_claims = sum(len(domains) >= 2 for domains in kept_domains.values())
    report = {
        "selected_claims": len(selected),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "claims_with_two_cited_archive_domains": sum(
            row["domain_count"] >= 2 for row in selected
        ),
        "two_domain_fraction": sum(row["domain_count"] >= 2 for row in selected)
        / len(selected),
        "claims_with_two_time_valid_source_domains_after_supplement": two_source_claims,
        "two_time_valid_source_domain_fraction_after_supplement": two_source_claims
        / len(selected),
        "audit_actor": "AI-assisted visual and scope inspection",
        "human_verified": False,
        "status": "selected_for_final_pilot_integrity_checks",
    }
    (output_root / "selection_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
