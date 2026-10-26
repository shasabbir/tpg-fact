from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .labels import normalize_rating


REVIEW_FIELDS = (
    "candidate_id",
    "decision",
    "exclusion_reason",
    "publisher",
    "publisher_site",
    "claim_text",
    "claimant",
    "api_claim_date",
    "api_review_date",
    "review_url",
    "review_title",
    "textual_rating",
    "suggested_label",
    "language_code",
    "collected_query",
    "collected_publisher_filter",
    "us_relevant",
    "multimodal",
    "target_image_verified",
    "rating_verified",
    "claim_image_url",
    "claim_time",
    "decision_cutoff",
    "cutoff_basis",
    "cutoff_confidence",
    "normalized_label",
    "event_id",
    "evidence_query",
    "sufficiency_target",
    "sufficiency_target_source",
    "reviewer_notes",
)


YES_VALUES = {"yes", "y", "true", "1"}
DECISIONS = {"pending", "keep", "reject"}
SUFFICIENCY_TARGETS = {"sufficient", "insufficient", "unlabeled"}


def candidate_to_review_row(candidate: Mapping[str, Any]) -> dict[str, str]:
    row = {field: "" for field in REVIEW_FIELDS}
    for field in REVIEW_FIELDS:
        if field in candidate:
            row[field] = str(candidate[field] or "")
    row["decision"] = "pending"
    row["suggested_label"] = normalize_rating(row["textual_rating"]) or ""
    row["cutoff_basis"] = "fact_check_publication"
    row["cutoff_confidence"] = "confirmed"
    row["sufficiency_target"] = "unlabeled"
    return row


def write_review_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(candidate_to_review_row(row))


def read_review_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def validate_review_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    rows = list(rows)
    errors: list[dict[str, str]] = []
    decisions: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        decision = row.get("decision", "").strip().casefold() or "pending"
        decisions[decision] += 1
        candidate_id = row.get("candidate_id", "").strip()
        if decision not in DECISIONS:
            errors.append(
                {"row": str(row_number), "candidate_id": candidate_id, "error": "invalid_decision"}
            )
            continue
        if decision == "reject":
            reason = row.get("exclusion_reason", "").strip()
            if not reason:
                errors.append(
                    {"row": str(row_number), "candidate_id": candidate_id, "error": "missing_exclusion_reason"}
                )
            else:
                exclusions[reason] += 1
            continue
        if decision != "keep":
            continue

        required_yes = ("us_relevant", "multimodal", "target_image_verified", "rating_verified")
        for field in required_yes:
            if row.get(field, "").strip().casefold() not in YES_VALUES:
                errors.append(
                    {"row": str(row_number), "candidate_id": candidate_id, "error": f"{field}_not_confirmed"}
                )
        for field in (
            "claim_text",
            "review_url",
            "claim_image_url",
            "claim_time",
            "decision_cutoff",
            "normalized_label",
            "event_id",
        ):
            if not row.get(field, "").strip():
                errors.append(
                    {"row": str(row_number), "candidate_id": candidate_id, "error": f"missing_{field}"}
                )
        target = row.get("sufficiency_target", "unlabeled").strip().casefold()
        if target not in SUFFICIENCY_TARGETS:
            errors.append(
                {"row": str(row_number), "candidate_id": candidate_id, "error": "invalid_sufficiency_target"}
            )
    return {
        "row_count": len(rows),
        "decisions": dict(sorted(decisions.items())),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "errors": errors,
        "status": "valid" if not errors else "needs_correction",
    }
