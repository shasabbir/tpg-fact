from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from urllib.parse import urlparse

from data.io import read_jsonl, write_json, write_jsonl
from data.schema import ClaimRecord, EvidenceRecord
from data.temporal import is_time_valid


YES_VALUES = {"yes", "y", "true", "1"}


def _csv_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize audited evidence and attach evidence IDs to pilot claims"
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--archive-audit")
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--claims-output", required=True)
    parser.add_argument("--exclude-domain", action="append", default=[])
    args = parser.parse_args()

    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    claim_by_id = {claim.claim_id: claim for claim in claims}
    archive_rows = _csv_rows(args.archive_audit) if args.archive_audit else []
    archive_by_id = {row.get("evidence_id", ""): row for row in archive_rows}
    excluded_domains = {
        domain.casefold().removeprefix("www.") for domain in args.exclude_domain
    }
    evidence: list[EvidenceRecord] = []
    evidence_by_claim: dict[str, list[str]] = {claim.claim_id: [] for claim in claims}
    errors: list[dict[str, str]] = []

    for row_number, row in enumerate(_csv_rows(args.review_csv), start=2):
        if row.get("decision", "").strip().casefold() != "keep":
            continue
        evidence_id = row.get("evidence_id", "").strip()
        claim_id = row.get("claim_id", "").strip()
        error_prefix = {"row": str(row_number), "evidence_id": evidence_id, "claim_id": claim_id}
        claim = claim_by_id.get(claim_id)
        if claim is None:
            errors.append({**error_prefix, "error": "unknown_claim_id"})
            continue
        for field in ("relevance_verified", "publication_time_verified", "cluster_verified"):
            if row.get(field, "").strip().casefold() not in YES_VALUES:
                errors.append({**error_prefix, "error": f"{field}_not_confirmed"})
        required = ("url", "title", "published_at", "media_type", "source_cluster_id")
        missing = [field for field in required if not row.get(field, "").strip()]
        if missing:
            errors.append({**error_prefix, "error": f"missing_{'_'.join(missing)}"})
            continue
        url = row["url"].strip()
        domain = (row.get("domain") or urlparse(url).netloc).casefold().removeprefix("www.")
        if domain in excluded_domains:
            errors.append({**error_prefix, "error": "excluded_fact_check_domain"})
            continue

        archive = archive_by_id.get(evidence_id, {})
        archived_at = row.get("archived_at", "").strip()
        timestamp_confidence = row.get("timestamp_confidence", "uncertain").strip().casefold()
        if archive.get("status") == "observed_pre_cutoff":
            archived_at = archive.get("archived_at", "").strip()
            if timestamp_confidence == "uncertain":
                timestamp_confidence = "observed"
        if not archived_at:
            errors.append({**error_prefix, "error": "missing_archived_at"})
            continue
        try:
            record = EvidenceRecord.from_dict(
                {
                    "evidence_id": evidence_id,
                    "url": url,
                    "domain": domain,
                    "title": row["title"].strip(),
                    "published_at": row["published_at"].strip(),
                    "archived_at": archived_at,
                    "text_sha256": row.get("text_sha256", "").strip(),
                    "image_phash": row.get("image_phash", "").strip(),
                    "media_type": row["media_type"].strip(),
                    "source_cluster_id": row["source_cluster_id"].strip(),
                    "outgoing_links": [
                        link.strip()
                        for link in row.get("outgoing_links", "").split("|")
                        if link.strip()
                    ],
                    "retrieval_query": row.get("retrieval_query", "").strip(),
                    "timestamp_confidence": timestamp_confidence,
                    "timestamp_source": (
                        f"common_crawl:{archive.get('common_crawl_index', '')}"
                        if archive.get("status") == "observed_pre_cutoff"
                        else "manual_archive_audit"
                    ),
                    "dependency_signals": [],
                }
            )
        except (TypeError, ValueError) as exc:
            errors.append({**error_prefix, "error": str(exc)})
            continue
        if not is_time_valid(record, claim.decision_cutoff):
            errors.append({**error_prefix, "error": "not_primary_time_valid"})
            continue
        evidence.append(record)
        evidence_by_claim[claim_id].append(evidence_id)

    report = {
        "accepted_evidence": len(evidence),
        "claims_with_evidence": sum(bool(ids) for ids in evidence_by_claim.values()),
        "errors": errors,
        "status": "valid" if not errors else "needs_correction",
    }
    report_path = Path(args.evidence_output).with_suffix(".report.json")
    write_json(report_path, report)
    if errors:
        raise SystemExit(f"Could not finalize {len(errors)} evidence row issue(s); see {report_path}")

    updated_claims = [
        replace(claim, evidence_ids=tuple(sorted(evidence_by_claim[claim.claim_id])))
        for claim in claims
    ]
    write_jsonl(args.evidence_output, (record.to_dict() for record in evidence))
    write_jsonl(args.claims_output, (claim.to_dict() for claim in updated_claims))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
