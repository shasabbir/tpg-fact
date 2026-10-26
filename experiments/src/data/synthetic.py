from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256

from .schema import ClaimRecord, EvidenceRecord


UTC = timezone.utc


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def build_synthetic_records() -> tuple[list[ClaimRecord], list[EvidenceRecord]]:
    """Create deterministic records for integration checks, never scientific use."""
    specifications = [
        ("c-train-1", "event-election-a", 2022, "supported", "Agency confirms election result"),
        ("c-train-2", "event-storm-a", 2023, "refuted", "Old storm image does not show current hurricane"),
        ("c-val-1", "event-health-a", 2024, "insufficient_evidence", "Officials report no evidence for viral treatment claim"),
        ("c-test-1", "event-election-b", 2025, "supported", "Official record supports the voting claim"),
        ("c-test-2", "event-fire-a", 2026, "refuted", "Archive refutes caption attached to historical fire image"),
        ("c-test-3", "event-tech-a", 2026, "insufficient_evidence", "No evidence establishes the alleged technology announcement"),
    ]
    claims: list[ClaimRecord] = []
    evidence: list[EvidenceRecord] = []

    for index, (claim_id, event_id, year, label, evidence_title) in enumerate(specifications):
        claim_time = datetime(year, 3, 1, 12, 0, tzinfo=UTC)
        cutoff = claim_time + timedelta(days=4)
        image_hash = f"{index + 1:016x}"
        first_evidence_id = f"e-{claim_id}-1"
        duplicate_evidence_id = f"e-{claim_id}-2"
        late_evidence_id = f"e-{claim_id}-late"
        claims.append(
            ClaimRecord(
                claim_id=claim_id,
                claim_text=evidence_title.replace("Official record ", "").replace("Archive ", ""),
                claim_image_url=f"https://images.example/{claim_id}.jpg",
                image_sha256=_hash(f"image:{claim_id}"),
                image_phash=image_hash,
                claim_time=claim_time,
                decision_cutoff=cutoff,
                fact_check_url=f"https://factcheck.example/{claim_id}",
                source_rating={
                    "supported": "True",
                    "refuted": "False",
                    "insufficient_evidence": "Unproven",
                }[label],
                normalized_label=label,
                event_id=event_id,
                evidence_ids=(first_evidence_id, duplicate_evidence_id, late_evidence_id),
                sufficiency_target=(
                    "insufficient" if label == "insufficient_evidence" else "sufficient"
                ),
                sufficiency_target_source="synthetic_fixture",
            )
        )
        evidence.extend(
            [
                EvidenceRecord(
                    evidence_id=first_evidence_id,
                    url=f"https://source{index}.example/original",
                    domain=f"source{index}.example",
                    title=evidence_title,
                    published_at=claim_time + timedelta(hours=6),
                    archived_at=claim_time + timedelta(hours=7),
                    text_sha256=_hash(evidence_title),
                    image_phash=image_hash,
                    media_type="text_image",
                    outgoing_links=(),
                    retrieval_query=claims[-1].claim_text,
                    timestamp_confidence="confirmed",
                    timestamp_source="synthetic_archive_observation",
                ),
                EvidenceRecord(
                    evidence_id=duplicate_evidence_id,
                    url=f"https://syndicated{index}.example/copy",
                    domain=f"syndicated{index}.example",
                    title=evidence_title + " - syndicated copy",
                    published_at=claim_time + timedelta(hours=10),
                    archived_at=claim_time + timedelta(hours=11),
                    text_sha256=_hash(evidence_title + " syndicated"),
                    image_phash=image_hash,
                    media_type="text_image",
                    outgoing_links=(f"https://source{index}.example/original",),
                    retrieval_query=claims[-1].claim_text,
                    timestamp_confidence="confirmed",
                    timestamp_source="synthetic_archive_observation",
                ),
                EvidenceRecord(
                    evidence_id=late_evidence_id,
                    url=f"https://late{index}.example/verdict",
                    domain=f"late{index}.example",
                    title=f"Final verdict: {label}",
                    published_at=cutoff + timedelta(days=1),
                    archived_at=cutoff + timedelta(days=1, hours=1),
                    text_sha256=_hash(f"late:{label}:{claim_id}"),
                    image_phash="",
                    media_type="text",
                    outgoing_links=(),
                    retrieval_query=claims[-1].claim_text,
                    timestamp_confidence="confirmed",
                    timestamp_source="synthetic_archive_observation",
                ),
            ]
        )
    return claims, evidence

