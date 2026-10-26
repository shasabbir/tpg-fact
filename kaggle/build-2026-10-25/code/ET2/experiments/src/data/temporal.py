from __future__ import annotations

from collections.abc import Iterable

from .schema import EvidenceRecord, parse_datetime


def is_time_valid(
    evidence: EvidenceRecord,
    cutoff,
    *,
    allow_uncertain: bool = False,
) -> bool:
    """Return whether evidence was demonstrably available by the cutoff.

    The primary protocol requires an independent pre-cutoff observation, stored
    as ``archived_at``. A self-reported publication date alone is not enough.
    ``allow_uncertain`` is reserved for the preregistered sensitivity analysis.
    """
    cutoff_utc = parse_datetime(cutoff)
    chronology_valid = evidence.archived_at >= evidence.published_at
    published_in_time = evidence.published_at <= cutoff_utc
    observed_in_time = evidence.archived_at <= cutoff_utc
    confidence_valid = evidence.timestamp_confidence in {"confirmed", "observed"}
    if allow_uncertain:
        return chronology_valid and published_in_time
    return chronology_valid and published_in_time and observed_in_time and confidence_valid


def filter_time_valid(
    evidence: Iterable[EvidenceRecord],
    cutoff,
    excluded_domains: set[str] | None = None,
    *,
    allow_uncertain: bool = False,
) -> list[EvidenceRecord]:
    excluded = {domain.casefold() for domain in (excluded_domains or set())}
    return [
        item
        for item in evidence
        if is_time_valid(item, cutoff, allow_uncertain=allow_uncertain)
        and item.domain.casefold() not in excluded
    ]


def temporal_audit(evidence: EvidenceRecord, cutoff) -> dict[str, object]:
    """Return the individual checks used by the temporal eligibility gate."""
    cutoff_utc = parse_datetime(cutoff)
    return {
        "evidence_id": evidence.evidence_id,
        "published_by_cutoff": evidence.published_at <= cutoff_utc,
        "observed_by_cutoff": evidence.archived_at <= cutoff_utc,
        "archive_after_publication": evidence.archived_at >= evidence.published_at,
        "timestamp_confidence": evidence.timestamp_confidence,
        "primary_time_valid": is_time_valid(evidence, cutoff_utc),
        "sensitivity_time_valid": is_time_valid(
            evidence, cutoff_utc, allow_uncertain=True
        ),
    }

