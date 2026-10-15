from __future__ import annotations

from collections.abc import Iterable

from .schema import EvidenceRecord, parse_datetime


def is_time_valid(evidence: EvidenceRecord, cutoff) -> bool:
    cutoff_utc = parse_datetime(cutoff)
    return evidence.published_at <= cutoff_utc and evidence.archived_at >= evidence.published_at


def filter_time_valid(
    evidence: Iterable[EvidenceRecord],
    cutoff,
    excluded_domains: set[str] | None = None,
) -> list[EvidenceRecord]:
    excluded = {domain.casefold() for domain in (excluded_domains or set())}
    return [
        item
        for item in evidence
        if is_time_valid(item, cutoff) and item.domain.casefold() not in excluded
    ]

