from __future__ import annotations

from collections import Counter

from data.schema import ClaimRecord, EvidenceRecord
from data.splits import split_name
from data.temporal import filter_time_valid


def audit_feasibility(
    claims: list[ClaimRecord],
    evidence: list[EvidenceRecord],
    *,
    minimum_total_cases: int = 2000,
    minimum_per_label: int = 300,
    minimum_test_events: int = 50,
    minimum_independent_cluster_fraction: float = 0.70,
) -> dict[str, object]:
    """Evaluate the preregistered benchmark gate without changing it."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    label_counts = Counter(claim.normalized_label for claim in claims)
    test_events = {
        claim.event_id for claim in claims if split_name(claim) == "test"
    }
    claims_with_two_clusters = 0
    timestamp_counts: Counter[str] = Counter()
    unassigned_valid_evidence = 0
    for claim in claims:
        candidates = [
            evidence_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        valid = filter_time_valid(candidates, claim.decision_cutoff)
        timestamp_counts.update(item.timestamp_confidence for item in candidates)
        unassigned_valid_evidence += sum(not item.source_cluster_id for item in valid)
        cluster_ids = {item.source_cluster_id for item in valid if item.source_cluster_id}
        if len(cluster_ids) >= 2:
            claims_with_two_clusters += 1

    independent_fraction = claims_with_two_clusters / len(claims) if claims else 0.0
    checks = {
        "total_cases": len(claims) >= minimum_total_cases,
        "cases_per_label": all(
            label_counts.get(label, 0) >= minimum_per_label
            for label in ("supported", "refuted", "insufficient_evidence")
        ),
        "test_events": len(test_events) >= minimum_test_events,
        "independent_cluster_fraction": (
            independent_fraction >= minimum_independent_cluster_fraction
        ),
    }
    return {
        "status": "benchmark" if all(checks.values()) else "stress_test_fallback",
        "checks": checks,
        "counts": {
            "total_cases": len(claims),
            "labels": dict(sorted(label_counts.items())),
            "test_events": len(test_events),
            "claims_with_two_independent_clusters": claims_with_two_clusters,
            "timestamp_confidence": dict(sorted(timestamp_counts.items())),
            "unassigned_valid_evidence": unassigned_valid_evidence,
        },
        "independent_cluster_fraction": independent_fraction,
        "thresholds": {
            "minimum_total_cases": minimum_total_cases,
            "minimum_per_label": minimum_per_label,
            "minimum_test_events": minimum_test_events,
            "minimum_independent_cluster_fraction": minimum_independent_cluster_fraction,
        },
    }
