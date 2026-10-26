from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from math import ceil
from typing import Any

from .schema import ClaimRecord, EvidenceRecord
from .temporal import filter_time_valid


def audit_pilot(
    review_rows: list[Mapping[str, str]],
    claims: list[ClaimRecord],
    evidence: list[EvidenceRecord],
    *,
    target_cases: int = 200,
    final_target_cases: int = 2000,
    independent_cluster_target: float = 0.70,
) -> dict[str, Any]:
    decisions = Counter(
        (row.get("decision", "").strip().casefold() or "pending")
        for row in review_rows
    )
    exclusions = Counter(
        row.get("exclusion_reason", "").strip()
        for row in review_rows
        if row.get("decision", "").strip().casefold() == "reject"
        and row.get("exclusion_reason", "").strip()
    )
    screened = decisions["keep"] + decisions["reject"]
    yield_rate = decisions["keep"] / screened if screened else 0.0
    projected_candidates = ceil(final_target_cases / yield_rate) if yield_rate else None

    evidence_by_id = {item.evidence_id: item for item in evidence}
    with_two_clusters = 0
    claims_with_time_valid_evidence = 0
    for claim in claims:
        candidates = [
            evidence_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        valid = filter_time_valid(candidates, claim.decision_cutoff)
        if valid:
            claims_with_time_valid_evidence += 1
        clusters = {item.source_cluster_id for item in valid if item.source_cluster_id}
        if len(clusters) >= 2:
            with_two_clusters += 1

    independent_fraction = with_two_clusters / len(claims) if claims else 0.0
    image_complete = sum(bool(claim.image_sha256 and claim.image_phash) for claim in claims)
    sufficiency_complete = sum(claim.sufficiency_target != "unlabeled" for claim in claims)
    label_counts = Counter(claim.normalized_label for claim in claims)
    year_counts = Counter(claim.claim_time.year for claim in claims)

    collection_complete = len(claims) >= target_cases
    evidence_complete = bool(claims) and claims_with_time_valid_evidence == len(claims)
    cluster_target_met = independent_fraction >= independent_cluster_target
    if not collection_complete:
        phase_status = "screening_in_progress"
    elif not evidence_complete:
        phase_status = "claim_pilot_complete_evidence_pending"
    elif not cluster_target_met:
        phase_status = "pilot_complete_cluster_target_not_met"
    else:
        phase_status = "pilot_complete_ready_for_scale_decision"

    return {
        "report_scope": "200-case feasibility pilot, not the final benchmark gate",
        "phase_status": phase_status,
        "screening": {
            "candidate_rows": len(review_rows),
            "decisions": dict(sorted(decisions.items())),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "screened_count": screened,
            "acceptance_yield": yield_rate,
            "projected_candidates_for_final_target": projected_candidates,
        },
        "retained_claims": {
            "count": len(claims),
            "target": target_cases,
            "labels": dict(sorted(label_counts.items())),
            "years": {str(key): value for key, value in sorted(year_counts.items())},
            "image_hash_complete": image_complete,
            "sufficiency_audited": sufficiency_complete,
        },
        "evidence": {
            "record_count": len(evidence),
            "claims_with_time_valid_evidence": claims_with_time_valid_evidence,
            "claims_with_two_independent_clusters": with_two_clusters,
            "independent_cluster_fraction": independent_fraction,
            "independent_cluster_target": independent_cluster_target,
        },
        "interpretation": (
            "This report measures pilot completion and projected collection burden. "
            "It must not be interpreted as evidence that the 2,000-case benchmark exists."
        ),
    }
