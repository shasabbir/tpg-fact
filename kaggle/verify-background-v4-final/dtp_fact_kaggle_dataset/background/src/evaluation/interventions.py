from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from data.schema import ClaimRecord, EvidenceRecord


def inject_syndicated_copies(
    evidence: list[EvidenceRecord],
    source_evidence_id: str,
    copies: int,
) -> list[EvidenceRecord]:
    """Add deterministic linked copies for duplicate-robustness evaluation."""
    if copies < 0:
        raise ValueError("copies must be non-negative")
    source = next(
        (item for item in evidence if item.evidence_id == source_evidence_id), None
    )
    if source is None:
        raise KeyError(source_evidence_id)
    cluster_id = source.source_cluster_id or f"intervention-cluster:{source.evidence_id}"
    injected = [
        replace(item, source_cluster_id=cluster_id)
        if item.evidence_id == source.evidence_id
        else item
        for item in evidence
    ]
    for index in range(1, copies + 1):
        injected.append(
            replace(
                source,
                evidence_id=f"{source.evidence_id}__copy_{index:02d}",
                url=f"https://syndication.invalid/{source.evidence_id}/{index}",
                domain=f"syndication-{index}.invalid",
                title=f"{source.title} [syndicated copy {index}]",
                published_at=source.published_at + timedelta(minutes=index),
                archived_at=source.archived_at + timedelta(minutes=index),
                source_cluster_id=cluster_id,
                outgoing_links=(source.url,),
                dependency_signals=("direct_hyperlink", "synthetic_duplicate_injection"),
            )
        )
    return injected


def inject_post_cutoff_verdict(
    claim: ClaimRecord,
    evidence: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Add a label-revealing document after the decision cutoff."""
    published = claim.decision_cutoff + timedelta(hours=1)
    injected = EvidenceRecord(
        evidence_id=f"{claim.claim_id}__post_cutoff_verdict",
        url=f"https://post-cutoff.invalid/{claim.claim_id}",
        domain="post-cutoff.invalid",
        title=f"Final verdict: {claim.normalized_label}",
        published_at=published,
        archived_at=published + timedelta(minutes=5),
        text_sha256="",
        image_phash="",
        media_type="text",
        retrieval_query=claim.claim_text,
        timestamp_confidence="confirmed",
        timestamp_source="synthetic_intervention",
        dependency_signals=("synthetic_post_cutoff_injection",),
    )
    return [*evidence, injected]


def remove_earliest_source(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """Remove the earliest observed item with a deterministic tie break."""
    if not evidence:
        return []
    earliest = min(evidence, key=lambda item: (item.archived_at, item.evidence_id))
    return [item for item in evidence if item.evidence_id != earliest.evidence_id]


def intervention_manifest(
    claim: ClaimRecord,
    original: list[EvidenceRecord],
    modified: list[EvidenceRecord],
    intervention: str,
) -> dict[str, object]:
    original_ids = {item.evidence_id for item in original}
    modified_ids = {item.evidence_id for item in modified}
    return {
        "claim_id": claim.claim_id,
        "intervention": intervention,
        "added_evidence_ids": sorted(modified_ids - original_ids),
        "removed_evidence_ids": sorted(original_ids - modified_ids),
        "original_count": len(original),
        "modified_count": len(modified),
        "decision_cutoff": claim.to_dict()["decision_cutoff"],
    }
