from __future__ import annotations

from dataclasses import replace

from .schema import ClaimPacket, EvidenceItem


def delete_evidence(packet: ClaimPacket, evidence_ids: set[str] | frozenset[str]) -> ClaimPacket:
    """Remove exact evidence identifiers from a packet."""
    return packet.replace_evidence(
        item for item in packet.evidence if item.evidence_id not in evidence_ids
    )


def add_irrelevant(packet: ClaimPacket, item: EvidenceItem) -> ClaimPacket:
    """Add a declared irrelevant item without changing the gold verdict."""
    if item.evidence_id in {existing.evidence_id for existing in packet.evidence}:
        raise ValueError("irrelevant evidence_id already exists")
    marked = replace(item, gold=False, metadata={**item.metadata, "intervention": "irrelevant"})
    return packet.replace_evidence((*packet.evidence, marked))


def duplicate_evidence(packet: ClaimPacket, evidence_id: str, copies: int = 1) -> ClaimPacket:
    """Add traceable duplicates of one evidence item."""
    if copies < 1:
        raise ValueError("copies must be positive")
    source = next((item for item in packet.evidence if item.evidence_id == evidence_id), None)
    if source is None:
        raise KeyError(evidence_id)
    duplicates = tuple(
        source.with_id(
            f"{source.evidence_id}__duplicate_{index}",
            intervention="duplicate",
            duplicate_of=source.evidence_id,
        )
        for index in range(1, copies + 1)
    )
    return packet.replace_evidence((*packet.evidence, *duplicates))


def swap_modality(
    packet: ClaimPacket,
    evidence_id: str,
    replacement: EvidenceItem,
) -> ClaimPacket:
    """Replace one item while retaining its identifier for paired evaluation."""
    if replacement.modality not in {"image", "figure", "table", "text", "caption"}:
        raise ValueError("replacement modality is unsupported")
    found = False
    items: list[EvidenceItem] = []
    for item in packet.evidence:
        if item.evidence_id == evidence_id:
            found = True
            items.append(
                replace(
                    replacement,
                    evidence_id=evidence_id,
                    gold=False,
                    metadata={
                        **replacement.metadata,
                        "intervention": "modality_swap",
                        "replaces": evidence_id,
                    },
                )
            )
        else:
            items.append(item)
    if not found:
        raise KeyError(evidence_id)
    return packet.replace_evidence(items)
