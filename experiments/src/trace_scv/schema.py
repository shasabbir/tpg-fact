from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence


ALLOWED_MODALITIES = frozenset({"text", "table", "figure", "image", "caption"})


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    modality: str
    content: str
    source_document_id: str | None = None
    media_path: str | None = None
    gold: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(f"unsupported modality: {self.modality}")
        if not self.content.strip() and not self.media_path:
            raise ValueError("evidence must contain text or a media path")

    def with_id(self, evidence_id: str, **metadata: str) -> "EvidenceItem":
        merged = dict(self.metadata)
        merged.update(metadata)
        return replace(self, evidence_id=evidence_id, metadata=merged)


@dataclass(frozen=True, slots=True)
class ClaimPacket:
    claim_id: str
    claim_text: str
    label: str
    evidence: tuple[EvidenceItem, ...]
    split: str
    dataset: str
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.claim_text.strip():
            raise ValueError("claim_id and claim_text must be non-empty")
        identifiers = [item.evidence_id for item in self.evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence identifiers must be unique within a packet")

    @property
    def gold_evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.evidence if item.gold)

    def replace_evidence(self, evidence: Sequence[EvidenceItem]) -> "ClaimPacket":
        return replace(self, evidence=tuple(evidence))


@dataclass(frozen=True, slots=True)
class Prediction:
    claim_id: str
    predicted_label: str
    probabilities: Mapping[str, float]
    cited_evidence_ids: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim_id must be non-empty")
        total = sum(float(value) for value in self.probabilities.values())
        if self.probabilities and abs(total - 1.0) > 1e-5:
            raise ValueError("probabilities must sum to one")
        if len(self.cited_evidence_ids) != len(set(self.cited_evidence_ids)):
            raise ValueError("cited evidence identifiers must be unique")
