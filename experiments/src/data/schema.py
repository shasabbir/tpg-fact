from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


LABELS = ("supported", "refuted", "insufficient_evidence")
SUFFICIENCY_TARGETS = ("sufficient", "insufficient", "unlabeled")
TIMESTAMP_CONFIDENCE_LEVELS = ("confirmed", "observed", "uncertain")


def parse_datetime(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def datetime_to_text(value: datetime) -> str:
    return parse_datetime(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    claim_text: str
    claim_image_url: str
    image_sha256: str
    image_phash: str
    claim_time: datetime
    decision_cutoff: datetime
    fact_check_url: str
    source_rating: str
    normalized_label: str
    event_id: str
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    cutoff_basis: str = "fact_check_publication"
    cutoff_confidence: str = "confirmed"
    sufficiency_target: str = "unlabeled"
    sufficiency_target_source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_time", parse_datetime(self.claim_time))
        object.__setattr__(self, "decision_cutoff", parse_datetime(self.decision_cutoff))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if self.normalized_label not in LABELS:
            raise ValueError(f"Unsupported label: {self.normalized_label}")
        if self.sufficiency_target not in SUFFICIENCY_TARGETS:
            raise ValueError(f"Unsupported sufficiency target: {self.sufficiency_target}")
        if self.claim_time > self.decision_cutoff:
            raise ValueError("claim_time must not be later than decision_cutoff")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimRecord":
        data = dict(value)
        data["claim_time"] = parse_datetime(data["claim_time"])
        data["decision_cutoff"] = parse_datetime(data["decision_cutoff"])
        data["evidence_ids"] = tuple(data.get("evidence_ids", ()))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["claim_time"] = datetime_to_text(self.claim_time)
        data["decision_cutoff"] = datetime_to_text(self.decision_cutoff)
        data["evidence_ids"] = list(self.evidence_ids)
        return data


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    url: str
    domain: str
    title: str
    published_at: datetime
    archived_at: datetime
    text_sha256: str
    image_phash: str
    media_type: str
    source_cluster_id: str = ""
    outgoing_links: tuple[str, ...] = field(default_factory=tuple)
    retrieval_query: str = ""
    timestamp_confidence: str = "uncertain"
    timestamp_source: str = "self_reported_metadata"
    dependency_signals: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", parse_datetime(self.published_at))
        object.__setattr__(self, "archived_at", parse_datetime(self.archived_at))
        object.__setattr__(self, "outgoing_links", tuple(self.outgoing_links))
        object.__setattr__(self, "dependency_signals", tuple(self.dependency_signals))
        if self.timestamp_confidence not in TIMESTAMP_CONFIDENCE_LEVELS:
            raise ValueError(
                f"Unsupported timestamp confidence: {self.timestamp_confidence}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        data = dict(value)
        data["published_at"] = parse_datetime(data["published_at"])
        data["archived_at"] = parse_datetime(data["archived_at"])
        data["outgoing_links"] = tuple(data.get("outgoing_links", ()))
        data["dependency_signals"] = tuple(data.get("dependency_signals", ()))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published_at"] = datetime_to_text(self.published_at)
        data["archived_at"] = datetime_to_text(self.archived_at)
        data["outgoing_links"] = list(self.outgoing_links)
        data["dependency_signals"] = list(self.dependency_signals)
        return data


@dataclass(frozen=True)
class PredictionRecord:
    claim_id: str
    predicted_label: str
    class_probabilities: Mapping[str, float]
    sufficient_evidence_score: float
    selected_evidence_ids: tuple[str, ...]
    source_cluster_ids: tuple[str, ...]
    decision_cutoff: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_cutoff", parse_datetime(self.decision_cutoff))
        object.__setattr__(self, "selected_evidence_ids", tuple(self.selected_evidence_ids))
        object.__setattr__(self, "source_cluster_ids", tuple(self.source_cluster_ids))
        if self.predicted_label not in LABELS:
            raise ValueError(f"Unsupported prediction label: {self.predicted_label}")
        missing = set(LABELS) - set(self.class_probabilities)
        if missing:
            raise ValueError(f"Missing class probabilities: {sorted(missing)}")
        total = sum(float(v) for v in self.class_probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("class_probabilities must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "predicted_label": self.predicted_label,
            "class_probabilities": dict(self.class_probabilities),
            "sufficient_evidence_score": self.sufficient_evidence_score,
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "source_cluster_ids": list(self.source_cluster_ids),
            "decision_cutoff": datetime_to_text(self.decision_cutoff),
        }

