import pytest

from trace_scv.schema import ClaimPacket, EvidenceItem, Prediction


def test_duplicate_evidence_ids_rejected() -> None:
    evidence = EvidenceItem("e", "text", "content")
    with pytest.raises(ValueError):
        ClaimPacket("c", "claim", "support", (evidence, evidence), "train", "test")


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        Prediction("c", "support", {"support": 0.4, "refute": 0.4}, (), "")
