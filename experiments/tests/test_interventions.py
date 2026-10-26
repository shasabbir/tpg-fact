import pytest

from trace_scv.interventions import add_irrelevant, delete_evidence, duplicate_evidence, swap_modality
from trace_scv.schema import ClaimPacket, EvidenceItem


@pytest.fixture
def packet() -> ClaimPacket:
    return ClaimPacket(
        claim_id="c",
        claim_text="A claim",
        label="support",
        split="test",
        dataset="synthetic",
        evidence=(
            EvidenceItem("gold", "text", "decisive", gold=True),
            EvidenceItem("other", "image", "context", media_path="other.png"),
        ),
    )


def test_delete_decisive(packet: ClaimPacket) -> None:
    changed = delete_evidence(packet, {"gold"})
    assert [item.evidence_id for item in changed.evidence] == ["other"]


def test_add_irrelevant_marks_intervention(packet: ClaimPacket) -> None:
    changed = add_irrelevant(packet, EvidenceItem("new", "text", "irrelevant"))
    assert changed.evidence[-1].metadata["intervention"] == "irrelevant"
    assert not changed.evidence[-1].gold


def test_duplicate_is_traceable(packet: ClaimPacket) -> None:
    changed = duplicate_evidence(packet, "gold", copies=2)
    assert len(changed.evidence) == 4
    assert changed.evidence[-1].metadata["duplicate_of"] == "gold"


def test_swap_retains_identifier_but_not_gold(packet: ClaimPacket) -> None:
    changed = swap_modality(packet, "other", EvidenceItem("replacement", "image", "swapped"))
    swapped = next(item for item in changed.evidence if item.evidence_id == "other")
    assert swapped.metadata["intervention"] == "modality_swap"
    assert not swapped.gold
