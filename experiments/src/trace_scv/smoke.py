from __future__ import annotations

import json
from pathlib import Path

from .evaluation.metrics import citation_macro_prf, comprehensiveness, label_macro_f1
from .interventions import add_irrelevant, delete_evidence, duplicate_evidence
from .schema import ClaimPacket, EvidenceItem, Prediction


def build_synthetic_packets() -> tuple[ClaimPacket, ...]:
    return (
        ClaimPacket(
            claim_id="c1",
            claim_text="Model A scores higher than Model B in the table.",
            label="supported",
            split="test",
            dataset="synthetic",
            group_id="paper-1",
            evidence=(
                EvidenceItem("t1", "table", "A: 91; B: 84", gold=True),
                EvidenceItem("x1", "text", "The experiment uses five seeds."),
            ),
        ),
        ClaimPacket(
            claim_id="c2",
            claim_text="The curve decreases after epoch ten.",
            label="refuted",
            split="test",
            dataset="synthetic",
            group_id="paper-2",
            evidence=(
                EvidenceItem("f1", "figure", "Curve rises from epochs 10 to 20", gold=True),
                EvidenceItem("c1", "caption", "Validation performance by epoch."),
            ),
        ),
    )


def run_smoke(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    packets = build_synthetic_packets()
    predictions = (
        Prediction("c1", "supported", {"supported": 0.9, "refuted": 0.1}, ("t1",), "A is higher [t1]."),
        Prediction("c2", "refuted", {"supported": 0.2, "refuted": 0.8}, ("f1",), "The curve rises [f1]."),
    )

    deleted = [delete_evidence(packet, packet.gold_evidence_ids) for packet in packets]
    irrelevant = EvidenceItem("irrelevant", "text", "Unrelated background material.")
    augmented = [add_irrelevant(packet, irrelevant.with_id(f"irrelevant-{index}")) for index, packet in enumerate(packets)]
    duplicated = [duplicate_evidence(packet, next(iter(packet.gold_evidence_ids))) for packet in packets]

    citation = citation_macro_prf(
        [prediction.cited_evidence_ids for prediction in predictions],
        [packet.gold_evidence_ids for packet in packets],
    )
    verdict_f1 = label_macro_f1(
        [packet.label for packet in packets],
        [prediction.predicted_label for prediction in predictions],
    )
    deletion_score = comprehensiveness(
        [prediction.probabilities for prediction in predictions],
        [{"supported": 0.5, "refuted": 0.5}, {"supported": 0.5, "refuted": 0.5}],
        [packet.label for packet in packets],
    )
    result = {
        "status": "pass",
        "packet_count": len(packets),
        "citation_macro": citation,
        "verdict_macro_f1": verdict_f1,
        "comprehensiveness": deletion_score,
        "intervention_sizes": {
            "deleted": [len(packet.evidence) for packet in deleted],
            "irrelevant_added": [len(packet.evidence) for packet in augmented],
            "duplicated": [len(packet.evidence) for packet in duplicated],
        },
        "scientific_results_validated": False,
        "interpretation": "Synthetic software check only.",
    }
    (output / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
