from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from trace_scv.schema import ClaimPacket, EvidenceItem, Prediction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_to_dict(item: EvidenceItem) -> dict:
    row = asdict(item)
    row["metadata"] = dict(item.metadata)
    return row


def packet_to_dict(packet: ClaimPacket) -> dict:
    return {
        "claim_id": packet.claim_id,
        "claim_text": packet.claim_text,
        "label": packet.label,
        "evidence": [evidence_to_dict(item) for item in packet.evidence],
        "split": packet.split,
        "dataset": packet.dataset,
        "group_id": packet.group_id,
    }


def packet_from_dict(row: dict) -> ClaimPacket:
    evidence = tuple(
        EvidenceItem(
            evidence_id=item["evidence_id"],
            modality=item["modality"],
            content=item.get("content", ""),
            source_document_id=item.get("source_document_id"),
            media_path=item.get("media_path"),
            gold=bool(item.get("gold", False)),
            metadata=item.get("metadata", {}),
        )
        for item in row["evidence"]
    )
    return ClaimPacket(
        claim_id=row["claim_id"],
        claim_text=row["claim_text"],
        label=row["label"],
        evidence=evidence,
        split=row["split"],
        dataset=row["dataset"],
        group_id=row.get("group_id"),
    )


def write_packets(path: Path, packets: Iterable[ClaimPacket]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet_to_dict(packet), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_packets(path: Path) -> list[ClaimPacket]:
    with path.open(encoding="utf-8") as handle:
        return [packet_from_dict(json.loads(line)) for line in handle if line.strip()]


def write_predictions(path: Path, predictions: Iterable[Prediction]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Write audit-friendly JSONL records without changing their public schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count
