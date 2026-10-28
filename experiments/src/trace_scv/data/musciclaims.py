from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_scv.data.io import sha256_file, write_packets
from trace_scv.schema import ClaimPacket, EvidenceItem


def prepare_musciclaims(
    dataset_root: Path,
    output_path: Path,
    *,
    max_candidates: int = 12,
    max_test: int | None = None,
) -> dict:
    source = dataset_root / "test_set.jsonl"
    if not source.exists():
        raise FileNotFoundError("MuSciClaims test_set.jsonl is required")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if max_test is not None:
        rows = rows[:max_test]
    figure_root = dataset_root / "paper_figures"
    packets: list[ClaimPacket] = []
    for index, row in enumerate(rows):
        label_text = str(row.get("label_2class") or row.get("label_3class", "")).upper()
        if label_text not in {"SUPPORT", "REFUTE"}:
            continue
        relative = Path(str(row["associated_figure_filepath"]))
        gold_path = dataset_root / relative
        paper_id = str(row["paper_id"])
        evidence = [
            EvidenceItem(
                evidence_id=f"figure:{relative.name}",
                modality="figure",
                content=str(row.get("caption") or row.get("associated_figure_number") or relative.name),
                source_document_id=paper_id,
                media_path=str(gold_path) if gold_path.exists() else None,
                gold=True,
                metadata={"panels": ",".join(row.get("associated_figure_panels") or [])},
            )
        ]
        distractors = sorted(figure_root.glob(f"*{paper_id}*"))
        for path in distractors:
            if path.name == relative.name or len(evidence) >= max_candidates:
                continue
            evidence.append(
                EvidenceItem(
                    evidence_id=f"figure:{path.name}",
                    modality="figure",
                    content=f"Additional figure from paper {paper_id}: {path.stem}",
                    source_document_id=paper_id,
                    media_path=str(path),
                    gold=False,
                )
            )
        packets.append(
            ClaimPacket(
                claim_id=f"musciclaims:{row.get('claim_id', index)}",
                claim_text=str(row["claim_text"]),
                label="supported" if label_text == "SUPPORT" else "refuted",
                evidence=tuple(evidence),
                split="external_test",
                dataset="musciclaims",
                group_id=paper_id,
            )
        )
    count = write_packets(output_path, packets)
    manifest = {
        "dataset": "MuSciClaims",
        "source_revision": "9134142d05fb473e0a10c4c05f0c2d5a2af51099",
        "source_sha256": sha256_file(source),
        "count": count,
        "labels": dict(Counter(packet.label for packet in packets)),
        "groups": len({packet.group_id for packet in packets}),
        "license_note": "The Hugging Face card has no declared license; retain as conditional external evaluation.",
    }
    output_path.with_name("external_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
