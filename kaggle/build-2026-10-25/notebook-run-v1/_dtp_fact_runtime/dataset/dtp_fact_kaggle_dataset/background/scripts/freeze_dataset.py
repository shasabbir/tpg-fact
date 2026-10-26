from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from data.io import read_jsonl, write_json, write_jsonl
from data.schema import ClaimRecord, EvidenceRecord
from data.splits import assert_no_group_leakage, partition


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze immutable temporal claim/evidence manifests"
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    evidence = [EvidenceRecord.from_dict(row) for row in read_jsonl(args.evidence)]
    if not claims or not evidence:
        raise ValueError("Claims and evidence must be non-empty")
    if len({row.claim_id for row in claims}) != len(claims):
        raise ValueError("Duplicate claim identifiers")
    if len({row.evidence_id for row in evidence}) != len(evidence):
        raise ValueError("Duplicate evidence identifiers")
    assert_no_group_leakage(claims)
    evidence_ids = {row.evidence_id for row in evidence}
    missing = {
        claim.claim_id: sorted(set(claim.evidence_ids) - evidence_ids)
        for claim in claims
        if set(claim.evidence_ids) - evidence_ids
    }
    if missing:
        raise ValueError(f"Claims reference missing evidence: {dict(list(missing.items())[:5])}")
    splits = partition(claims)
    if any(not rows for rows in splits.values()):
        raise ValueError(f"Every split must be non-empty: { {k: len(v) for k, v in splits.items()} }")
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    claims_path = output_root / "claims.jsonl"
    evidence_path = output_root / "evidence.jsonl"
    write_jsonl(claims_path, (row.to_dict() for row in claims))
    write_jsonl(evidence_path, (row.to_dict() for row in evidence))
    split_records: dict[str, dict[str, object]] = {}
    for split, rows in splits.items():
        path = output_root / f"{split}_claims.jsonl"
        write_jsonl(path, (row.to_dict() for row in rows))
        split_records[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "claims": len(rows),
            "events": len({row.event_id for row in rows}),
            "labels": dict(sorted(Counter(row.normalized_label for row in rows).items())),
            "years": dict(sorted(Counter(row.claim_time.year for row in rows).items())),
        }
    report = {
        "status": "frozen_dataset",
        "frozen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_claims": str(Path(args.claims).resolve()),
        "source_evidence": str(Path(args.evidence).resolve()),
        "claims": str(claims_path),
        "claims_sha256": _sha256(claims_path),
        "evidence": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "claims_with_evidence": sum(bool(row.evidence_ids) for row in claims),
        "independent_source_cluster_fraction": sum(
            len(
                {
                    item.source_cluster_id
                    for item in evidence
                    if item.evidence_id in claim.evidence_ids and item.source_cluster_id
                }
            )
            >= 2
            for claim in claims
        )
        / len(claims),
        "splits": split_records,
        "group_leakage_checked": ["event_id", "image_phash"],
    }
    write_json(output_root / "dataset_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
