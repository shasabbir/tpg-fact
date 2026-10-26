from __future__ import annotations

import argparse
from pathlib import Path

from data.io import read_jsonl, write_json, write_jsonl
from data.schema import ClaimRecord, EvidenceRecord
from evaluation.interventions import (
    inject_post_cutoff_verdict,
    inject_syndicated_copies,
    intervention_manifest,
    remove_earliest_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic evidence intervention sets"
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    evidence = [EvidenceRecord.from_dict(row) for row in read_jsonl(args.evidence)]
    evidence_by_id = {item.evidence_id: item for item in evidence}
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []

    for claim in claims:
        original = [
            evidence_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        if not original:
            continue
        variants: dict[str, list[EvidenceRecord]] = {
            "post_cutoff_verdict": inject_post_cutoff_verdict(claim, original),
            "remove_earliest_source": remove_earliest_source(original),
        }
        for copies in args.copies:
            variants[f"duplicate_{copies}"] = inject_syndicated_copies(
                original, original[0].evidence_id, copies
            )
        for name, records in variants.items():
            target = output / name / f"{claim.claim_id}.jsonl"
            write_jsonl(target, (item.to_dict() for item in records))
            manifest = intervention_manifest(claim, original, records, name)
            manifest["path"] = str(target)
            manifests.append(manifest)

    write_json(output / "manifest.json", manifests)
    print(f"claims={len(claims)} interventions={len(manifests)} output={output}")


if __name__ == "__main__":
    main()
