from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.claimreview import normalize_claimreview
from data.io import write_json, write_jsonl


def _records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "itemListElement" in payload:
        return payload["itemListElement"]
    if isinstance(payload, dict) and "@graph" in payload:
        return [item for item in payload["@graph"] if item.get("@type") == "ClaimReview"]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Unsupported ClaimReview JSON shape")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize strict ClaimReview JSON-LD records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    claims, rejected = normalize_claimreview(_records(payload))
    write_jsonl(args.output, (claim.to_dict() for claim in claims))
    report_path = Path(args.output).with_suffix(".report.json")
    write_json(
        report_path,
        {"accepted": len(claims), "rejected": rejected, "policy": "explicit_ratings_only"},
    )
    print(f"accepted={len(claims)} rejected={rejected} output={args.output}")


if __name__ == "__main__":
    main()

