from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.io import read_jsonl
from retrieval.dense import BgeM3Encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode evidence text with BGE-M3")
    parser.add_argument("--input", required=True, help="JSONL containing evidence_id and text")
    parser.add_argument("--output", required=True, help="Compressed NPZ output")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    identifiers = [str(row["evidence_id"]) for row in rows]
    texts = [str(row.get(args.text_field) or row.get("title") or "") for row in rows]
    encoder = BgeM3Encoder(args.model, revision=args.revision)
    embeddings = encoder.encode(texts, batch_size=args.batch_size)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, ids=np.asarray(identifiers), embeddings=embeddings)
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "text_field": args.text_field,
        "count": len(identifiers),
        "dimension": int(embeddings.shape[1]) if len(embeddings) else 0,
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
