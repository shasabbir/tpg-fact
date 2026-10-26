from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from data.io import read_jsonl, write_json
from retrieval.dense import BgeM3Encoder, Siglip2ImageEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen claim/evidence text and image embedding archives"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--claim-image-manifest", required=True)
    parser.add_argument("--evidence-image-manifest")
    parser.add_argument("--evidence-text", help="Optional JSONL with evidence_id and text")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--device")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_rows(path: str | Path, id_field: str) -> dict[str, Path]:
    source = Path(path).resolve()
    result: dict[str, Path] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("status", "ok").strip().casefold() != "ok":
                continue
            identifier = row.get(id_field, "").strip()
            local_value = row.get("local_path", "").strip()
            if not identifier or not local_value:
                continue
            local_path = Path(local_value)
            if not local_path.is_absolute():
                working_relative = Path.cwd() / local_path
                local_path = working_relative if working_relative.exists() else source.parent / local_path
            if not local_path.exists():
                raise FileNotFoundError(f"Image manifest path does not exist: {local_path}")
            if identifier in result:
                raise ValueError(f"Duplicate image manifest identifier: {identifier}")
            result[identifier] = local_path.resolve()
    return result


def _write_npz(path: Path, identifiers: list[str], embeddings: np.ndarray) -> None:
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Feature identifiers must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(identifiers, dtype=str),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claims_path = Path(args.claims).resolve()
    evidence_path = Path(args.evidence).resolve()
    claim_rows = read_jsonl(claims_path)
    evidence_rows = read_jsonl(evidence_path)
    if not claim_rows or not evidence_rows:
        raise ValueError("Claims and evidence manifests must both be non-empty")

    text_ids = [str(row["claim_id"]) for row in claim_rows] + [
        str(row["evidence_id"]) for row in evidence_rows
    ]
    evidence_text = {str(row["evidence_id"]): str(row.get("title") or "") for row in evidence_rows}
    if args.evidence_text:
        for row in read_jsonl(args.evidence_text):
            identifier = str(row["evidence_id"])
            evidence_text[identifier] = str(row.get("text") or row.get("title") or "")
    texts = [str(row["claim_text"]) for row in claim_rows] + [
        evidence_text[str(row["evidence_id"])] for row in evidence_rows
    ]
    retrieval = config["retrieval"]
    text_encoder = BgeM3Encoder(
        retrieval["dense_text_model"], revision=retrieval["dense_text_revision"]
    )
    text_embeddings = text_encoder.encode(texts, batch_size=args.text_batch_size)

    claim_images = _image_rows(args.claim_image_manifest, "candidate_id")
    missing_claim_images = [
        str(row["claim_id"]) for row in claim_rows if str(row["claim_id"]) not in claim_images
    ]
    if missing_claim_images:
        raise ValueError(
            f"Missing verified local images for {len(missing_claim_images)} claims; "
            f"examples={missing_claim_images[:5]}"
        )
    evidence_images = (
        _image_rows(args.evidence_image_manifest, "evidence_id")
        if args.evidence_image_manifest
        else {}
    )
    image_ids = [str(row["claim_id"]) for row in claim_rows] + sorted(evidence_images)
    image_paths = [claim_images[str(row["claim_id"])] for row in claim_rows] + [
        evidence_images[identifier] for identifier in sorted(evidence_images)
    ]
    image_encoder = Siglip2ImageEncoder(
        retrieval["visual_model"],
        revision=retrieval["visual_model_revision"],
        device=args.device,
    )
    image_embeddings = image_encoder.encode(image_paths, batch_size=args.image_batch_size)

    output_root = Path(args.output_directory).resolve()
    text_path = output_root / "text_embeddings.npz"
    image_path = output_root / "image_embeddings.npz"
    _write_npz(text_path, text_ids, text_embeddings)
    _write_npz(image_path, image_ids, image_embeddings)
    metadata = {
        "status": "frozen_feature_store",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "claims": str(claims_path),
            "claims_sha256": _sha256(claims_path),
            "evidence": str(evidence_path),
            "evidence_sha256": _sha256(evidence_path),
            "claim_image_manifest": str(Path(args.claim_image_manifest).resolve()),
            "claim_image_manifest_sha256": _sha256(Path(args.claim_image_manifest).resolve()),
            "evidence_image_manifest": str(Path(args.evidence_image_manifest).resolve()) if args.evidence_image_manifest else None,
            "evidence_text": str(Path(args.evidence_text).resolve()) if args.evidence_text else None,
        },
        "text": {
            "model": retrieval["dense_text_model"],
            "revision": retrieval["dense_text_revision"],
            "count": len(text_ids),
            "dimension": int(text_embeddings.shape[1]),
            "archive": str(text_path),
            "archive_sha256": _sha256(text_path),
        },
        "image": {
            "model": retrieval["visual_model"],
            "revision": retrieval["visual_model_revision"],
            "count": len(image_ids),
            "dimension": int(image_embeddings.shape[1]),
            "evidence_image_count": len(evidence_images),
            "archive": str(image_path),
            "archive_sha256": _sha256(image_path),
        },
    }
    write_json(output_root / "feature_manifest.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
