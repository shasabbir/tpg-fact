from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from trace_scv.data.averimatec import (
    candidate_identifier,
    load_annotations,
    normalize_source_url,
    sha256_text,
)
from trace_scv.data.io import sha256_file
from trace_scv.external_retrieval import (
    ExternalRetrievalConfig,
    image_filename_text,
    rank_external_candidates,
)
from trace_scv.mocheg_features import (
    HashMochegEncoder,
    MochegEncoderConfig,
    TransformersMochegEncoder,
)


TOKEN_PATTERN = re.compile(r"[\w'-]+", flags=re.UNICODE)


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value)]


def _read_json_or_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"expected list in {path}")
        return [dict(row) for row in data if isinstance(row, Mapping)]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _build_store_index(store_root: Path) -> dict[tuple[str, int], list[tuple[str, Path]]]:
    index: dict[tuple[str, int], list[tuple[str, Path]]] = {}
    for path in store_root.rglob("*.json"):
        if not path.stem.isdigit():
            continue
        lowered = path.as_posix().casefold()
        split = "validation" if "val" in lowered else "train" if "train" in lowered else None
        if split is None:
            continue
        branch = "image_related" if "image_related" in lowered else "text_related"
        index.setdefault((split, int(path.stem)), []).append((branch, path))
    for key in index:
        index[key] = sorted(set(index[key]), key=lambda item: item[1].as_posix())
    return index


def _build_image_index(store_root: Path) -> dict[tuple[str, int], list[Path]]:
    index: dict[tuple[str, int], list[Path]] = {}
    accepted = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
    for path in store_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in accepted:
            continue
        lowered = path.as_posix().casefold()
        if "image_related_store_image" not in lowered or not path.parent.name.isdigit():
            continue
        split = "validation" if "val" in lowered else "train" if "train" in lowered else None
        if split is None:
            continue
        index.setdefault((split, int(path.parent.name)), []).append(path)
    for key in index:
        index[key] = sorted(set(index[key]), key=lambda path: path.as_posix())
    return index


def _content_segments(row: Mapping[str, object], maximum_characters: int = 1500) -> list[str]:
    value = row.get("url2text", row.get("scrape_content", row.get("text", "")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_segments = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw = str(value or "").strip()
        raw_segments = [raw] if raw else []
    segments: list[str] = []
    for raw in raw_segments:
        for start in range(0, len(raw), maximum_characters):
            chunk = raw[start : start + maximum_characters].strip()
            if chunk:
                segments.append(chunk)
    return segments


def _best_label_blind_chunk(row: Mapping[str, object], query_tokens: set[str]) -> str:
    segments = _content_segments(row)
    if not segments:
        return ""
    return max(
        enumerate(segments),
        key=lambda item: (
            len(set(_tokens(item[1])) & query_tokens),
            -item[0],
        ),
    )[1]


def _source_url(row: Mapping[str, object]) -> str:
    return str(row.get("url", row.get("source_url", row.get("link", ""))) or "")


def _lexical_features(
    claim: Mapping[str, object], candidates: Sequence[dict[str, object]]
) -> list[list[float]]:
    claim_tokens = set(_tokens(str(claim.get("claim_text", ""))))
    metadata = claim.get("metadata") if isinstance(claim.get("metadata"), Mapping) else {}
    transcription_tokens = set(_tokens(str(metadata.get("transcription", ""))))
    query_tokens = claim_tokens | transcription_tokens
    document_sets = [set(_tokens(str(row["content"]))) for row in candidates]
    frequencies = Counter(token for values in document_sets for token in (values & query_tokens))
    count = max(1, len(document_sets))

    def overlap(left: set[str], right: set[str]) -> float:
        return len(left & right) / len(left | right) if left or right else 0.0

    output: list[list[float]] = []
    for ordinal, (row, document) in enumerate(zip(candidates, document_sets, strict=True)):
        weighted = sum(math.log((count + 1) / (frequencies[token] + 1)) + 1.0 for token in query_tokens & document)
        parsed = urlsplit(str(row["normalized_source_url"]))
        output.append(
            [
                overlap(claim_tokens, document),
                overlap(query_tokens, document),
                overlap(transcription_tokens, document),
                weighted / max(1.0, math.sqrt(len(document))),
                1.0 / (1.0 + ordinal),
                float(row["branch"] == "image_related"),
                math.log1p(len(document)),
                float(parsed.path.casefold().endswith(".pdf")),
            ]
        )
    return output


def _candidate_rows(
    claim: Mapping[str, object],
    *,
    split: str,
    ordinal: int,
    store_index: Mapping[tuple[str, int], Sequence[tuple[str, Path]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw: list[dict[str, object]] = []
    files = list(store_index.get((split, ordinal), ()))
    metadata = claim.get("metadata") if isinstance(claim.get("metadata"), Mapping) else {}
    query_tokens = set(
        _tokens(
            " ".join(
                (
                    str(claim.get("claim_text", "")),
                    str(metadata.get("transcription", "")),
                )
            )
        )
    )
    for branch, path in files:
        for local_ordinal, source in enumerate(_read_json_or_jsonl(path)):
            url = _source_url(source)
            normalized = normalize_source_url(url)
            content = _best_label_blind_chunk(source, query_tokens)
            if not normalized or not content:
                continue
            raw.append(
                {
                    "source_url": url,
                    "normalized_source_url": normalized,
                    "content": content,
                    "branch": branch,
                    "source_file": path.as_posix(),
                    "source_ordinal": local_ordinal,
                    "retrieval_ordinal": len(raw),
                }
            )
    deduplicated: dict[tuple[str, str], dict[str, object]] = {}
    for row in raw:
        key = (str(row["normalized_source_url"]), str(row["branch"]))
        previous = deduplicated.get(key)
        if previous is None:
            deduplicated[key] = row
    candidates = sorted(
        deduplicated.values(),
        key=lambda row: int(row["retrieval_ordinal"]),
    )
    features = _lexical_features(claim, candidates)
    claim_id = str(claim["claim_id"])
    rows: list[dict[str, object]] = []
    for local_ordinal, (candidate, values) in enumerate(zip(candidates, features, strict=True)):
        content = str(candidate["content"])
        rows.append(
            {
                "split": split,
                "claim_id": claim_id,
                "claim_ordinal": ordinal,
                "evidence_id": candidate_identifier(
                    claim_id,
                    str(candidate["source_url"]),
                    str(candidate["branch"]),
                    content,
                    local_ordinal,
                ),
                "source_url": str(candidate["source_url"]),
                "normalized_source_url": str(candidate["normalized_source_url"]),
                "source_domain": (urlsplit(str(candidate["normalized_source_url"])).hostname or ""),
                "branch": str(candidate["branch"]),
                "modality": "text",
                "evaluation_id": f"url:{candidate['normalized_source_url']}",
                "content": content,
                "content_sha256": sha256_text(content),
                "content_word_count": len(content.split()),
                "features": values,
                "source_file": str(candidate["source_file"]),
                "source_ordinal": int(candidate["source_ordinal"]),
            }
        )
    audit = {
        "claim_id": claim_id,
        "store_files": len(files),
        "raw_candidates": len(raw),
        "deduplicated_candidates": len(rows),
        "duplicate_candidates": len(raw) - len(rows),
    }
    return rows, audit


def _image_candidate_rows(
    claim: Mapping[str, object],
    *,
    split: str,
    ordinal: int,
    image_index: Mapping[tuple[str, int], Sequence[Path]],
) -> list[dict[str, object]]:
    claim_id = str(claim["claim_id"])
    rows: list[dict[str, object]] = []
    for local_ordinal, path in enumerate(image_index.get((split, ordinal), ())):
        content = image_filename_text(path)
        rows.append(
            {
                "split": split,
                "claim_id": claim_id,
                "claim_ordinal": ordinal,
                "evidence_id": candidate_identifier(
                    claim_id,
                    path.name,
                    "image",
                    content,
                    local_ordinal,
                ),
                "source_url": "",
                "normalized_source_url": "",
                "source_domain": "",
                "branch": "image_candidate",
                "modality": "image",
                "evaluation_id": f"image:{path.name.casefold()}",
                "content": content,
                "content_sha256": sha256_text(content),
                "content_word_count": len(content.split()),
                "features": [0.0] * 8,
                "media_path": path.as_posix(),
                "source_file": path.as_posix(),
                "source_ordinal": local_ordinal,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-blind AVerImaTeC candidate rankings.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/public/averimatec/raw"))
    parser.add_argument("--store-root", type=Path, default=Path("data/public/averimatec/raw/knowledge_store/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/external_confirmation/averimatec/prepared"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument(
        "--retrieval-backend",
        choices=("lexical", "multimodal"),
        default="lexical",
        help="multimodal enables BM25/BGE/RRF/cross-encoder and actual-image SigLIP2 ranking",
    )
    parser.add_argument("--encoder", choices=("transformers", "hash"), default="transformers")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--claim-limit", type=int)
    parser.add_argument("--text-depth", type=int, default=100)
    parser.add_argument("--image-depth", type=int, default=20)
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite populated {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_prejoin: list[dict[str, object]] = []
    all_joined: list[dict[str, object]] = []
    claim_audits: list[dict[str, object]] = []
    full_gold: dict[str, list[str]] = {}
    claims_by_key: dict[str, dict[str, object]] = {}
    candidates_by_claim: dict[str, list[dict[str, object]]] = {}
    claim_inputs: list[dict[str, object]] = []
    store_index = _build_store_index(args.store_root)
    image_index = _build_image_index(args.store_root)
    split_files = {"train": "train.json", "validation": "val.json"}
    for split in args.splits:
        filename = split_files[split]
        claims = load_annotations(
            args.raw_root / filename, split=split, include_outcomes=False
        )
        if args.claim_limit is not None:
            if args.claim_limit < 1:
                raise ValueError("claim-limit must be positive")
            claims = claims[: args.claim_limit]
        for ordinal, claim in enumerate(claims):
            rows, audit = _candidate_rows(
                claim,
                split=split,
                ordinal=ordinal,
                store_index=store_index,
            )
            image_rows = _image_candidate_rows(
                claim,
                split=split,
                ordinal=ordinal,
                image_index=image_index,
            )
            rows.extend(image_rows)
            key = f"{split}:{claim['claim_id']}"
            claims_by_key[key] = claim
            candidates_by_claim[key] = rows
            metadata = claim.get("metadata") if isinstance(claim.get("metadata"), Mapping) else {}
            claim_inputs.append(
                {
                    "split": split,
                    "claim_id": str(claim["claim_id"]),
                    "claim_text": str(claim.get("claim_text", "")),
                    "transcription": str(metadata.get("transcription", "")),
                }
            )
            audit["split"] = split
            audit["image_candidates"] = len(image_rows)
            claim_audits.append(audit)

    retrieval_manifest: dict[str, object]
    if args.retrieval_backend == "multimodal":
        if args.encoder == "hash":
            encoder = HashMochegEncoder()
        else:
            encoder = TransformersMochegEncoder(
                MochegEncoderConfig(device=args.device)
            )
        all_prejoin, retrieval_manifest = rank_external_candidates(
            claim_inputs,
            candidates_by_claim,
            encoder=encoder,
            config=ExternalRetrievalConfig(
                text_depth=args.text_depth,
                image_depth=args.image_depth,
                rerank_depth=args.rerank_depth,
            ),
            include_cross_encoder=not args.skip_cross_encoder,
        )
    else:
        all_prejoin = [
            row
            for key in sorted(candidates_by_claim)
            for row in candidates_by_claim[key]
        ]
        retrieval_manifest = {
            "profile": "averimatec_lexical_access_audit",
            "scientific_results_allowed": False,
            "reason": "lexical-only preparation is not the prespecified external retrieval path",
            "claims": len(claim_inputs),
            "input_text_candidates": sum(row["modality"] == "text" for row in all_prejoin),
            "input_image_candidates": sum(row["modality"] == "image" for row in all_prejoin),
            "gold_questions_used_for_retrieval": False,
            "verdict_labels_accessed": False,
            "test_annotations_accessed": False,
        }

    prejoin_path = args.output_root / "rankings.prejoin.jsonl"
    joined_path = args.output_root / "candidate_records.jsonl"
    audit_path = args.output_root / "preparation.audit.json"
    _write_jsonl(prejoin_path, all_prejoin)
    prejoin_hash = sha256_file(prejoin_path)
    # Outcome-bearing annotation fields are joined only after the immutable
    # label-blind ranking exists and has been hashed.
    outcome_claims_by_key: dict[str, dict[str, object]] = {}
    for split in args.splits:
        outcome_claims = load_annotations(
            args.raw_root / split_files[split], split=split, include_outcomes=True
        )
        if args.claim_limit is not None:
            outcome_claims = outcome_claims[: args.claim_limit]
        outcome_claims_by_key.update(
            {f"{split}:{claim['claim_id']}": claim for claim in outcome_claims}
        )
    if set(outcome_claims_by_key) != set(claims_by_key):
        raise RuntimeError("sanitized and outcome-bearing annotation identities differ")
    prejoin_by_key: dict[str, list[dict[str, object]]] = {}
    for row in all_prejoin:
        prejoin_by_key.setdefault(f"{row['split']}:{row['claim_id']}", []).append(row)
    for key in sorted(claims_by_key):
        claim = outcome_claims_by_key[key]
        gold_targets = set(map(str, claim["gold_evidence_targets"]))
        full_gold[key] = sorted(gold_targets)
        for row in prejoin_by_key.get(key, ()):
            joined = dict(row)
            joined["gold"] = str(row["evaluation_id"]) in gold_targets
            joined["claim_label"] = str(claim["label"])
            all_joined.append(joined)
        audit = next(
            item
            for item in claim_audits
            if item["split"] == claim["split"] and item["claim_id"] == claim["claim_id"]
        )
        reachable = {str(row["evaluation_id"]) for row in prejoin_by_key.get(key, ())}
        audit["gold_targets"] = len(gold_targets)
        audit["reachable_gold_targets"] = len(gold_targets & reachable)
    _write_jsonl(joined_path, all_joined)
    (args.output_root / "full_gold_sources.json").write_text(
        json.dumps(full_gold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(args.output_root / "claim_audit.jsonl", claim_audits)
    audit = {
        "status": "pass",
        "prejoin_rankings": {"path": prejoin_path.as_posix(), "sha256": prejoin_hash, "rows": len(all_prejoin)},
        "gold_join_performed_after_prejoin_write_and_hash": True,
        "joined_records": {"path": joined_path.as_posix(), "sha256": sha256_file(joined_path), "rows": len(all_joined)},
        "claims": len(claim_audits),
        "claims_without_store_candidates": sum(
            not (int(row["deduplicated_candidates"]) + int(row["image_candidates"]))
            for row in claim_audits
        ),
        "claims_without_reachable_gold": sum(
            not int(row["reachable_gold_targets"]) for row in claim_audits
        ),
        "retrieval": retrieval_manifest,
        "exact_diagnostic_target_identity": "url:<normalized-url> or image:<released-basename>",
        "official_semantic_evidence_score_computed": False,
        "test_annotations_accessed": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
