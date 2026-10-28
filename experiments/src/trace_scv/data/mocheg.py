from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from trace_scv.data.io import sha256_file


@dataclass(frozen=True, slots=True)
class MochegMirror:
    key: str
    repo_id: str
    revision: str


MOCHEG_MIRRORS: dict[str, MochegMirror] = {
    "train": MochegMirror(
        key="train",
        repo_id="vcmt794/mocheg-train",
        revision="985e4206cf9990ff0a875473616be204cfbd09c3",
    ),
    "validation": MochegMirror(
        key="validation",
        repo_id="vcmt794/mocheg-val",
        revision="9dabf73ad41ddecafb3b525fb82a3de018e3e378",
    ),
    "test": MochegMirror(
        key="test",
        repo_id="vcmt794/mocheg-test",
        revision="a72342a67b6bb5e930f8a2aa5fc3b051690c0f74",
    ),
    "corpus": MochegMirror(
        key="corpus",
        repo_id="tonyvu1289/mocheg",
        revision="d9b15bf2b7d2be790c40efa960e8f05a58487ce2",
    ),
}

MOCHEG_METADATA_FILES = (
    "train.json",
    "img_evidence_qrels.csv",
    "text_evidence_qrels_article_level.csv",
    "text_evidence_qrels_sentence_level.csv",
    "load_dataset.py",
)

MOCHEG_EXPECTED: dict[str, dict[str, object]] = {
    "train": {
        "claims": 11_669,
        "text_evidence": 23_545,
        "gold_image_evidence": 8_927,
        "labels": {"supported": 3_826, "refuted": 4_542, "nei": 3_301},
    },
    "validation": {
        "claims": 1_490,
        "text_evidence": 4_067,
        "gold_image_evidence": 1_178,
        "labels": {"supported": 501, "refuted": 488, "nei": 501},
    },
    "corpus": {"rows": 91_822},
}


@dataclass(frozen=True, slots=True)
class MochegTextEvidence:
    evidence_id: str
    content: str


@dataclass(frozen=True, slots=True)
class MochegClaim:
    claim_id: str
    claim: str
    label: str
    text_evidence: tuple[MochegTextEvidence, ...]
    source_image_ids: tuple[str, ...]
    ruling_outline: str
    source_urls: tuple[str, ...]
    split: str


@dataclass(frozen=True, slots=True)
class MochegQrel:
    claim_id: str
    document_id: str
    relevant: bool
    evidence_id: str | None


@dataclass(frozen=True, slots=True)
class MochegCandidate:
    evidence_id: str
    modality: str
    gold: bool
    source_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MochegSentenceText:
    document_id: str
    source_claim_id: str
    source_document_id: str
    sentence_index: int
    content: str


def _as_list(value: object) -> list[object]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def normalize_mocheg_label(value: object) -> str:
    token = _clean(value).lower().replace("_", " ").replace("-", " ")
    mapping = {
        "support": "supported",
        "supported": "supported",
        "refute": "refuted",
        "refuted": "refuted",
        "nei": "nei",
        "not enough information": "nei",
        "not enough info": "nei",
    }
    try:
        return mapping[token]
    except KeyError as error:
        raise ValueError(f"unsupported MOCHEG label: {value!r}") from error


def _single_consistent(values: Iterable[str], *, field: str, claim_id: str) -> str:
    unique = {value for value in values if value}
    if len(unique) != 1:
        raise ValueError(
            f"claim {claim_id} has {len(unique)} distinct non-empty {field} values"
        )
    return next(iter(unique))


def load_mocheg_claims(path: Path, *, split: str) -> list[MochegClaim]:
    """Normalize both released MOCHEG JSON layouts to one row per claim."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"MOCHEG JSON must contain a list: {path}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"MOCHEG rows must be objects: {path}")
        claim_id = _clean(row.get("claim_id"))
        if not claim_id:
            raise ValueError(f"MOCHEG row has no claim_id: {path}")
        grouped[claim_id].append(row)

    claims: list[MochegClaim] = []
    for claim_id in sorted(grouped, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        claim_rows = grouped[claim_id]
        claim = _single_consistent(
            (_clean(row.get("claim")) for row in claim_rows),
            field="claim",
            claim_id=claim_id,
        )
        label = _single_consistent(
            (normalize_mocheg_label(row.get("label")) for row in claim_rows),
            field="label",
            claim_id=claim_id,
        )
        ruling_values = {_clean(row.get("ruling_outline")) for row in claim_rows}
        ruling_values.discard("")
        if len(ruling_values) > 1:
            raise ValueError(f"claim {claim_id} has conflicting ruling outlines")

        evidence_by_id: dict[str, str] = {}
        image_ids: set[str] = set()
        source_urls: set[str] = set()
        for row in claim_rows:
            evidence_ids = [_clean(value) for value in _as_list(row.get("evidence_id"))]
            evidence_texts = [_clean(value) for value in _as_list(row.get("text_evidence"))]
            if len(evidence_ids) != len(evidence_texts):
                raise ValueError(
                    f"claim {claim_id} has {len(evidence_ids)} evidence IDs but "
                    f"{len(evidence_texts)} evidence texts"
                )
            for evidence_id, content in zip(evidence_ids, evidence_texts, strict=True):
                if not evidence_id:
                    raise ValueError(f"claim {claim_id} has an empty text evidence ID")
                # The public train mirror retains 4,556 official evidence IDs whose
                # converted text field is blank. Preserve the identifiers so the
                # omission remains auditable; the preparation pipeline later
                # recovers content through the official qrels/corpus mapping.
                existing = evidence_by_id.get(evidence_id)
                if existing is not None and existing != content:
                    raise ValueError(
                        f"claim {claim_id} evidence {evidence_id} has conflicting text"
                    )
                evidence_by_id[evidence_id] = content
            image_ids.update(
                _clean(value) for value in _as_list(row.get("image")) if _clean(value)
            )
            source_urls.update(
                _clean(value)
                for value in _as_list(row.get("snopes_url"))
                if _clean(value)
            )

        evidence = tuple(
            MochegTextEvidence(evidence_id=evidence_id, content=evidence_by_id[evidence_id])
            for evidence_id in sorted(
                evidence_by_id,
                key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
            )
        )
        claims.append(
            MochegClaim(
                claim_id=claim_id,
                claim=claim,
                label=label,
                text_evidence=evidence,
                source_image_ids=tuple(sorted(image_ids)),
                ruling_outline=next(iter(ruling_values), ""),
                source_urls=tuple(sorted(source_urls)),
                split=split,
            )
        )
    return claims


def load_mocheg_qrels(path: Path) -> list[MochegQrel]:
    rows: list[MochegQrel] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"TOPIC", "DOCUMENT#", "RELEVANCY"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"invalid MOCHEG qrel header in {path}")
        for row in reader:
            relevance = _clean(row.get("RELEVANCY"))
            if relevance not in {"0", "1"}:
                raise ValueError(f"invalid qrel relevance {relevance!r} in {path}")
            claim_id = _clean(row.get("TOPIC"))
            document_id = _clean(row.get("DOCUMENT#"))
            if not claim_id or not document_id:
                raise ValueError(f"empty qrel identifier in {path}")
            evidence_id = _clean(row.get("evidence_id")) or None
            rows.append(
                MochegQrel(
                    claim_id=claim_id,
                    document_id=document_id,
                    relevant=relevance == "1",
                    evidence_id=evidence_id,
                )
            )
    return rows


def parse_mocheg_sentence_document_id(document_id: str) -> tuple[str, str, int]:
    """Parse the corpus-qualified sentence identifier used by MOCHEG qrels."""
    parts = document_id.rsplit("-", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid MOCHEG sentence document ID: {document_id!r}")
    try:
        sentence_index = int(parts[2])
    except ValueError as error:
        raise ValueError(
            f"invalid MOCHEG sentence index in document ID: {document_id!r}"
        ) from error
    if sentence_index < 0:
        raise ValueError(f"negative MOCHEG sentence index: {document_id!r}")
    return parts[0], parts[1], sentence_index


def reconstruct_mocheg_sentence_texts(
    corpus_path: Path,
    qrels: Iterable[MochegQrel],
    *,
    sentence_splitter: Callable[[str], list[str]],
) -> tuple[dict[str, MochegSentenceText], dict[str, object]]:
    """Rebuild only qrel-referenced sentences using the official corpus recipe.

    The official code removes literal ``<p>`` tags, applies NLTK sentence
    tokenization, numbers retained sentences from zero, and lets a later
    duplicate corpus row overwrite an earlier sentence with the same ID.  This
    implementation reproduces that behavior without materializing the full
    sentence corpus in memory.
    """
    required_ids = {row.document_id for row in qrels}
    required_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    parsed_ids: dict[str, tuple[str, str, int]] = {}
    for document_id in required_ids:
        parsed = parse_mocheg_sentence_document_id(document_id)
        parsed_ids[document_id] = parsed
        required_by_source[parsed[:2]].add(document_id)

    csv.field_size_limit(2**31 - 1)
    recovered_content: dict[str, str] = {}
    seen_source_hashes: dict[tuple[str, str], str] = {}
    matched_source_rows = 0
    duplicate_source_rows = 0
    conflicting_source_rows = 0
    duplicate_sentence_assignments = 0
    conflicting_sentence_assignments = 0
    with corpus_path.open(
        encoding="utf-8-sig", newline="", errors="replace"
    ) as handle:
        reader = csv.DictReader(handle)
        required_columns = {"claim_id", "relevant_document_id", "Origin Document"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"invalid MOCHEG Corpus3 header in {corpus_path}")
        for row in reader:
            source_key = (
                _clean(row.get("claim_id")),
                _clean(row.get("relevant_document_id")),
            )
            target_ids = required_by_source.get(source_key)
            if not target_ids:
                continue
            raw_content = str(row.get("Origin Document") or "")
            matched_source_rows += 1
            content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            if source_key in seen_source_hashes:
                duplicate_source_rows += 1
                conflicting_source_rows += int(
                    seen_source_hashes[source_key] != content_hash
                )
            seen_source_hashes[source_key] = content_hash

            cleaned_content = raw_content.replace("<p>", "").replace("</p>", "")
            retained_index = 0
            for sentence in sentence_splitter(cleaned_content):
                if sentence == "N/A":
                    continue
                document_id = f"{source_key[0]}-{source_key[1]}-{retained_index}"
                retained_index += 1
                if document_id not in target_ids:
                    continue
                if document_id in recovered_content:
                    duplicate_sentence_assignments += 1
                    conflicting_sentence_assignments += int(
                        recovered_content[document_id] != sentence
                    )
                recovered_content[document_id] = sentence

    records = {
        document_id: MochegSentenceText(
            document_id=document_id,
            source_claim_id=parsed_ids[document_id][0],
            source_document_id=parsed_ids[document_id][1],
            sentence_index=parsed_ids[document_id][2],
            content=content,
        )
        for document_id, content in recovered_content.items()
    }
    missing_ids = sorted(required_ids - records.keys())
    missing_source_keys = sorted(set(required_by_source) - set(seen_source_hashes))
    audit: dict[str, object] = {
        "required_sentence_ids": len(required_ids),
        "recovered_sentence_ids": len(records),
        "missing_sentence_ids": len(missing_ids),
        "missing_document_ids": missing_ids,
        "referenced_source_documents": len(required_by_source),
        "matched_source_rows": matched_source_rows,
        "missing_source_documents": [list(key) for key in missing_source_keys],
        "duplicate_source_rows": duplicate_source_rows,
        "conflicting_source_rows": conflicting_source_rows,
        "duplicate_sentence_assignments": duplicate_sentence_assignments,
        "conflicting_sentence_assignments": conflicting_sentence_assignments,
        "duplicate_resolution": "last sentence assignment wins, matching official loader",
    }
    return records, audit


def _write_sentence_texts(
    path: Path, records: Mapping[str, MochegSentenceText]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document_id in sorted(records):
            handle.write(
                json.dumps(asdict(records[document_id]), ensure_ascii=False) + "\n"
            )
    return len(records)


def _nltk_sentence_splitter(
    nltk_data_root: Path,
) -> tuple[Callable[[str], list[str]], dict[str, str]]:
    """Load the pinned NLTK tokenizer and record its local resource identity."""
    try:
        import nltk
    except ImportError as error:
        raise RuntimeError(
            "NLTK is required for MOCHEG sentence reconstruction; install the data extra"
        ) from error

    nltk_data_root.mkdir(parents=True, exist_ok=True)
    root_text = str(nltk_data_root.resolve())
    if root_text not in nltk.data.path:
        nltk.data.path.insert(0, root_text)
    try:
        resource = nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        if not nltk.download("punkt_tab", download_dir=root_text, quiet=True):
            raise RuntimeError("failed to download the NLTK punkt_tab resource")
        resource = nltk.data.find("tokenizers/punkt_tab/english")

    resource_path = Path(str(resource))
    digest = hashlib.sha256()
    for path in sorted(item for item in resource_path.rglob("*") if item.is_file()):
        digest.update(path.relative_to(resource_path).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())

    def split(text: str) -> list[str]:
        return list(nltk.tokenize.sent_tokenize(text, language="english"))

    return split, {
        "library": "nltk",
        "version": nltk.__version__,
        "resource": "punkt_tab/english",
        "resource_sha256": digest.hexdigest(),
    }


def prepare_mocheg_sentence_text_candidates(
    dataset_root: Path,
    output_root: Path,
    *,
    nltk_data_root: Path | None = None,
) -> dict[str, object]:
    """Create a complete train/development qrel-to-sentence lookup artifact."""
    assert_mocheg_development_sealed(dataset_root)
    metadata_root = dataset_root / "metadata"
    split_qrels = {
        "train": load_mocheg_qrels(
            metadata_root / "train" / "text_evidence_qrels_sentence_level.csv"
        ),
        "validation": load_mocheg_qrels(
            metadata_root / "val" / "text_evidence_qrels_sentence_level.csv"
        ),
    }
    tokenizer, tokenizer_identity = _nltk_sentence_splitter(
        nltk_data_root or dataset_root / "cache" / "nltk_data"
    )
    records, audit = reconstruct_mocheg_sentence_texts(
        metadata_root / "corpus" / "Corpus3.csv",
        [row for rows in split_qrels.values() for row in rows],
        sentence_splitter=tokenizer,
    )
    split_coverage: dict[str, dict[str, int]] = {}
    for split, rows in split_qrels.items():
        required = {row.document_id for row in rows}
        recovered = required & records.keys()
        split_coverage[split] = {
            "required_sentence_ids": len(required),
            "recovered_sentence_ids": len(recovered),
            "missing_sentence_ids": len(required - recovered),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "sentence_text.jsonl"
    _write_sentence_texts(output_path, records)
    result: dict[str, object] = {
        **audit,
        "split_coverage": split_coverage,
        "tokenizer": tokenizer_identity,
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "formal_test_annotations_accessed": False,
        "status": "pass" if not audit["missing_sentence_ids"] else "failed",
    }
    audit_path = output_root / "sentence_text_audit.json"
    audit_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if result["status"] != "pass":
        raise RuntimeError(
            "MOCHEG sentence-text reconstruction is incomplete; inspect "
            f"{audit_path}"
        )
    return result


def _write_claims(path: Path, claims: Iterable[MochegClaim]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(asdict(claim), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_mocheg_candidate_sets(
    claims: Iterable[MochegClaim],
    *,
    text_qrels: Iterable[MochegQrel],
    image_qrels: Iterable[MochegQrel],
) -> dict[str, tuple[MochegCandidate, ...]]:
    """Create one deduplicated, modality-qualified candidate set per claim."""
    claim_ids = {claim.claim_id for claim in claims}
    grouped: dict[str, dict[str, MochegCandidate]] = {
        claim_id: {} for claim_id in claim_ids
    }
    for modality, rows in (("text", text_qrels), ("image", image_qrels)):
        for row in rows:
            if row.claim_id not in claim_ids:
                raise ValueError(
                    f"qrel topic {row.claim_id} is absent from normalized claims"
                )
            evidence_id = f"{modality}:{row.document_id}"
            existing = grouped[row.claim_id].get(evidence_id)
            source_ids = tuple(sorted({row.evidence_id} - {None}))
            candidate = MochegCandidate(
                evidence_id=evidence_id,
                modality=modality,
                gold=row.relevant,
                source_evidence_ids=source_ids,
            )
            if existing is None:
                grouped[row.claim_id][evidence_id] = candidate
                continue
            if existing.gold != candidate.gold:
                raise ValueError(
                    f"candidate {evidence_id} for claim {row.claim_id} has conflicting relevance"
                )
            grouped[row.claim_id][evidence_id] = MochegCandidate(
                evidence_id=evidence_id,
                modality=modality,
                gold=existing.gold,
                source_evidence_ids=tuple(
                    sorted({*existing.source_evidence_ids, *candidate.source_evidence_ids})
                ),
            )
    return {
        claim_id: tuple(
            sorted(rows.values(), key=lambda row: (row.modality, row.evidence_id))
        )
        for claim_id, rows in grouped.items()
    }


def _write_candidate_sets(
    path: Path,
    claims: Iterable[MochegClaim],
    candidate_sets: Mapping[str, tuple[MochegCandidate, ...]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(
                json.dumps(
                    {
                        "claim_id": claim.claim_id,
                        "label": claim.label,
                        "candidates": [
                            asdict(candidate)
                            for candidate in candidate_sets[claim.claim_id]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def _candidate_stats(
    candidate_sets: Mapping[str, tuple[MochegCandidate, ...]],
) -> dict[str, object]:
    candidates = [
        candidate
        for claim_candidates in candidate_sets.values()
        for candidate in claim_candidates
    ]
    modality_counts = Counter(candidate.modality for candidate in candidates)
    gold_modality_counts = Counter(
        candidate.modality for candidate in candidates if candidate.gold
    )
    claims_with_candidates = sum(bool(rows) for rows in candidate_sets.values())
    return {
        "claims": len(candidate_sets),
        "claims_with_candidates": claims_with_candidates,
        "claims_without_candidates": len(candidate_sets) - claims_with_candidates,
        "candidates": len(candidates),
        "gold_candidates": sum(candidate.gold for candidate in candidates),
        "modality_counts": dict(sorted(modality_counts.items())),
        "gold_modality_counts": dict(sorted(gold_modality_counts.items())),
    }


def assert_mocheg_development_sealed(dataset_root: Path) -> None:
    """Reject formal-test annotations in a development data root."""
    candidates = (
        dataset_root / "test",
        dataset_root / "mocheg-test",
        dataset_root / "metadata" / "test",
        dataset_root / "metadata" / "mocheg-test",
    )
    present = [str(path) for path in candidates if path.exists()]
    if present:
        raise RuntimeError(
            "MOCHEG development root contains formal-test material: " + ", ".join(present)
        )


def _qrel_stats(rows: list[MochegQrel]) -> dict[str, int]:
    positive = [row for row in rows if row.relevant]
    return {
        "rows": len(rows),
        "claims": len({row.claim_id for row in rows}),
        "documents": len({row.document_id for row in rows}),
        "positive_rows": len(positive),
        "negative_rows": len(rows) - len(positive),
        "gold_evidence_ids": len(
            {row.evidence_id for row in positive if row.evidence_id is not None}
        ),
    }


def _corpus_stats(path: Path) -> dict[str, int]:
    csv.field_size_limit(2**31 - 1)
    rows = 0
    claim_ids: set[str] = set()
    document_ids: set[str] = set()
    composite_hashes: dict[tuple[str, str], str] = {}
    duplicate_composite_rows = 0
    conflicting_composite_rows = 0
    empty_documents = 0
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required = {"claim_id", "relevant_document_id", "Origin Document"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"invalid MOCHEG Corpus3 header in {path}")
        for row in reader:
            rows += 1
            claim_id = _clean(row.get("claim_id"))
            document_id = _clean(row.get("relevant_document_id"))
            content = _clean(row.get("Origin Document"))
            claim_ids.add(claim_id)
            document_ids.add(document_id)
            empty_documents += int(not content)
            key = (claim_id, document_id)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if key in composite_hashes:
                duplicate_composite_rows += 1
                conflicting_composite_rows += int(composite_hashes[key] != content_hash)
            else:
                composite_hashes[key] = content_hash
    return {
        "rows": rows,
        "claim_ids": len(claim_ids),
        "document_ids": len(document_ids),
        "unique_claim_document_keys": len(composite_hashes),
        "duplicate_claim_document_rows": duplicate_composite_rows,
        "conflicting_claim_document_rows": conflicting_composite_rows,
        "empty_documents": empty_documents,
    }


def _claim_stats(claims: list[MochegClaim]) -> dict[str, object]:
    return {
        "claims": len(claims),
        "text_evidence": sum(len(claim.text_evidence) for claim in claims),
        "blank_text_evidence": sum(
            not evidence.content
            for claim in claims
            for evidence in claim.text_evidence
        ),
        "source_image_references": sum(len(claim.source_image_ids) for claim in claims),
        "unique_source_image_ids": len(
            {image_id for claim in claims for image_id in claim.source_image_ids}
        ),
        "labels": dict(sorted(Counter(claim.label for claim in claims).items())),
    }


def _required_metadata_paths(dataset_root: Path) -> dict[str, Path]:
    metadata = dataset_root / "metadata"
    return {
        **{
            f"train/{name}": metadata / "train" / name
            for name in MOCHEG_METADATA_FILES
        },
        **{
            f"validation/{name}": metadata / "val" / name
            for name in MOCHEG_METADATA_FILES
        },
        "corpus/Corpus3.csv": metadata / "corpus" / "Corpus3.csv",
    }


def run_mocheg_development_preflight(
    dataset_root: Path,
    output_root: Path,
    *,
    expected: Mapping[str, Mapping[str, object]] = MOCHEG_EXPECTED,
) -> dict[str, object]:
    """Audit and normalize only the MOCHEG train/development release."""
    assert_mocheg_development_sealed(dataset_root)
    required_paths = _required_metadata_paths(dataset_root)
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MOCHEG development metadata: {missing}")

    metadata = dataset_root / "metadata"
    train_claims = load_mocheg_claims(metadata / "train" / "train.json", split="train")
    validation_claims = load_mocheg_claims(
        metadata / "val" / "train.json", split="validation"
    )
    train_ids = {claim.claim_id for claim in train_claims}
    validation_ids = {claim.claim_id for claim in validation_claims}

    split_results: dict[str, dict[str, object]] = {}
    for split, folder, claims in (
        ("train", "train", train_claims),
        ("validation", "val", validation_claims),
    ):
        image_qrels = load_mocheg_qrels(metadata / folder / "img_evidence_qrels.csv")
        article_qrels = load_mocheg_qrels(
            metadata / folder / "text_evidence_qrels_article_level.csv"
        )
        sentence_qrels = load_mocheg_qrels(
            metadata / folder / "text_evidence_qrels_sentence_level.csv"
        )
        claim_stats = _claim_stats(claims)
        image_stats = _qrel_stats(image_qrels)
        candidate_sets = build_mocheg_candidate_sets(
            claims,
            text_qrels=sentence_qrels,
            image_qrels=image_qrels,
        )
        known_ids = {claim.claim_id for claim in claims}
        unknown_topics = sorted(
            {
                row.claim_id
                for row in [*image_qrels, *article_qrels, *sentence_qrels]
                if row.claim_id not in known_ids
            }
        )
        expected_split = expected[split]
        checks = {
            "claim_count": claim_stats["claims"] == expected_split["claims"],
            "text_evidence_count": (
                claim_stats["text_evidence"] == expected_split["text_evidence"]
            ),
            "gold_image_evidence_count": (
                image_stats["gold_evidence_ids"]
                == expected_split["gold_image_evidence"]
            ),
            "label_counts": claim_stats["labels"] == expected_split["labels"],
            "qrel_topics_known": not unknown_topics,
        }
        split_results[split] = {
            "claims": claim_stats,
            "image_qrels": image_stats,
            "article_qrels": _qrel_stats(article_qrels),
            "sentence_qrels": _qrel_stats(sentence_qrels),
            "candidate_sets": _candidate_stats(candidate_sets),
            "unknown_qrel_topics": unknown_topics[:100],
            "checks": checks,
        }
        _write_candidate_sets(
            output_root / f"{split}_candidates.jsonl", claims, candidate_sets
        )

    corpus = _corpus_stats(metadata / "corpus" / "Corpus3.csv")
    global_checks = {
        "train_validation_claim_ids_disjoint": not (train_ids & validation_ids),
        "corpus_row_count": corpus["rows"] == expected["corpus"]["rows"],
        "corpus_has_no_empty_documents": corpus["empty_documents"] == 0,
        "formal_test_annotations_absent": True,
    }
    all_checks = [
        *global_checks.values(),
        *(
            passed
            for split in split_results.values()
            for passed in split["checks"].values()
        ),
    ]

    files = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(required_paths.items())
    }
    media_manifest_path = dataset_root / "media" / "pilot_media_manifest.json"
    sentence_audit_path = output_root / "sentence_text_audit.json"
    packet_manifest_path = output_root / "pilot_packet_manifest.json"
    downstream_gates: dict[str, bool] = {
        "media": False,
        "sentence_text": False,
        "pilot_packets": False,
    }
    if media_manifest_path.is_file():
        downstream_gates["media"] = bool(
            json.loads(media_manifest_path.read_text(encoding="utf-8")).get(
                "smoke_pass"
            )
        )
    if sentence_audit_path.is_file():
        downstream_gates["sentence_text"] = (
            json.loads(sentence_audit_path.read_text(encoding="utf-8")).get("status")
            == "pass"
        )
    if packet_manifest_path.is_file():
        downstream_gates["pilot_packets"] = (
            json.loads(packet_manifest_path.read_text(encoding="utf-8")).get("status")
            == "pass"
        )
    pilot_ready = all(all_checks) and all(downstream_gates.values())
    pending_gates = [name for name, passed in downstream_gates.items() if not passed]
    result: dict[str, object] = {
        "dataset": "MOCHEG",
        "profile": "development",
        "status": (
            "pilot_packet_pass"
            if pilot_ready
            else "metadata_pass_downstream_pending"
            if all(all_checks)
            else "failed"
        ),
        "metadata_gate_pass": all(all_checks),
        "pilot_ready": pilot_ready,
        "downstream_gates": downstream_gates,
        "reason_pilot_not_ready": (
            None
            if pilot_ready
            else "pending gates: " + ", ".join(pending_gates)
            if all(all_checks)
            else "metadata gate failed"
        ),
        "mirrors": {key: asdict(value) for key, value in MOCHEG_MIRRORS.items()},
        "formal_test": {
            "annotations_downloaded": False,
            "metrics_computed": False,
            "sealed_revision": MOCHEG_MIRRORS["test"].revision,
        },
        "splits": split_results,
        "corpus": corpus,
        "global_checks": global_checks,
        "files": files,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_claims(output_root / "train_claims.jsonl", train_claims)
    _write_claims(output_root / "validation_claims.jsonl", validation_claims)
    (output_root / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not result["metadata_gate_pass"]:
        raise RuntimeError(
            "MOCHEG development metadata gate failed; inspect preflight.json"
        )
    return result


def _deterministic_claim_sample(
    claims: Iterable[MochegClaim], *, limit: int | None, seed: int
) -> list[MochegClaim]:
    claims = list(claims)
    if limit is None or limit >= len(claims):
        return sorted(claims, key=lambda claim: claim.claim_id)
    return sorted(
        claims,
        key=lambda claim: hashlib.sha256(
            f"{seed}:{claim.split}:{claim.claim_id}".encode("utf-8")
        ).hexdigest(),
    )[:limit]


def _audit_downloaded_images(
    media_root: Path, downloaded: list[dict[str, object]]
) -> dict[str, object]:
    from PIL import Image

    readable = 0
    unreadable: list[dict[str, str]] = []
    widths: list[int] = []
    heights: list[int] = []
    hashes: list[str] = []
    for row in downloaded:
        path = media_root / str(row["mirror"]) / str(row["relative_path"])
        hashes.append(str(row["sha256"]))
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"invalid dimensions {width}x{height}")
                widths.append(width)
                heights.append(height)
            readable += 1
        except Exception as error:
            unreadable.append({"path": str(path), "error": str(error)})
    return {
        "files": len(downloaded),
        "readable": readable,
        "unreadable": unreadable,
        "unique_sha256": len(set(hashes)),
        "exact_duplicate_occurrences": len(hashes) - len(set(hashes)),
        "minimum_width": min(widths, default=None),
        "maximum_width": max(widths, default=None),
        "minimum_height": min(heights, default=None),
        "maximum_height": max(heights, default=None),
    }


def _classify_missing_media(
    *,
    claims: Iterable[MochegClaim],
    qrels: Iterable[MochegQrel],
    missing_names: Iterable[str],
) -> dict[str, object]:
    missing = set(missing_names)
    claim_rows = list(claims)
    qrel_rows = list(qrels)
    missing_candidates = [row for row in qrel_rows if row.document_id in missing]
    missing_evidence_references = [
        row for row in qrel_rows if row.evidence_id and row.evidence_id in missing
    ]
    source_image_ids = {
        image_id for claim in claim_rows for image_id in claim.source_image_ids
    }
    missing_source_images = sorted(source_image_ids & missing)
    positive_candidates = [row for row in missing_candidates if row.relevant]
    return {
        "missing_names": len(missing),
        "missing_candidate_rows": len(missing_candidates),
        "missing_positive_candidate_rows": len(positive_candidates),
        "missing_negative_candidate_rows": sum(
            not row.relevant for row in missing_candidates
        ),
        "missing_gold_evidence_references": len(missing_evidence_references),
        "missing_gold_evidence_ids": len(
            {row.evidence_id for row in missing_evidence_references}
        ),
        "missing_source_image_ids": missing_source_images,
        "affected_claims": len({row.claim_id for row in missing_candidates}),
        "affected_positive_claims": len(
            {row.claim_id for row in positive_candidates}
        ),
        "gold_complete": (
            not positive_candidates
            and not missing_evidence_references
            and not missing_source_images
        ),
    }


def download_mocheg_development_metadata(dataset_root: Path) -> Path:
    """Download pinned metadata/corpus without ever accessing the test mirror."""
    from trace_scv.data.download import _download_hf_file

    assert_mocheg_development_sealed(dataset_root)
    metadata = dataset_root / "metadata"
    for split, folder in (("train", "train"), ("validation", "val")):
        mirror = MOCHEG_MIRRORS[split]
        target = metadata / folder
        target.mkdir(parents=True, exist_ok=True)
        for name in MOCHEG_METADATA_FILES:
            if not (target / name).is_file():
                _download_hf_file(
                    repo_id=mirror.repo_id,
                    revision=mirror.revision,
                    filename=name,
                    target=target,
                )
    corpus_target = metadata / "corpus"
    corpus_target.mkdir(parents=True, exist_ok=True)
    corpus = MOCHEG_MIRRORS["corpus"]
    if not (corpus_target / "Corpus3.csv").is_file():
        _download_hf_file(
            repo_id=corpus.repo_id,
            revision=corpus.revision,
            filename="Corpus3.csv",
            target=corpus_target,
        )
    assert_mocheg_development_sealed(dataset_root)
    return metadata


def _mirror_image_index(mirror: MochegMirror) -> dict[str, str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id=mirror.repo_id,
        repo_type="dataset",
        revision=mirror.revision,
    )
    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for relative in files:
        if not relative.startswith("images/"):
            continue
        name = Path(relative).name
        if name in index and index[name] != relative:
            ambiguous.add(name)
        else:
            index[name] = relative
    for name in ambiguous:
        index.pop(name, None)
    return index


def _portable_media_path(
    media_root: Path, mirror_key: str, source_relative_path: str
) -> Path:
    """Map repository paths to Windows-safe, deterministic local object names."""
    suffix = Path(source_relative_path).suffix.lower()
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        suffix = ".bin"
    object_id = hashlib.sha256(source_relative_path.encode("utf-8")).hexdigest()
    return media_root / mirror_key / "objects" / f"{object_id}{suffix}"


def _download_mocheg_media_object(
    *,
    mirror: MochegMirror,
    mirror_key: str,
    source_relative_path: str,
    media_root: Path,
    token: str | None,
    attempts: int = 5,
) -> Path:
    """Download one image without exposing repository filenames to Windows paths."""
    import shutil
    import time
    from urllib.error import HTTPError, URLError
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    target = _portable_media_path(media_root, mirror_key, source_relative_path)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    encoded_path = quote(source_relative_path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{mirror.repo_id}/resolve/"
        f"{mirror.revision}/{encoded_path}"
    )
    headers = {"User-Agent": "trace-scv/0.4.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(attempts):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=180) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size <= 0:
                raise RuntimeError(f"empty media response for {source_relative_path}")
            partial.replace(target)
            return target
        except HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt + 1 >= attempts:
                partial.unlink(missing_ok=True)
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt + 1 >= attempts:
                partial.unlink(missing_ok=True)
                raise
        time.sleep(min(30.0, 2**attempt))
    raise RuntimeError(f"unreachable download state for {source_relative_path}")


def download_mocheg_pilot_media(
    dataset_root: Path,
    *,
    max_train_claims: int = 200,
    max_validation_claims: int = 200,
    seed: int = 13,
    max_workers: int = 8,
) -> dict[str, object]:
    """Fetch bounded train/dev image candidates while keeping test annotations sealed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from trace_scv.data.download import _hf_token

    assert_mocheg_development_sealed(dataset_root)
    metadata = dataset_root / "metadata"
    claims_by_split = {
        "train": load_mocheg_claims(metadata / "train" / "train.json", split="train"),
        "validation": load_mocheg_claims(
            metadata / "val" / "train.json", split="validation"
        ),
    }
    folders = {"train": "train", "validation": "val"}
    limits = {"train": max_train_claims, "validation": max_validation_claims}
    selected: dict[str, list[MochegClaim]] = {
        split: _deterministic_claim_sample(rows, limit=limits[split], seed=seed)
        for split, rows in claims_by_split.items()
    }
    needed: dict[str, set[str]] = {"train": set(), "validation": set()}
    image_qrels_by_split: dict[str, list[MochegQrel]] = {}
    for split, claims in selected.items():
        claim_ids = {claim.claim_id for claim in claims}
        needed[split].update(
            image_id for claim in claims for image_id in claim.source_image_ids
        )
        qrels = load_mocheg_qrels(
            metadata / folders[split] / "img_evidence_qrels.csv"
        )
        image_qrels_by_split[split] = [
            row for row in qrels if row.claim_id in claim_ids
        ]
        for row in image_qrels_by_split[split]:
            needed[split].add(row.document_id)
            if row.evidence_id:
                needed[split].add(row.evidence_id)

    indexes = {
        split: _mirror_image_index(MOCHEG_MIRRORS[split])
        for split in ("train", "validation")
    }
    media_root = dataset_root / "media"
    missing: dict[str, list[str]] = {"train": [], "validation": []}
    requested_files: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("train", "validation"):
        preferred = (split, "validation" if split == "train" else "train")
        for name in sorted(needed[split]):
            location: tuple[str, str] | None = None
            for mirror_key in preferred:
                relative = indexes[mirror_key].get(name)
                if relative:
                    location = (mirror_key, relative)
                    break
            if location is None:
                missing[split].append(name)
                continue
            mirror_key, relative = location
            requested_files[(mirror_key, relative)].add(split)

    token = _hf_token()
    resolved_paths: dict[tuple[str, str], Path] = {}
    pending: list[tuple[str, str]] = []
    for mirror_key, relative in sorted(requested_files):
        legacy_path = media_root / mirror_key / relative
        try:
            legacy_available = legacy_path.is_file() and legacy_path.stat().st_size > 0
        except OSError:
            legacy_available = False
        if legacy_available:
            resolved_paths[(mirror_key, relative)] = legacy_path
            continue
        object_path = _portable_media_path(media_root, mirror_key, relative)
        if object_path.is_file() and object_path.stat().st_size > 0:
            resolved_paths[(mirror_key, relative)] = object_path
        else:
            pending.append((mirror_key, relative))

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                _download_mocheg_media_object,
                mirror=MOCHEG_MIRRORS[mirror_key],
                mirror_key=mirror_key,
                source_relative_path=relative,
                media_root=media_root,
                token=token,
            ): (mirror_key, relative)
            for mirror_key, relative in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            resolved_paths[key] = future.result()

    downloaded: list[dict[str, object]] = []
    for (mirror_key, relative), requested_splits in sorted(requested_files.items()):
        target_path = resolved_paths.get((mirror_key, relative))
        if target_path is None:
            for split in requested_splits:
                missing[split].append(Path(relative).name)
            continue
        local_relative = target_path.relative_to(media_root / mirror_key).as_posix()
        if not target_path.is_file():
            for split in requested_splits:
                missing[split].append(Path(relative).name)
            continue
        downloaded.append(
            {
                "requested_splits": sorted(requested_splits),
                "mirror": mirror_key,
                "source_relative_path": relative,
                "relative_path": local_relative,
                "bytes": target_path.stat().st_size,
                "sha256": sha256_file(target_path),
            }
        )

    image_audit = _audit_downloaded_images(media_root, downloaded)
    missing_audit = {
        split: _classify_missing_media(
            claims=selected[split],
            qrels=image_qrels_by_split[split],
            missing_names=missing[split],
        )
        for split in ("train", "validation")
    }
    gold_media_complete = all(
        split_audit["gold_complete"] for split_audit in missing_audit.values()
    )
    manifest: dict[str, object] = {
        "profile": "development_pilot_media",
        "seed": seed,
        "max_workers": max(1, max_workers),
        "selected_claim_ids": {
            split: [claim.claim_id for claim in claims]
            for split, claims in selected.items()
        },
        "needed_image_counts": {split: len(rows) for split, rows in needed.items()},
        "downloaded": downloaded,
        "missing": missing,
        "missing_audit": missing_audit,
        "image_audit": image_audit,
        "gold_media_complete": gold_media_complete,
        "candidate_availability_policy": (
            "unavailable negative distractors are excluded identically from all selectors"
        ),
        "smoke_pass": gold_media_complete and not image_audit["unreadable"],
        "formal_test_annotations_downloaded": False,
    }
    media_root.mkdir(parents=True, exist_ok=True)
    (media_root / "pilot_media_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert_mocheg_development_sealed(dataset_root)
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_mocheg_pilot_packets(
    dataset_root: Path,
    processed_root: Path,
) -> dict[str, object]:
    """Join claims, qrels, sentence text, and portable media for one pilot packet."""
    assert_mocheg_development_sealed(dataset_root)
    media_manifest_path = dataset_root / "media" / "pilot_media_manifest.json"
    sentence_audit_path = processed_root / "sentence_text_audit.json"
    if not media_manifest_path.is_file() or not sentence_audit_path.is_file():
        raise FileNotFoundError(
            "run --download-pilot-media and --build-sentence-text before packets"
        )
    media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    sentence_audit = json.loads(sentence_audit_path.read_text(encoding="utf-8"))
    if not media_manifest.get("smoke_pass"):
        raise RuntimeError("pilot media gate has not passed")
    if sentence_audit.get("status") != "pass":
        raise RuntimeError("sentence-text gate has not passed")

    sentence_text = {
        row["document_id"]: row["content"]
        for row in _read_jsonl(processed_root / "sentence_text.jsonl")
    }
    media_by_name: dict[str, str] = {}
    ambiguous_names: set[str] = set()
    for row in media_manifest["downloaded"]:
        name = Path(str(row["source_relative_path"])).name
        local_path = (
            dataset_root
            / "media"
            / str(row["mirror"])
            / str(row["relative_path"])
        ).resolve()
        value = str(local_path)
        if name in media_by_name and media_by_name[name] != value:
            ambiguous_names.add(name)
        else:
            media_by_name[name] = value
    for name in ambiguous_names:
        media_by_name.pop(name, None)

    packet_stats: dict[str, dict[str, int]] = {}
    output_files: dict[str, dict[str, object]] = {}
    all_gold_available = True
    all_text_available = True
    for split in ("train", "validation"):
        selected_ids = set(media_manifest["selected_claim_ids"][split])
        claims = {
            row["claim_id"]: row
            for row in _read_jsonl(processed_root / f"{split}_claims.jsonl")
            if row["claim_id"] in selected_ids
        }
        candidate_rows = {
            row["claim_id"]: row
            for row in _read_jsonl(processed_root / f"{split}_candidates.jsonl")
            if row["claim_id"] in selected_ids
        }
        output_path = processed_root / f"{split}_pilot_packets.jsonl"
        candidates_total = 0
        candidates_available = 0
        unavailable_text = 0
        unavailable_image = 0
        unavailable_gold = 0
        claims_without_available_candidates = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for claim_id in sorted(
                selected_ids,
                key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
            ):
                claim = claims[claim_id]
                joined_candidates: list[dict[str, object]] = []
                for candidate in candidate_rows[claim_id]["candidates"]:
                    evidence_id = str(candidate["evidence_id"])
                    modality = str(candidate["modality"])
                    document_id = evidence_id.split(":", 1)[1]
                    content: str | None = None
                    media_path: str | None = None
                    if modality == "text":
                        content = sentence_text.get(document_id)
                        available = content is not None
                        unavailable_text += int(not available)
                    else:
                        media_path = media_by_name.get(document_id)
                        available = media_path is not None
                        unavailable_image += int(not available)
                    unavailable_gold += int(bool(candidate["gold"]) and not available)
                    candidates_total += 1
                    candidates_available += int(available)
                    joined_candidates.append(
                        {
                            **candidate,
                            "document_id": document_id,
                            "available": available,
                            "content": content,
                            "media_path": media_path,
                        }
                    )
                claims_without_available_candidates += int(
                    not any(row["available"] for row in joined_candidates)
                )
                handle.write(
                    json.dumps(
                        {
                            "claim_id": claim_id,
                            "claim": claim["claim"],
                            "label": claim["label"],
                            "split": split,
                            "candidates": joined_candidates,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        stats = {
            "claims": len(selected_ids),
            "candidates": candidates_total,
            "available_candidates": candidates_available,
            "unavailable_text_candidates": unavailable_text,
            "unavailable_image_candidates": unavailable_image,
            "unavailable_gold_candidates": unavailable_gold,
            "claims_without_available_candidates": claims_without_available_candidates,
        }
        packet_stats[split] = stats
        output_files[split] = {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
        all_gold_available = all_gold_available and unavailable_gold == 0
        all_text_available = all_text_available and unavailable_text == 0

    result: dict[str, object] = {
        "profile": "mocheg_development_pilot_packets",
        "status": "pass" if all_gold_available and all_text_available else "failed",
        "all_gold_candidates_available": all_gold_available,
        "all_text_candidates_available": all_text_available,
        "ambiguous_media_basenames": sorted(ambiguous_names),
        "splits": packet_stats,
        "outputs": output_files,
        "formal_test_annotations_accessed": False,
    }
    manifest_path = processed_root / "pilot_packet_manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if result["status"] != "pass":
        raise RuntimeError(f"pilot packet gate failed; inspect {manifest_path}")
    return result
