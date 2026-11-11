from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRAIN_ROLES = ("fit", "calibration", "model_selection", "internal_holdout")
TRAIN_ROLE_CUTOFFS = (0.55, 0.70, 0.85, 1.0)
OFFICIAL_LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)
_PSEUDO_SOURCES = {"", "metadata", "none", "n/a", "na", "unknown"}
_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_source_url(value: str) -> str:
    """Return a conservative, deterministic source key for exact-source evaluation."""
    raw = str(value or "").strip()
    if raw.lower() in _PSEUDO_SOURCES:
        return ""
    if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
        return raw.casefold().rstrip("/")
    parts = urlsplit(raw)
    host = (parts.hostname or "").casefold()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    netloc = host if port is None else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = []
    for key, item in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered.startswith("utm_") or lowered in _TRACKING_PARAMETERS:
            continue
        query.append((key, item))
    return urlunsplit((parts.scheme.casefold(), netloc, path, urlencode(sorted(query)), ""))


def claim_identifier(row: Mapping[str, object], ordinal: int) -> str:
    images = row.get("claim_images") or []
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        stem = Path(str(images[0])).stem
        prefix = stem.split("#", 1)[0].strip()
        if prefix:
            return prefix
    stable = "\n".join(
        (
            str(row.get("claim_text", "")).strip(),
            str(row.get("article", "")).strip(),
            str(row.get("date", "")).strip(),
            str(ordinal),
        )
    )
    return sha256_text(stable)[:24]


def train_role(claim_id: str, seed: int) -> str:
    integer = int(sha256_text(f"averimatec:{seed}:{claim_id}")[:16], 16)
    proportion = integer / float(16**16)
    for role, cutoff in zip(TRAIN_ROLES, TRAIN_ROLE_CUTOFFS, strict=True):
        if proportion < cutoff:
            return role
    raise AssertionError("unreachable role partition")


def gold_source_urls(row: Mapping[str, object]) -> set[str]:
    sources: set[str] = set()
    questions = row.get("questions") or []
    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
        return sources
    for question in questions:
        if not isinstance(question, Mapping):
            continue
        answers = question.get("answers") or []
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            continue
        for answer in answers:
            if not isinstance(answer, Mapping):
                continue
            normalized = normalize_source_url(str(answer.get("source_url", "")))
            if normalized:
                sources.add(normalized)
    return sources


def load_annotations(path: Path, *, split: str) -> list[dict[str, object]]:
    """Load released annotations while preventing accidental hidden-test labels."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list in {path}")
    output: list[dict[str, object]] = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"record {ordinal} in {path} is not an object")
        if split == "test" and any(key in raw for key in ("label", "questions", "justification")):
            raise RuntimeError("refusing a test record containing outcome-bearing fields")
        label = raw.get("label")
        if split != "test" and label not in OFFICIAL_LABELS:
            raise ValueError(f"unknown AVerImaTeC label at {split}:{ordinal}: {label!r}")
        row = dict(raw)
        row["claim_id"] = claim_identifier(raw, ordinal)
        row["split"] = split
        row["gold_source_urls"] = sorted(gold_source_urls(raw)) if split != "test" else []
        output.append(row)
    identifiers = [str(row["claim_id"]) for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate claim identifiers in {path}")
    return output


def candidate_identifier(
    claim_id: str,
    source_url: str,
    medium: str,
    content: str,
    ordinal: int,
) -> str:
    normalized = normalize_source_url(source_url)
    content_hash = sha256_text(content)
    payload = "\0".join((claim_id, normalized, medium.casefold(), content_hash, str(ordinal)))
    return sha256_text(payload)


def assert_disjoint_roles(assignments: Mapping[str, Iterable[str]]) -> None:
    seen: set[str] = set()
    for role in TRAIN_ROLES:
        current = set(map(str, assignments.get(role, ())))
        overlap = seen & current
        if overlap:
            raise ValueError(f"claim leakage across roles: {sorted(overlap)[:3]}")
        seen.update(current)
