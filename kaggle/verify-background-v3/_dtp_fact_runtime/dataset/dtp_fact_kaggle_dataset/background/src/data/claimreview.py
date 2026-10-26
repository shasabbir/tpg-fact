from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

from .labels import normalize_rating
from .schema import ClaimRecord, parse_datetime


def _first(value: Any, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _image_url(record: Mapping[str, Any]) -> str:
    image = _first(record.get("image", ""))
    if isinstance(image, Mapping):
        return str(image.get("url", ""))
    return str(image)


def normalize_claimreview_record(record: Mapping[str, Any]) -> ClaimRecord | None:
    """Normalize a JSON-LD ClaimReview record using the strict rating policy.

    Records without an explicit rating, image URL, claim date, or publication
    date are excluded instead of inferred.
    """
    label = normalize_rating(str(record.get("reviewRating", {}).get("alternateName", "")))
    item = record.get("itemReviewed", {}) or {}
    if isinstance(item, list):
        item = _first(item, {})
    claim_text = str(item.get("claimReviewed") or record.get("claimReviewed") or "").strip()
    claim_time_raw = item.get("datePublished") or item.get("dateCreated")
    cutoff_raw = record.get("datePublished")
    image_url = _image_url(item) or _image_url(record)
    fact_check_url = str(record.get("url", "")).strip()
    if not all([label, claim_text, claim_time_raw, cutoff_raw, image_url, fact_check_url]):
        return None

    claim_time = parse_datetime(str(claim_time_raw))
    cutoff = parse_datetime(str(cutoff_raw))
    if claim_time > cutoff:
        return None
    identity = sha256(f"{claim_text}|{image_url}|{fact_check_url}".encode("utf-8")).hexdigest()
    event_seed = f"{claim_time.date()}|{urlparse(fact_check_url).netloc}|{claim_text.casefold()}"
    event_id = sha256(event_seed.encode("utf-8")).hexdigest()[:16]
    return ClaimRecord(
        claim_id=identity[:20],
        claim_text=claim_text,
        claim_image_url=image_url,
        image_sha256="",
        image_phash="",
        claim_time=claim_time,
        decision_cutoff=cutoff,
        fact_check_url=fact_check_url,
        source_rating=str(record.get("reviewRating", {}).get("alternateName", "")),
        normalized_label=label,
        event_id=event_id,
        evidence_ids=(),
    )


def normalize_claimreview(records: Iterable[Mapping[str, Any]]) -> tuple[list[ClaimRecord], int]:
    accepted: list[ClaimRecord] = []
    rejected = 0
    for record in records:
        normalized = normalize_claimreview_record(record)
        if normalized is None:
            rejected += 1
        else:
            accepted.append(normalized)
    return accepted, rejected

