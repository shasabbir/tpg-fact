from __future__ import annotations

import re


SUPPORTED = {
    "true",
    "correct",
    "accurate",
}

REFUTED = {
    "false",
    "fake",
    "miscaptioned",
    "incorrect",
}

INSUFFICIENT = {
    "unproven",
    "unsubstantiated",
    "no evidence",
    "not enough evidence",
}

EXCLUDED = {
    "mostly true",
    "half true",
    "partly false",
    "mixture",
    "satire",
    "opinion",
}


def canonicalize_rating(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_rating(value: str) -> str | None:
    """Map only explicit ratings; ambiguous ratings return None."""
    rating = canonicalize_rating(value)
    if rating in SUPPORTED:
        return "supported"
    if rating in REFUTED:
        return "refuted"
    if rating in INSUFFICIENT:
        return "insufficient_evidence"
    return None


def is_explicitly_excluded(value: str) -> bool:
    return canonicalize_rating(value) in EXCLUDED

