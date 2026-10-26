from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import re
from typing import Any
from urllib.parse import urlparse


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def default_evidence_query(claim_text: str, maximum_terms: int = 12) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", claim_text)
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        folded = token.casefold()
        if folded in STOPWORDS or folded in seen:
            continue
        selected.append(token)
        seen.add(folded)
        if len(selected) >= maximum_terms:
            break
    return " ".join(selected)


def gdelt_articles_to_review_rows(
    claim_id: str,
    decision_cutoff: str,
    query: str,
    articles: Iterable[Mapping[str, Any]],
    *,
    excluded_domains: set[str] | None = None,
) -> list[dict[str, str]]:
    excluded = {domain.casefold().removeprefix("www.") for domain in (excluded_domains or set())}
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for article in articles:
        url = str(article.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        domain = urlparse(url).netloc.casefold().removeprefix("www.")
        if domain in excluded:
            continue
        seen_urls.add(url)
        evidence_id = sha256(f"{claim_id}|{url}".encode("utf-8")).hexdigest()[:20]
        rows.append(
            {
                "evidence_id": evidence_id,
                "claim_id": claim_id,
                "decision": "pending",
                "exclusion_reason": "",
                "url": url,
                "domain": domain,
                "title": str(article.get("title") or "").strip(),
                "gdelt_seen_at": str(article.get("seendate") or "").strip(),
                "published_at": "",
                "decision_cutoff": decision_cutoff,
                "archived_at": "",
                "timestamp_confidence": "uncertain",
                "relevance_verified": "",
                "publication_time_verified": "",
                "relation": "",
                "media_type": "text",
                "source_cluster_id": "",
                "cluster_verified": "",
                "text_sha256": "",
                "image_phash": "",
                "outgoing_links": "",
                "retrieval_query": query,
                "reviewer_notes": "",
            }
        )
    return rows
