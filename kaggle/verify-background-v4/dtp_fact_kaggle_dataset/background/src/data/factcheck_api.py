from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


FACTCHECK_SEARCH_ENDPOINT = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)
USER_AGENT = "DTP-Fact/0.1 research-contact: 26-94030-1@student.aiub.edu"


class FactCheckApiError(RuntimeError):
    """A sanitized Fact Check Tools API failure that never includes the API key."""


def utc_collection_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _publisher_site(review: Mapping[str, Any]) -> str:
    publisher = review.get("publisher") or {}
    site = str(publisher.get("site") or "").strip()
    if site:
        return site.casefold()
    return urlparse(str(review.get("url") or "")).netloc.casefold()


def _candidate_id(claim_text: str, review_url: str) -> str:
    value = f"{claim_text.strip()}|{review_url.strip()}"
    return sha256(value.encode("utf-8")).hexdigest()[:20]


def claim_api_payload_to_candidates(
    payload: Mapping[str, Any],
    *,
    collected_query: str = "",
    collected_publisher_filter: str = "",
) -> list[dict[str, str]]:
    """Flatten one Google Claim object into one row per publisher review.

    Google returns claim-level objects containing one or more reviews. The
    flattened rows intentionally leave image and audited fields blank: the API
    does not reliably identify the image that was actually fact-checked.
    """

    claim_text = str(payload.get("text") or "").strip()
    claimant = str(payload.get("claimant") or "").strip()
    claim_date = str(payload.get("claimDate") or "").strip()
    candidates: list[dict[str, str]] = []
    for review in payload.get("claimReview") or []:
        review_url = str(review.get("url") or "").strip()
        if not claim_text or not review_url:
            continue
        publisher = review.get("publisher") or {}
        candidates.append(
            {
                "candidate_id": _candidate_id(claim_text, review_url),
                "publisher": str(publisher.get("name") or "").strip(),
                "publisher_site": _publisher_site(review),
                "claim_text": claim_text,
                "claimant": claimant,
                "api_claim_date": claim_date,
                "api_review_date": str(review.get("reviewDate") or "").strip(),
                "review_url": review_url,
                "review_title": str(review.get("title") or "").strip(),
                "textual_rating": str(review.get("textualRating") or "").strip(),
                "language_code": str(review.get("languageCode") or "").strip(),
                "collected_query": collected_query,
                "collected_publisher_filter": collected_publisher_filter,
            }
        )
    return candidates


def _candidate_year(candidate: Mapping[str, str]) -> int | None:
    for key in ("api_review_date", "api_claim_date"):
        text = candidate.get(key, "").strip()
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
    return None


class FactCheckApiClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: int = 30,
        retries: int = 3,
        retry_delay_seconds: float = 1.0,
        fetch_json: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty Fact Check Tools API key is required")
        self._api_key = api_key.strip()
        self.timeout = timeout
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._fetch_json = fetch_json or self._network_fetch_json

    def _network_fetch_json(self, url: str) -> Mapping[str, Any]:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                try:
                    payload = json.loads(exc.read().decode("utf-8"))
                    reason = str(payload.get("error", {}).get("message") or "")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    reason = ""
                reason = reason.replace(self._api_key, "[REDACTED]")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.retries:
                    time.sleep(self.retry_delay_seconds * (2**attempt))
                    continue
                suffix = f": {reason}" if reason else ""
                raise FactCheckApiError(
                    f"Fact Check Tools API returned HTTP {exc.code}{suffix}"
                ) from None
            except URLError as exc:
                if attempt < self.retries:
                    time.sleep(self.retry_delay_seconds * (2**attempt))
                    continue
                raise FactCheckApiError(
                    f"Fact Check Tools API request failed: {exc.reason}"
                ) from None
        raise AssertionError("unreachable")

    def search_pages(
        self,
        *,
        query: str = "",
        publisher_site: str = "",
        language_code: str = "en",
        page_size: int = 100,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        if not query.strip() and not publisher_site.strip():
            raise ValueError("query or publisher_site must be supplied")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")

        pages: list[dict[str, Any]] = []
        page_token = ""
        seen_tokens: set[str] = set()
        for _ in range(max_pages):
            parameters = {
                "key": self._api_key,
                "languageCode": language_code,
                "pageSize": str(page_size),
            }
            if query.strip():
                parameters["query"] = query.strip()
            if publisher_site.strip():
                parameters["reviewPublisherSiteFilter"] = publisher_site.strip()
            if page_token:
                parameters["pageToken"] = page_token
            url = f"{FACTCHECK_SEARCH_ENDPOINT}?{urlencode(parameters)}"
            payload = dict(self._fetch_json(url))
            pages.append(payload)
            next_token = str(payload.get("nextPageToken") or "")
            if not next_token:
                break
            if next_token in seen_tokens:
                raise FactCheckApiError("API pagination repeated a page token")
            seen_tokens.add(next_token)
            page_token = next_token
        return pages


def candidates_from_pages(
    pages: Iterable[Mapping[str, Any]],
    *,
    collected_query: str = "",
    collected_publisher_filter: str = "",
    start_year: int = 2020,
    end_year: int = 2026,
) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for page in pages:
        for claim in page.get("claims") or []:
            for candidate in claim_api_payload_to_candidates(
                claim,
                collected_query=collected_query,
                collected_publisher_filter=collected_publisher_filter,
            ):
                year = _candidate_year(candidate)
                if year is None or not start_year <= year <= end_year:
                    continue
                by_id[candidate["candidate_id"]] = candidate
    return [by_id[key] for key in sorted(by_id)]
