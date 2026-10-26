from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


USER_AGENT = "DTP-Fact/0.1 research-contact: 26-94030-1@student.aiub.edu"


def _get_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class ArchiveObservation:
    url: str
    timestamp: str
    status: str
    mime: str
    filename: str
    offset: str
    length: str


def query_gdelt(
    query: str,
    max_records: int = 100,
    *,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
) -> list[dict[str, Any]]:
    """Query the public GDELT DOC 2.0 endpoint.

    Network results must be persisted with the query time before use in a study.
    """
    endpoint = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={quote_plus(query)}&mode=ArtList&maxrecords={int(max_records)}&format=json"
    )
    if start_datetime:
        endpoint += f"&startdatetime={quote_plus(start_datetime)}"
    if end_datetime:
        endpoint += f"&enddatetime={quote_plus(end_datetime)}"
    payload = _get_json(endpoint)
    return list(payload.get("articles", []))


def list_common_crawl_indexes() -> list[dict[str, Any]]:
    """Return the current Common Crawl index catalogue."""
    payload = _get_json("https://index.commoncrawl.org/collinfo.json")
    if not isinstance(payload, list):
        raise ValueError("Unexpected Common Crawl index catalogue response")
    return [dict(item) for item in payload]


def common_crawl_observations(
    target_url: str,
    index: str = "CC-MAIN-2026-39",
) -> list[ArchiveObservation]:
    endpoint = (
        f"https://index.commoncrawl.org/{index}-index?"
        f"url={quote_plus(target_url)}&output=json"
    )
    request = Request(endpoint, headers={"User-Agent": USER_AGENT})
    observations: list[ArchiveObservation] = []
    with urlopen(request, timeout=30) as response:
        for line in response.read().decode("utf-8").splitlines():
            row = json.loads(line)
            observations.append(
                ArchiveObservation(
                    url=row.get("url", ""),
                    timestamp=row.get("timestamp", ""),
                    status=row.get("status", ""),
                    mime=row.get("mime", ""),
                    filename=row.get("filename", ""),
                    offset=row.get("offset", ""),
                    length=row.get("length", ""),
                )
            )
    return observations


def wayback_pre_cutoff_observation(
    target_url: str,
    published_at: datetime,
    cutoff: datetime,
) -> ArchiveObservation | None:
    """Return the first Internet Archive CDX capture in the admissible window.

    Wayback is used as an independent open-archive fallback when Common Crawl
    has no collection overlapping a short publication-to-decision interval.
    """
    published_at = published_at.astimezone(timezone.utc)
    cutoff = cutoff.astimezone(timezone.utc)
    if published_at > cutoff:
        raise ValueError("published_at must not be later than cutoff")
    endpoint = (
        "https://web.archive.org/cdx/search/cdx?"
        f"url={quote_plus(target_url)}&output=json&filter=statuscode%3A200"
        f"&from={published_at.strftime('%Y%m%d%H%M%S')}"
        f"&to={cutoff.strftime('%Y%m%d%H%M%S')}"
        "&limit=1&fl=timestamp%2Coriginal%2Cstatuscode%2Cdigest"
    )
    payload = _get_json(endpoint)
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    row = payload[1]
    if not isinstance(row, list) or len(row) < 3:
        return None
    timestamp = str(row[0])
    try:
        observed_at = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    if not published_at <= observed_at <= cutoff or str(row[2]) != "200":
        return None
    return ArchiveObservation(
        url=str(row[1]),
        timestamp=timestamp,
        status=str(row[2]),
        mime="text/html",
        filename="internet_archive_cdx",
        offset="",
        length=str(row[3]) if len(row) > 3 else "",
    )


def find_pre_cutoff_observation(
    target_url: str,
    published_at: datetime,
    cutoff: datetime,
    *,
    indexes: list[dict[str, Any]] | None = None,
    max_indexes: int | None = None,
) -> tuple[ArchiveObservation | None, list[str]]:
    """Find proof that a URL was crawled between publication and cutoff.

    Indexes are searched from newest to oldest. The returned audit trail lists
    every queried Common Crawl collection so negative results remain visible.
    """
    published_at = published_at.astimezone(timezone.utc)
    cutoff = cutoff.astimezone(timezone.utc)
    if published_at > cutoff:
        raise ValueError("published_at must not be later than cutoff")
    catalogue = indexes if indexes is not None else list_common_crawl_indexes()
    eligible: list[str] = []
    for item in catalogue:
        index_id = str(item.get("id") or "")
        to_text = str(item.get("to") or "")
        from_text = str(item.get("from") or "")
        if not index_id or not to_text:
            continue
        try:
            index_to = datetime.fromisoformat(to_text.replace("Z", "+00:00"))
            index_from = datetime.fromisoformat(from_text.replace("Z", "+00:00")) if from_text else index_to
        except ValueError:
            continue
        if index_to.tzinfo is None:
            index_to = index_to.replace(tzinfo=timezone.utc)
        else:
            index_to = index_to.astimezone(timezone.utc)
        if index_from.tzinfo is None:
            index_from = index_from.replace(tzinfo=timezone.utc)
        else:
            index_from = index_from.astimezone(timezone.utc)
        if index_from <= cutoff and index_to >= published_at:
            eligible.append(index_id)
    eligible.sort(reverse=True)
    if max_indexes is not None:
        eligible = eligible[:max_indexes]

    queried: list[str] = []
    matches: list[ArchiveObservation] = []
    for index_id in eligible:
        queried.append(index_id)
        for observation in common_crawl_observations(target_url, index=index_id):
            try:
                observed_at = datetime.strptime(
                    observation.timestamp, "%Y%m%d%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if published_at <= observed_at <= cutoff and observation.status == "200":
                matches.append(observation)
        if matches:
            break
    matches.sort(key=lambda item: item.timestamp)
    return (matches[0] if matches else None), queried


def utc_collection_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

