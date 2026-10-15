from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


USER_AGENT = "TPG-Fact/0.1 research-contact: 26-94030-1@student.aiub.edu"


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


def query_gdelt(query: str, max_records: int = 100) -> list[dict[str, Any]]:
    """Query the public GDELT DOC 2.0 endpoint.

    Network results must be persisted with the query time before use in a study.
    """
    endpoint = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={quote_plus(query)}&mode=ArtList&maxrecords={int(max_records)}&format=json"
    )
    payload = _get_json(endpoint)
    return list(payload.get("articles", []))


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


def utc_collection_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

