from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .registry import DATASETS, DatasetSpec


USER_AGENT = "TRACE-SCV-public-data-preflight/0.1"


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:  # nosec B310: fixed HTTPS APIs
        return json.loads(response.read().decode("utf-8"))


def check_dataset(spec: DatasetSpec) -> dict:
    if spec.provider == "huggingface":
        url = f"https://huggingface.co/api/datasets/{spec.identifier}"
    elif spec.provider == "github":
        url = f"https://api.github.com/repos/{spec.identifier}"
    else:
        raise ValueError(f"unsupported provider: {spec.provider}")

    result = {
        "key": spec.key,
        "name": spec.name,
        "provider": spec.provider,
        "identifier": spec.identifier,
        "required": spec.required,
        "homepage": spec.homepage,
        "expected_revision": spec.revision,
        "license_expected": spec.license_expected,
        "accessible": False,
    }
    try:
        payload = _fetch_json(url)
        result["accessible"] = True
        if spec.provider == "huggingface":
            card = payload.get("cardData") or {}
            result.update(
                {
                    "private": payload.get("private"),
                    "gated": payload.get("gated"),
                    "revision": payload.get("sha"),
                    "license_reported": card.get("license"),
                    "file_count": len(payload.get("siblings") or []),
                }
            )
        else:
            license_data = payload.get("license") or {}
            result.update(
                {
                    "private": payload.get("private"),
                    "archived": payload.get("archived"),
                    "default_branch": payload.get("default_branch"),
                    "revision": payload.get("pushed_at"),
                    "license_reported": license_data.get("spdx_id"),
                }
            )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        result["error"] = str(error)
    return result


def run_preflight() -> dict:
    rows = [check_dataset(spec) for spec in DATASETS.values()]
    required_ok = all(row["accessible"] for row in rows if row["required"])
    revisions_match = all(
        row.get("expected_revision") in {None, row.get("revision")}
        for row in rows
        if row["required"]
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if required_ok and revisions_match else "fail",
        "required_accessible": required_ok,
        "required_revisions_match": revisions_match,
        "datasets": rows,
        "scientific_results_validated": False,
        "interpretation": "This checks public release access and pinned revisions only.",
    }
