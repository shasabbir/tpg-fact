from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data.factcheck_api import FactCheckApiClient, FactCheckApiError
from data.io import write_json
from training.preflight import inspect_full_run_readiness


def _conflict_manifest_status(root: Path, value: str | None) -> str:
    if not value:
        return "not_checked"
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return "invalid"
    identifiers: set[str] = set()
    for row in rows:
        claim_id = str(row.get("claim_id", ""))
        donor_id = str(row.get("donor_evidence_id", ""))
        if (
            not claim_id
            or not donor_id
            or row.get("semantic_conflict_verified") is not True
            or claim_id in identifiers
        ):
            return "invalid"
        identifiers.add(claim_id)
    return "valid" if identifiers else "invalid"


def _api_status(root: Path, check_api: bool, collection_manifest: str | None) -> str:
    if collection_manifest:
        path = Path(collection_manifest)
        if not path.is_absolute():
            path = root / path
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "invalid_frozen_snapshot"
        if (
            manifest.get("source") == "Google Fact Check Tools API v1alpha1 claims.search"
            and int(manifest.get("candidate_count_after_deduplication", 0)) > 0
            and manifest.get("api_key_recorded") is False
        ):
            return "frozen_snapshot"
        return "invalid_frozen_snapshot"
    if not check_api:
        return "not_checked"
    api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()
    key_file = root / ".secrets" / "google_factcheck_api_key.txt"
    if not api_key and key_file.exists():
        api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        return "missing_key"
    try:
        FactCheckApiClient(api_key, retries=0).search_pages(
            publisher_site="politifact.com", page_size=1, max_pages=1
        )
        return "ok"
    except FactCheckApiError:
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report whether data, model identities, trainer, and hardware are ready for a full run"
    )
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument(
        "--collection-manifest",
        help="Use a preserved key-free collection manifest instead of live API access",
    )
    parser.add_argument("--conflict-manifest")
    parser.add_argument("--output", default="outputs/preflight/full_run_readiness.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = inspect_full_run_readiness(
        root,
        api_status=_api_status(root, args.check_api, args.collection_manifest),
        conflict_manifest_status=_conflict_manifest_status(root, args.conflict_manifest),
    )
    write_json(root / args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
