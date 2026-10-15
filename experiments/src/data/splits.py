from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .schema import ClaimRecord


def split_name(record: ClaimRecord) -> str:
    year = record.claim_time.year
    if 2020 <= year <= 2023:
        return "train"
    if year == 2024:
        return "validation"
    if 2025 <= year <= 2026:
        return "test"
    raise ValueError(f"Claim year {year} is outside the preregistered range")


def partition(records: Iterable[ClaimRecord]) -> dict[str, list[ClaimRecord]]:
    result = {"train": [], "validation": [], "test": []}
    for record in records:
        result[split_name(record)].append(record)
    return result


def find_group_leakage(records: Iterable[ClaimRecord]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, set[str]]] = {
        "event_id": defaultdict(set),
        "image_phash": defaultdict(set),
    }
    for record in records:
        split = split_name(record)
        groups["event_id"][record.event_id].add(split)
        if record.image_phash:
            groups["image_phash"][record.image_phash].add(split)

    leakage: dict[str, dict[str, list[str]]] = {"event_id": {}, "image_phash": {}}
    for group_type, values in groups.items():
        for key, splits in values.items():
            if len(splits) > 1:
                leakage[group_type][key] = sorted(splits)
    return leakage


def assert_no_group_leakage(records: Iterable[ClaimRecord]) -> None:
    leakage = find_group_leakage(records)
    if leakage["event_id"] or leakage["image_phash"]:
        raise ValueError(f"Split leakage detected: {leakage}")

