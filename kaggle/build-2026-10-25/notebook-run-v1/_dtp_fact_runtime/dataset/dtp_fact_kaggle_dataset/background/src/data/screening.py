from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from typing import Iterable, Mapping


LABELS = ("supported", "refuted", "insufficient_evidence")
SPLIT_RATIOS = {"train": 0.50, "validation": 0.20, "test": 0.30}
MULTIMODAL_TERMS = re.compile(
    r"\b(image|photo|photograph|picture|video|meme|screenshot|caption|"
    r"ai[- ]generated|altered|doctored|deepfake)\b",
    re.IGNORECASE,
)
US_TERMS = re.compile(
    r"\b(united states|u\.s\.|usa|american|white house|congress|senate|"
    r"supreme court|federal|president|governor|pentagon|washington,? d\.?c\.?)\b",
    re.IGNORECASE,
)


def candidate_year(row: Mapping[str, str]) -> int | None:
    for field in ("api_review_date", "api_claim_date"):
        value = row.get(field, "").strip()
        if len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def temporal_split(row: Mapping[str, str]) -> str | None:
    year = candidate_year(row)
    if year is None:
        return None
    if 2020 <= year <= 2023:
        return "train"
    if year == 2024:
        return "validation"
    if 2025 <= year <= 2026:
        return "test"
    return None


def priority_signals(row: Mapping[str, str]) -> tuple[int, tuple[str, ...]]:
    text = " ".join(
        (
            row.get("claim_text", ""),
            row.get("review_title", ""),
            row.get("textual_rating", ""),
        )
    )
    score = 0
    signals: list[str] = []
    if MULTIMODAL_TERMS.search(text):
        score += 3
        signals.append("multimodal_term")
    if US_TERMS.search(text):
        score += 2
        signals.append("us_term")
    if row.get("api_review_date", "").strip() and row.get("api_claim_date", "").strip():
        score += 1
        signals.append("both_api_dates")
    if row.get("claimant", "").strip():
        score += 1
        signals.append("claimant_present")
    return score, tuple(signals)


def _tie_break(row: Mapping[str, str], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}|{row.get('candidate_id', '')}".encode("utf-8")
    ).hexdigest()


def _balanced_label_targets(
    pools: Mapping[str, list[dict[str, str]]], target_size: int
) -> dict[str, int]:
    base = target_size // len(LABELS)
    targets = {label: base for label in LABELS}
    for label in LABELS[: target_size % len(LABELS)]:
        targets[label] += 1
    deficit = 0
    for label in LABELS:
        if len(pools[label]) < targets[label]:
            deficit += targets[label] - len(pools[label])
            targets[label] = len(pools[label])
    while deficit:
        changed = False
        for label in LABELS:
            if targets[label] < len(pools[label]):
                targets[label] += 1
                deficit -= 1
                changed = True
                if not deficit:
                    break
        if not changed:
            break
    return targets


def _split_targets(
    rows: list[dict[str, str]], target: int
) -> dict[str, int]:
    available = {
        split: sum(temporal_split(row) == split for row in rows) for split in SPLIT_RATIOS
    }
    targets = {
        split: min(available[split], int(math.floor(target * ratio)))
        for split, ratio in SPLIT_RATIOS.items()
    }
    assigned = sum(targets.values())
    while assigned < target:
        changed = False
        for split in ("train", "test", "validation"):
            if targets[split] < available[split]:
                targets[split] += 1
                assigned += 1
                changed = True
                if assigned == target:
                    break
        if not changed:
            break
    return targets


def _publisher_round_robin(
    rows: list[dict[str, str]], target: int, seed: int
) -> list[dict[str, str]]:
    by_publisher: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        publisher = row.get("collected_publisher_filter", "") or row.get(
            "publisher_site", "unknown"
        )
        by_publisher[publisher].append(row)
    for publisher_rows in by_publisher.values():
        publisher_rows.sort(
            key=lambda row: (
                -priority_signals(row)[0],
                _tie_break(row, seed),
            )
        )
    selected: list[dict[str, str]] = []
    publishers = sorted(by_publisher)
    while len(selected) < target:
        changed = False
        for publisher in publishers:
            if by_publisher[publisher]:
                selected.append(by_publisher[publisher].pop(0))
                changed = True
                if len(selected) == target:
                    break
        if not changed:
            break
    return selected


def build_screening_queue(
    rows: Iterable[Mapping[str, str]],
    *,
    target_size: int = 1200,
    seed: int = 2026,
) -> list[dict[str, str]]:
    if target_size < 1:
        raise ValueError("target_size must be positive")
    eligible = [
        dict(row)
        for row in rows
        if row.get("suggested_label", "") in LABELS and temporal_split(row) is not None
    ]
    pools = {
        label: [row for row in eligible if row["suggested_label"] == label]
        for label in LABELS
    }
    label_targets = _balanced_label_targets(pools, min(target_size, len(eligible)))
    selected: list[dict[str, str]] = []
    for label in LABELS:
        split_targets = _split_targets(pools[label], label_targets[label])
        for split in ("train", "validation", "test"):
            candidates = [row for row in pools[label] if temporal_split(row) == split]
            selected.extend(
                _publisher_round_robin(candidates, split_targets[split], seed)
            )
    selected.sort(
        key=lambda row: (
            LABELS.index(row["suggested_label"]),
            ("train", "validation", "test").index(temporal_split(row)),
            -priority_signals(row)[0],
            _tie_break(row, seed),
        )
    )
    for rank, row in enumerate(selected, start=1):
        score, signals = priority_signals(row)
        row["screening_rank"] = str(rank)
        row["temporal_split"] = str(temporal_split(row))
        row["priority_score"] = str(score)
        row["priority_signals"] = "|".join(signals)
    return selected
