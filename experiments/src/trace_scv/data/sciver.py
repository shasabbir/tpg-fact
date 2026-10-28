from __future__ import annotations

import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

from trace_scv.data.io import sha256_file, write_packets
from trace_scv.image_io import load_rgb_on_white
from trace_scv.schema import ClaimPacket, EvidenceItem


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_VISUAL_MODALITIES = frozenset({"table", "figure", "image"})
_REQUIRED_MEDIA_MODALITIES = frozenset({"figure", "image"})


def _plain_text(value: object, limit: int = 4000) -> str:
    text = html.unescape(_TAG_RE.sub(" ", str(value or "")))
    return _SPACE_RE.sub(" ", text).strip()[:limit]


def group_split(group_id: str, validation_fraction: float, seed: int) -> str:
    token = f"{seed}:{group_id}".encode("utf-8")
    fraction = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if fraction < validation_fraction else "train"


def bounded_rows(
    indexed_rows: list[tuple[int, dict]],
    limit: int | None,
    *,
    seed: int,
    split: str,
) -> list[tuple[int, dict]]:
    """Take a deterministic, order-independent bounded sample for non-formal runs."""
    if limit is None:
        return indexed_rows

    def key(item: tuple[int, dict]) -> str:
        index, row = item
        token = (
            f"{seed}:{split}:{row.get('paperid', '')}:"
            f"{row.get('request_id', index)}:{index}"
        )
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    return sorted(indexed_rows, key=key)[:limit]


def _perceptual_dhash(path: Path) -> int | None:
    try:
        image = load_rgb_on_white(path)
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.tobytes())
        gray.close()
        image.close()
        value = 0
        for row in range(8):
            start = row * 9
            for column in range(8):
                value = (value << 1) | int(
                    pixels[start + column] > pixels[start + column + 1]
                )
        return value
    except Exception:
        return None


def _split_quality_audit(
    packets: list[ClaimPacket],
    media_sha_cache: dict[str, str],
    media_dhash_cache: dict[str, int | None],
) -> tuple[dict, dict[str, set]]:
    modalities: Counter[str] = Counter()
    gold_modalities: Counter[str] = Counter()
    content_hashes: list[str] = []
    media_hashes: list[str] = []
    perceptual_hashes: list[int] = []
    references: dict[str, dict[object, list[dict]]] = {
        "content": {},
        "media": {},
        "perceptual": {},
    }
    required_media_items = required_media_present = 0
    table_items = table_media_present = 0
    missing_media_paths = 0

    for packet in packets:
        for item in packet.evidence:
            modalities[item.modality] += 1
            if item.gold:
                gold_modalities[item.modality] += 1
            content_hash = hashlib.sha256(
                f"{item.modality}\0{_SPACE_RE.sub(' ', item.content).strip()}".encode(
                    "utf-8"
                )
            ).hexdigest()
            content_hashes.append(content_hash)
            reference = {
                "claim_id": packet.claim_id,
                "group_id": packet.group_id or packet.claim_id,
                "evidence_id": item.evidence_id,
                "modality": item.modality,
                "gold": bool(item.gold),
            }
            references["content"].setdefault(content_hash, []).append(reference)
            media_path = Path(item.media_path) if item.media_path else None
            media_exists = bool(media_path and media_path.is_file())
            if item.modality in _REQUIRED_MEDIA_MODALITIES:
                required_media_items += 1
                required_media_present += int(media_exists)
            if item.modality == "table":
                table_items += 1
                table_media_present += int(media_exists)
            if item.media_path and not media_exists:
                missing_media_paths += 1
            if not media_exists or media_path is None:
                continue
            key = str(media_path.resolve())
            media_sha_cache.setdefault(key, sha256_file(media_path))
            if key not in media_dhash_cache:
                media_dhash_cache[key] = _perceptual_dhash(media_path)
            media_hashes.append(media_sha_cache[key])
            references["media"].setdefault(media_sha_cache[key], []).append(reference)
            if media_dhash_cache[key] is not None:
                perceptual_hash = int(media_dhash_cache[key])
                perceptual_hashes.append(perceptual_hash)
                references["perceptual"].setdefault(perceptual_hash, []).append(reference)

    content_counts = Counter(content_hashes)
    media_counts = Counter(media_hashes)
    perceptual_counts = Counter(perceptual_hashes)
    stats = {
        "claims": len(packets),
        "evidence_items": len(content_hashes),
        "gold_evidence_items": sum(gold_modalities.values()),
        "fallback_gold_evidence_items": sum(
            item.metadata.get("fallback") == "missing_annotated_evidence"
            for packet in packets
            for item in packet.evidence
            if item.gold
        ),
        "modality_counts": dict(sorted(modalities.items())),
        "gold_modality_counts": dict(sorted(gold_modalities.items())),
        "visual_items": sum(modalities[name] for name in _VISUAL_MODALITIES),
        "required_visual_media_items": required_media_items,
        "required_visual_media_present": required_media_present,
        "required_visual_media_missing": required_media_items - required_media_present,
        "required_visual_media_coverage": (
            required_media_present / required_media_items if required_media_items else None
        ),
        "table_items": table_items,
        "table_media_present": table_media_present,
        "missing_declared_media_paths": missing_media_paths,
        "unique_content_fingerprints": len(content_counts),
        "duplicate_content_occurrences": len(content_hashes) - len(content_counts),
        "unique_media_sha256": len(media_counts),
        "duplicate_media_occurrences": len(media_hashes) - len(media_counts),
        "unique_perceptual_hashes": len(perceptual_counts),
        "duplicate_perceptual_occurrences": len(perceptual_hashes)
        - len(perceptual_counts),
    }
    fingerprints = {
        "content": set(content_counts),
        "media": set(media_counts),
        "perceptual": set(perceptual_counts),
        "references": references,
    }
    return stats, fingerprints


def _near_perceptual_pairs(left: set[int], right: set[int], threshold: int = 4) -> int:
    return sum((a ^ b).bit_count() <= threshold for a in left for b in right)


def _exact_duplicate_details(
    left: dict[str, set], right: dict[str, set], kind: str
) -> list[dict]:
    details = []
    for fingerprint in sorted(left[kind] & right[kind]):
        left_rows = left["references"][kind][fingerprint]
        right_rows = right["references"][kind][fingerprint]
        rendered = f"{fingerprint:016x}" if kind == "perceptual" else str(fingerprint)
        details.append(
            {
                "fingerprint": rendered,
                "left": left_rows,
                "right": right_rows,
                "involves_gold": any(row["gold"] for row in [*left_rows, *right_rows]),
            }
        )
    return details


def _near_perceptual_details(
    left: dict[str, set], right: dict[str, set], threshold: int = 4, limit: int = 10000
) -> tuple[list[dict], bool]:
    details = []
    for left_hash in sorted(left["perceptual"]):
        for right_hash in sorted(right["perceptual"]):
            distance = (left_hash ^ right_hash).bit_count()
            if distance > threshold:
                continue
            if len(details) >= limit:
                return details, True
            left_rows = left["references"]["perceptual"][left_hash]
            right_rows = right["references"]["perceptual"][right_hash]
            details.append(
                {
                    "left_fingerprint": f"{left_hash:016x}",
                    "right_fingerprint": f"{right_hash:016x}",
                    "hamming_distance": distance,
                    "left": left_rows,
                    "right": right_rows,
                    "involves_gold": any(
                        row["gold"] for row in [*left_rows, *right_rows]
                    ),
                }
            )
    return details, False


def _find_section(paper: dict, section_id: str) -> dict | None:
    target = str(section_id).strip()
    sections = [*paper.get("sections", []), *paper.get("appendix", [])]
    # SciVer identifiers are hierarchical. An earlier implementation accepted a
    # parent (for example, ``4``) while looking for an annotated child (``4.3``).
    # Parent rows often contain only a heading, so that behavior silently replaced
    # the gold subsection with generic, duplicated evidence. Exact matches must
    # therefore take precedence and a child request must never fall back upward.
    for section in sections:
        current = str(section.get("section_id", "")).strip()
        if current == target:
            return section
    # A broad parent annotation may legitimately map to its first material child
    # when the release has no explicit parent row. This direction is safe because
    # it cannot turn a precise annotation into a generic ancestor.
    for section in sections:
        current = str(section.get("section_id", "")).strip()
        if current.startswith(target + "."):
            return section
    return None


def _table_content(paper: dict, item: str) -> tuple[str, str | None]:
    table = paper.get("tables", {}).get(str(item), {})
    if not isinstance(table, dict):
        return _plain_text(table), None
    caption = table.get("caption") or table.get("table_caption") or ""
    body = table.get("table_html") or table.get("table") or table
    text = _plain_text(f"{caption} {body}")
    return text or f"Table {item}", None


def _image_content(paper: dict, item: str, root: Path) -> tuple[str, str | None]:
    row = paper.get("image_paths", {}).get(str(item), {})
    if not isinstance(row, dict):
        row = {}
    caption = _plain_text(row.get("caption") or f"Figure {item}")
    file_name = row.get("figure_path")
    media = root / "images" / file_name if file_name else None
    return caption, str(media) if media and media.exists() else None


def _distractor_sections(paper: dict, excluded: set[str], count: int, seed: str) -> list[dict]:
    choices = [
        row
        for row in paper.get("sections", [])
        if str(row.get("section_id", "")) not in excluded and _plain_text(row.get("text"))
    ]
    random.Random(seed).shuffle(choices)
    return choices[:count]


def build_sciver_packet(
    row: dict,
    paper: dict,
    dataset_root: Path,
    split: str,
    row_index: int,
    max_candidates: int = 12,
) -> ClaimPacket:
    paper_id = str(row["paperid"])
    evidence: list[EvidenceItem] = []
    gold_sections = {str(value) for value in row.get("section", [])}

    for section_id in sorted(gold_sections):
        section = _find_section(paper, section_id)
        if section is None:
            continue
        content = _plain_text(
            f"{section.get('section_name', '')}. {section.get('text', '')}"
        )
        if content:
            evidence.append(
                EvidenceItem(
                    evidence_id=f"text:{section_id}",
                    modality="text",
                    content=content,
                    source_document_id=paper_id,
                    gold=True,
                    metadata={"section_id": section_id},
                )
            )

    modality = row.get("type")
    item = str(row.get("item", ""))
    if modality == "table":
        content, media_path = _table_content(paper, item)
        image_name = Path(str(row.get("image_path", ""))).name
        image_path = dataset_root / "images" / image_name if image_name else None
        if image_path is not None and image_path.is_file():
            media_path = str(image_path)
        evidence.append(
            EvidenceItem(
                evidence_id=f"table:{item}",
                modality="table",
                content=content,
                source_document_id=paper_id,
                media_path=media_path,
                gold=True,
                metadata={"item": item},
            )
        )
    elif modality == "chart":
        content, media_path = _image_content(paper, item, dataset_root)
        image_name = Path(str(row.get("image_path", ""))).name
        image_path = dataset_root / "images" / image_name if image_name else None
        if image_path is not None and image_path.is_file():
            media_path = str(image_path)
        evidence.append(
            EvidenceItem(
                evidence_id=f"figure:{item}",
                modality="figure",
                content=content,
                source_document_id=paper_id,
                media_path=media_path,
                gold=True,
                metadata={"item": item},
            )
        )

    if not any(item_row.gold for item_row in evidence):
        evidence.append(
            EvidenceItem(
                evidence_id="abstract",
                modality="text",
                content=_plain_text(paper.get("abstract") or paper.get("title") or paper_id),
                source_document_id=paper_id,
                gold=True,
                metadata={"fallback": "missing_annotated_evidence"},
            )
        )

    remaining = max(0, max_candidates - len(evidence))
    for section in _distractor_sections(paper, gold_sections, remaining, f"{paper_id}:{row_index}"):
        section_id = str(section.get("section_id", "unknown"))
        evidence.append(
            EvidenceItem(
                evidence_id=f"text:{section_id}",
                modality="text",
                content=_plain_text(
                    f"{section.get('section_name', '')}. {section.get('text', '')}"
                ),
                source_document_id=paper_id,
                gold=False,
                metadata={"section_id": section_id},
            )
        )

    # Preserve the first occurrence if a release row repeats an identifier.
    unique: dict[str, EvidenceItem] = {}
    for item_row in evidence:
        unique.setdefault(item_row.evidence_id, item_row)

    label = "supported" if bool(row["label"]) else "refuted"
    return ClaimPacket(
        claim_id=f"sciver:{split}:{paper_id}:{row.get('request_id', row_index)}:{row_index}",
        claim_text=_plain_text(row["claim"], limit=2000),
        label=label,
        evidence=tuple(list(unique.values())[:max_candidates]),
        split=split,
        dataset="sciver",
        group_id=paper_id,
    )


def prepare_sciver(
    dataset_root: Path,
    output_root: Path,
    validation_fraction: float = 0.2,
    seed: int = 13,
    max_candidates: int = 12,
    max_train: int | None = None,
    max_validation: int | None = None,
    max_test: int | None = None,
) -> dict:
    development_path = dataset_root / "valset.json"
    test_path = dataset_root / "testset.json"
    if not development_path.exists() or not test_path.exists():
        raise FileNotFoundError("SciVer valset.json and testset.json are required")

    development = json.loads(development_path.read_text(encoding="utf-8"))
    test_rows = json.loads(test_path.read_text(encoding="utf-8"))
    official_test_groups = {str(row["paperid"]) for row in test_rows}
    eligible_development = [
        (index, row)
        for index, row in enumerate(development)
        if str(row["paperid"]) not in official_test_groups
    ]
    split_rows: dict[str, list[tuple[int, dict]]] = {"train": [], "validation": [], "test": []}
    for index, row in eligible_development:
        split_rows[group_split(str(row["paperid"]), validation_fraction, seed)].append((index, row))
    split_rows["test"] = list(enumerate(test_rows))

    limits = {"train": max_train, "validation": max_validation, "test": max_test}
    manifest = {
        "dataset": "SciVer",
        "source_revision": "5695ac384d247ae5df2a7b73568dc63d189296b0",
        "source_files": {
            "valset.json": sha256_file(development_path),
            "testset.json": sha256_file(test_path),
        },
        "split_policy": (
            "paper-grouped 80/20 split of official valset after excluding every source-paper "
            "group present in the official testset; official testset otherwise untouched"
        ),
        "development_rows_removed_for_test_group_overlap": len(development) - len(eligible_development),
        "bounded_sampling_policy": "deterministic SHA-256 order independent of release row order",
        "seed": seed,
        "max_candidates": max_candidates,
        "splits": {},
    }

    output_root.mkdir(parents=True, exist_ok=True)
    packets_by_split: dict[str, list[ClaimPacket]] = {}
    for split, indexed_rows in split_rows.items():
        indexed_rows = bounded_rows(
            indexed_rows, limits[split], seed=seed, split=split
        )
        packets: list[ClaimPacket] = []
        skipped = 0
        for row_index, row in indexed_rows:
            paper_path = dataset_root / "papers" / f"{row['paperid']}.json"
            if not paper_path.exists():
                skipped += 1
                continue
            paper = json.loads(paper_path.read_text(encoding="utf-8"))
            packets.append(
                build_sciver_packet(
                    row, paper, dataset_root, split, row_index, max_candidates=max_candidates
                )
            )
        path = output_root / f"{split}.jsonl"
        write_packets(path, packets)
        packets_by_split[split] = packets
        manifest["splits"][split] = {
            "count": len(packets),
            "skipped_missing_paper": skipped,
            "labels": dict(Counter(packet.label for packet in packets)),
            "groups": len({packet.group_id for packet in packets}),
            "sha256": sha256_file(path),
        }

    groups = {
        split: {
            json.loads(line)["group_id"]
            for line in (output_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        }
        for split in ("train", "validation", "test")
    }
    intersections = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    manifest["group_intersections"] = {
        name: len(values) for name, values in intersections.items()
    }
    media_sha_cache: dict[str, str] = {}
    media_dhash_cache: dict[str, int | None] = {}
    quality: dict[str, dict] = {}
    fingerprints: dict[str, dict[str, set]] = {}
    for split in ("train", "validation", "test"):
        quality[split], fingerprints[split] = _split_quality_audit(
            packets_by_split[split], media_sha_cache, media_dhash_cache
        )
    cross_split = {}
    duplicate_details = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        name = f"{left}_{right}"
        exact_content = _exact_duplicate_details(
            fingerprints[left], fingerprints[right], "content"
        )
        exact_media = _exact_duplicate_details(
            fingerprints[left], fingerprints[right], "media"
        )
        exact_perceptual = _exact_duplicate_details(
            fingerprints[left], fingerprints[right], "perceptual"
        )
        near_perceptual, near_truncated = _near_perceptual_details(
            fingerprints[left], fingerprints[right]
        )
        cross_split[name] = {
            "exact_content_fingerprint_overlap": len(exact_content),
            "exact_content_fingerprint_overlap_involving_gold": sum(
                row["involves_gold"] for row in exact_content
            ),
            "exact_media_sha256_overlap": len(exact_media),
            "exact_perceptual_hash_overlap": len(exact_perceptual),
            "near_perceptual_pairs_hamming_le_4": len(near_perceptual),
            "near_perceptual_details_truncated": near_truncated,
        }
        duplicate_details[name] = {
            "exact_content": exact_content,
            "exact_media": exact_media,
            "exact_perceptual": exact_perceptual,
            "near_perceptual_hamming_le_4": near_perceptual,
        }
    all_claim_ids = [
        packet.claim_id for split in packets_by_split.values() for packet in split
    ]
    manifest["data_quality"] = {
        "splits": quality,
        "cross_split_duplicates": cross_split,
        "cross_split_duplicate_details": duplicate_details,
        "duplicate_claim_ids": len(all_claim_ids) - len(set(all_claim_ids)),
        "perceptual_hash": "64-bit difference hash; exact matches and cross-split Hamming distance <= 4 reported",
    }
    no_exact_visual_overlap = all(
        row["exact_media_sha256_overlap"] == 0
        and row["exact_perceptual_hash_overlap"] == 0
        for row in cross_split.values()
    )
    no_gold_content_overlap = all(
        row["exact_content_fingerprint_overlap_involving_gold"] == 0
        for row in cross_split.values()
    )
    manifest["data_quality"]["release_gate"] = {
        "no_group_overlap": not any(intersections.values()),
        "no_exact_visual_overlap": no_exact_visual_overlap,
        "no_gold_content_overlap": no_gold_content_overlap,
        "pass": (
            not any(intersections.values())
            and no_exact_visual_overlap
            and no_gold_content_overlap
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if any(intersections.values()):
        raise RuntimeError(f"paper group leakage across SciVer splits: {manifest['group_intersections']}")
    return manifest
