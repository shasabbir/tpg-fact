from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

import numpy as np

from trace_scv.data.download import _hf_token
from trace_scv.data.io import sha256_file
from trace_scv.data.mocheg import (
    MOCHEG_MIRRORS,
    _audit_downloaded_images,
    _nltk_sentence_splitter,
    assert_mocheg_development_sealed,
    load_mocheg_qrels,
)
from trace_scv.mocheg_features import MochegEncoder


@dataclass(frozen=True, slots=True)
class FullCorpusRetrievalConfig:
    first_stage_text_documents: int = 50
    retained_text_candidates: int = 40
    retained_image_candidates: int = 20
    audit_ranking_depth: int = 60


_HTTP_STATE = threading.local()


def _download_full_media_object(
    *, mirror_key: str, relative: str, media_root: Path, token: str | None, attempts: int = 5
) -> Path:
    """Download one object through a thread-local persistent HTTP session."""
    import requests
    from urllib.parse import quote

    from trace_scv.data.mocheg import _portable_media_path

    mirror = MOCHEG_MIRRORS[mirror_key]
    target = _portable_media_path(media_root, mirror_key, relative)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    session = getattr(_HTTP_STATE, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        session.mount("https://", adapter)
        _HTTP_STATE.session = session
    headers = {"User-Agent": "trace-scv/0.4.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = (
        f"https://huggingface.co/datasets/{mirror.repo_id}/resolve/"
        f"{mirror.revision}/{quote(relative, safe='/')}"
    )
    for attempt in range(attempts):
        try:
            with session.get(url, headers=headers, timeout=(30, 180), stream=True) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size <= 0:
                raise RuntimeError(f"empty media response for {relative}")
            partial.replace(target)
            return target
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", None)
            retryable = status in {None, 429} or (status is not None and 500 <= status < 600)
            if not retryable or attempt + 1 >= attempts:
                partial.unlink(missing_ok=True)
                raise
        time.sleep(min(30.0, 2**attempt))
    raise RuntimeError(f"unreachable download state for {relative}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def acquire_full_development_image_corpus(
    dataset_root: Path, *, max_workers: int = 12
) -> dict[str, object]:
    """Acquire every train/development mirror image without listing the test mirror."""
    from huggingface_hub import HfApi

    assert_mocheg_development_sealed(dataset_root)
    started = time.perf_counter()
    api = HfApi()
    requested: list[tuple[str, str]] = []
    for mirror_key in ("train", "validation"):
        mirror = MOCHEG_MIRRORS[mirror_key]
        requested.extend(
            (mirror_key, relative)
            for relative in api.list_repo_files(
                repo_id=mirror.repo_id,
                repo_type="dataset",
                revision=mirror.revision,
            )
            if relative.startswith("images/")
        )
    requested = sorted(set(requested))
    media_root = dataset_root / "media"
    token = _hf_token()
    resolved: dict[tuple[str, str], Path] = {}
    pending: list[tuple[str, str]] = []
    from trace_scv.data.mocheg import _portable_media_path

    for mirror_key, relative in requested:
        path = _portable_media_path(media_root, mirror_key, relative)
        if path.is_file() and path.stat().st_size > 0:
            resolved[(mirror_key, relative)] = path
        else:
            pending.append((mirror_key, relative))
    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                _download_full_media_object,
                mirror_key=mirror_key,
                relative=relative,
                media_root=media_root,
                token=token,
            ): (mirror_key, relative)
            for mirror_key, relative in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                resolved[key] = future.result()
            except Exception as error:  # retained in the failure artifact
                failures.append(
                    {"mirror": key[0], "source_relative_path": key[1], "error": repr(error)}
                )
            completed += 1
            if completed % 1000 == 0:
                print(f"full-media progress: {completed}/{len(pending)}", flush=True)
    downloaded: list[dict[str, object]] = []
    basename_paths: dict[str, list[str]] = defaultdict(list)
    for (mirror_key, relative), path in sorted(resolved.items()):
        digest = sha256_file(path)
        basename_paths[Path(relative).name].append(f"{mirror_key}:{relative}")
        downloaded.append(
            {
                "mirror": mirror_key,
                "source_relative_path": relative,
                "relative_path": path.relative_to(media_root / mirror_key).as_posix(),
                "basename": Path(relative).name,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    image_audit = _audit_downloaded_images(media_root, downloaded)
    basename_collisions = {
        name: paths for name, paths in basename_paths.items() if len(paths) > 1
    }
    transport_artifacts = {}
    for name in ("full_media_lfs_import.json", "full_media_transport_log.json"):
        path = media_root / name
        if path.is_file():
            transport_artifacts[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_path = Path(__file__).resolve()
    result: dict[str, object] = {
        "profile": "mocheg_full_development_image_corpus",
        "status": "pass" if not failures and not image_audit["unreadable"] else "failed",
        "mirrors": {key: asdict(MOCHEG_MIRRORS[key]) for key in ("train", "validation")},
        "requested_files": len(requested),
        "previously_available_files": len(requested) - len(pending),
        "download_attempts": len(pending),
        "downloaded": downloaded,
        "failures": failures,
        "image_audit": image_audit,
        "basename_collisions": basename_collisions,
        "transport_artifacts": transport_artifacts,
        "implementation": {
            "path": str(implementation_path),
            "sha256": sha256_file(implementation_path),
            "source_commit": source_commit,
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "formal_test_repository_listed": False,
        "formal_test_annotations_accessed": False,
    }
    target = media_root / "full_development_media_manifest.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert_mocheg_development_sealed(dataset_root)
    if result["status"] != "pass":
        raise RuntimeError(f"full development media acquisition failed; inspect {target}")
    return result


def import_full_media_from_lfs_cache(
    dataset_root: Path, cache_root: Path
) -> dict[str, object]:
    """Hard-link pinned LFS objects into portable source-path-derived filenames."""
    import os
    import shutil

    from trace_scv.data.mocheg import _portable_media_path

    assert_mocheg_development_sealed(dataset_root)
    imported = 0
    reused = 0
    missing: list[dict[str, str]] = []
    inventories: dict[str, dict[str, object]] = {}
    for mirror_key in ("train", "validation"):
        mirror = MOCHEG_MIRRORS[mirror_key]
        repository = cache_root / f"{mirror_key}.git"
        completed = subprocess.run(
            ["git", "-C", str(repository), "lfs", "ls-files", "-l", mirror.revision],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        listed = 0
        mirror_imported = 0
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            oid, marker, relative = line.split(" ", 2)
            if marker != "-" or not relative.startswith("images/"):
                continue
            listed += 1
            source = repository / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
            if not source.is_file():
                missing.append({"mirror": mirror_key, "path": relative, "oid": oid})
                continue
            target = _portable_media_path(dataset_root / "media", mirror_key, relative)
            if target.is_file() and target.stat().st_size > 0:
                reused += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            imported += 1
            mirror_imported += 1
        inventories[mirror_key] = {
            "revision": mirror.revision,
            "listed_image_lfs_objects": listed,
            "imported": mirror_imported,
        }
    result = {
        "profile": "mocheg_full_media_lfs_import",
        "status": "pass" if not missing else "incomplete",
        "cache_root": str(cache_root),
        "inventories": inventories,
        "imported": imported,
        "reused": reused,
        "missing_count": len(missing),
        "missing": missing,
        "formal_test_repository_listed": False,
        "formal_test_annotations_accessed": False,
    }
    target = dataset_root / "media" / "full_media_lfs_import.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _claims_from_pilot(processed_root: Path) -> list[dict[str, str]]:
    packets = _read_jsonl(processed_root / "train_pilot_packets.jsonl") + _read_jsonl(
        processed_root / "validation_pilot_packets.jsonl"
    )
    return [
        {"claim_id": str(row["claim_id"]), "claim": str(row["claim"]), "split": str(row["split"])}
        for row in packets
        if any(bool(candidate["available"]) for candidate in row["candidates"])
    ]


def _load_corpus_documents(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    csv.field_size_limit(2**31 - 1)
    contents: list[str] = []
    metadata: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required = {"claim_id", "relevant_document_id", "Origin Document"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("invalid MOCHEG full-corpus header")
        for row_index, row in enumerate(reader):
            raw = str(row.get("Origin Document") or "")
            content = raw.replace("<p>", "").replace("</p>", "")
            contents.append(content)
            metadata.append(
                {
                    "row_index": row_index,
                    "source_claim_id": str(row.get("claim_id") or "").strip(),
                    "source_document_id": str(row.get("relevant_document_id") or "").strip(),
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
    return contents, metadata


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(scores))
    if count == len(scores):
        return np.argsort(-scores, kind="stable")
    unordered = np.argpartition(scores, -count)[-count:]
    return unordered[np.argsort(-scores[unordered], kind="stable")]


def _ranking_metrics(ranking: Sequence[str], gold: set[str]) -> dict[str, float]:
    if not gold:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "recall_at_50": 0.0, "map_at_60": 0.0, "ndcg_at_60": 0.0}
    recalls = {
        f"recall_at_{depth}": len(set(ranking[:depth]) & gold) / len(gold)
        for depth in (5, 10, 50)
    }
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    for rank, evidence_id in enumerate(ranking[:60], start=1):
        if evidence_id in gold:
            hits += 1
            precision_sum += hits / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 60) + 1))
    return {
        **recalls,
        "map_at_60": precision_sum / len(gold),
        "ndcg_at_60": dcg / ideal if ideal else 0.0,
    }


def build_full_corpus_features(
    dataset_root: Path,
    processed_root: Path,
    output_path: Path,
    *,
    encoder: MochegEncoder,
    config: FullCorpusRetrievalConfig = FullCorpusRetrievalConfig(),
) -> dict[str, object]:
    """Retrieve a bounded packet from all released train/development corpus units."""
    assert_mocheg_development_sealed(dataset_root)
    started = time.perf_counter()
    peak_gpu_memory_bytes: int | None = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    media_manifest_path = dataset_root / "media" / "full_development_media_manifest.json"
    media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    if media_manifest["status"] != "pass":
        raise RuntimeError("full development media manifest has not passed")
    claims = _claims_from_pilot(processed_root)
    claim_texts = [row["claim"] for row in claims]
    corpus_path = dataset_root / "metadata" / "corpus" / "Corpus3.csv"
    document_texts, document_metadata = _load_corpus_documents(corpus_path)
    document_embeddings = encoder.encode_bge([*claim_texts, *document_texts])
    claim_bge = document_embeddings[: len(claims)]
    document_bge = document_embeddings[len(claims) :]
    first_stage: list[np.ndarray] = [
        _top_indices(document_bge @ query, config.first_stage_text_documents)
        for query in claim_bge
    ]

    splitter, tokenizer_identity = _nltk_sentence_splitter(dataset_root / "cache" / "nltk_data")
    required_document_indices = sorted({int(index) for rows in first_stage for index in rows})
    sentence_rows: list[dict[str, object]] = []
    sentences_by_document: dict[int, list[int]] = defaultdict(list)
    for document_index in required_document_indices:
        metadata = document_metadata[document_index]
        retained_index = 0
        for sentence in splitter(document_texts[document_index]):
            if sentence == "N/A":
                continue
            row_index = len(sentence_rows)
            evidence_id = (
                f"text:{metadata['source_claim_id']}-{metadata['source_document_id']}-{retained_index}"
            )
            sentence_rows.append(
                {
                    "evidence_id": evidence_id,
                    "content": sentence,
                    "source_corpus_row": document_index,
                    "sentence_index": retained_index,
                }
            )
            sentences_by_document[document_index].append(row_index)
            retained_index += 1
    sentence_embeddings = encoder.encode_bge([str(row["content"]) for row in sentence_rows])

    image_rows = list(media_manifest["downloaded"])
    basename_hashes: dict[str, set[str]] = defaultdict(set)
    for row in image_rows:
        basename_hashes[str(row["basename"])].add(str(row["sha256"]))
    conflicting_basenames = {name for name, hashes in basename_hashes.items() if len(hashes) > 1}
    if conflicting_basenames:
        raise RuntimeError(
            f"image basename collisions with conflicting content: {len(conflicting_basenames)}"
        )
    unique_images: dict[str, dict[str, object]] = {}
    for row in image_rows:
        unique_images.setdefault(str(row["basename"]), row)
    image_names = sorted(unique_images)
    image_paths = [
        str(
            dataset_root
            / "media"
            / str(unique_images[name]["mirror"])
            / str(unique_images[name]["relative_path"])
        )
        for name in image_names
    ]
    claim_siglip, image_embeddings = encoder.encode_siglip(claim_texts, image_paths)

    text_rankings: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
    image_rankings: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
    for claim_index, claim in enumerate(claims):
        candidate_indices = sorted(
            {
                sentence_index
                for document_index in first_stage[claim_index]
                for sentence_index in sentences_by_document[int(document_index)]
            }
        )
        scores = sentence_embeddings[candidate_indices] @ claim_bge[claim_index]
        order = _top_indices(scores, config.audit_ranking_depth)
        deduplicated: list[tuple[str, float, str]] = []
        seen: set[str] = set()
        for local_index in order:
            row = sentence_rows[candidate_indices[int(local_index)]]
            evidence_id = str(row["evidence_id"])
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            deduplicated.append((evidence_id, float(scores[int(local_index)]), str(row["content"])))
        text_rankings[(claim["split"], claim["claim_id"])] = deduplicated

        image_scores = image_embeddings @ claim_siglip[claim_index]
        image_order = _top_indices(image_scores, config.audit_ranking_depth)
        image_rankings[(claim["split"], claim["claim_id"])] = [
            (f"image:{image_names[int(index)]}", float(image_scores[int(index)]), image_paths[int(index)])
            for index in image_order
        ]

    split_folders = {"train": "train", "validation": "val"}
    full_gold: dict[str, set[str]] = defaultdict(set)
    for split, folder in split_folders.items():
        claim_ids = {row["claim_id"] for row in claims if row["split"] == split}
        for modality, filename in (
            ("text", "text_evidence_qrels_sentence_level.csv"),
            ("image", "img_evidence_qrels.csv"),
        ):
            for qrel in load_mocheg_qrels(dataset_root / "metadata" / folder / filename):
                if qrel.claim_id in claim_ids and qrel.relevant:
                    full_gold[f"{split}:{qrel.claim_id}"].add(f"{modality}:{qrel.document_id}")

    features: list[list[float]] = []
    labels: list[int] = []
    records: list[dict[str, object]] = []
    ranking_records: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    for claim in claims:
        key = (claim["split"], claim["claim_id"])
        gold = full_gold[f"{claim['split']}:{claim['claim_id']}"]
        text_ranking = text_rankings[key]
        image_ranking = image_rankings[key]
        retained = [
            *( (evidence_id, score, "text", content) for evidence_id, score, content in text_ranking[: config.retained_text_candidates] ),
            *( (evidence_id, score, "image", path) for evidence_id, score, path in image_ranking[: config.retained_image_candidates] ),
        ]
        available_count = len(retained)
        for evidence_id, similarity, modality, payload in retained:
            token_count = len(payload.split()) if modality == "text" else 0
            features.append(
                [
                    similarity,
                    similarity * similarity,
                    float(modality == "text"),
                    float(modality == "image"),
                    math.log1p(token_count) / 10.0,
                    math.log1p(len(claim["claim"].split())) / 10.0,
                    math.log1p(available_count) / 10.0,
                ]
            )
            labels.append(int(evidence_id in gold))
            records.append(
                {
                    "claim_id": claim["claim_id"],
                    "evidence_id": evidence_id,
                    "modality": modality,
                    "split": claim["split"],
                    "gold": evidence_id in gold,
                }
            )
        for modality, ranking in (("text", text_ranking), ("image", image_ranking)):
            ids = [row[0] for row in ranking]
            modality_gold = {item for item in gold if item.startswith(f"{modality}:")}
            metrics = _ranking_metrics(ids, modality_gold)
            retrieval_rows.append(
                {"split": claim["split"], "claim_id": claim["claim_id"], "modality": modality, "gold_count": len(modality_gold), **metrics}
            )
            ranking_records.append(
                {
                    "split": claim["split"],
                    "claim_id": claim["claim_id"],
                    "modality": modality,
                    "ranking": [
                        {"rank": rank, "evidence_id": row[0], "score": row[1]}
                        for rank, row in enumerate(ranking, start=1)
                    ],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, features=np.asarray(features, dtype=np.float32), gold=np.asarray(labels, dtype=np.int8))
    sidecar_path = output_path.with_suffix(".json")
    sidecar = {
        "feature_names": [
            "semantic_similarity", "semantic_similarity_squared", "is_text", "is_image",
            "log_text_token_count", "log_claim_token_count", "log_available_candidate_count",
        ],
        "encoder": encoder.identity(),
        "retrieval_config": asdict(config),
        "records": records,
        "full_gold_evidence_ids_by_claim": {
            claim_id.split(":", 1)[1]: sorted(values) for claim_id, values in full_gold.items()
        },
        "formal_test_annotations_accessed": False,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    rankings_path = output_path.with_suffix(".rankings.jsonl")
    with rankings_path.open("w", encoding="utf-8") as handle:
        for row in ranking_records:
            handle.write(json.dumps(row) + "\n")
    eligible = [row for row in retrieval_rows if row["gold_count"]]
    metrics = {
        modality: {
            key: fmean(float(row[key]) for row in eligible if row["modality"] == modality)
            for key in ("recall_at_5", "recall_at_10", "recall_at_50", "map_at_60", "ndcg_at_60")
        }
        for modality in ("text", "image")
    }
    if torch is not None and torch.cuda.is_available():
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    implementation_path = Path(__file__).resolve()
    feature_implementation_path = implementation_path.parent / "mocheg_features.py"
    data_implementation_path = implementation_path.parent / "data" / "mocheg.py"
    selector_implementation_path = implementation_path.parent / "selection" / "mocheg_study.py"
    development_config_path = Path.cwd() / "configs" / "mocheg_development.yaml"
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "profile": "mocheg_full_corpus_label_blind_retrieval",
        "status": "pass",
        "retrieval_config": asdict(config),
        "claims": len(claims),
        "corpus_documents": len(document_texts),
        "first_stage_unique_documents": len(required_document_indices),
        "reranked_sentences": len(sentence_rows),
        "image_files": len(image_rows),
        "unique_image_basenames": len(image_names),
        "feature_rows": len(records),
        "gold_feature_rows": sum(labels),
        "rows": len(records),
        "claims": len(claims),
        "features_sha256": sha256_file(output_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "scientific_results_allowed": True,
        "retrieval_metrics": metrics,
        "tokenizer": tokenizer_identity,
        "artifacts": {
            "features": {"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": sha256_file(output_path)},
            "sidecar": {"path": str(sidecar_path), "bytes": sidecar_path.stat().st_size, "sha256": sha256_file(sidecar_path)},
            "rankings": {"path": str(rankings_path), "bytes": rankings_path.stat().st_size, "sha256": sha256_file(rankings_path)},
            "corpus": {"path": str(corpus_path), "bytes": corpus_path.stat().st_size, "sha256": sha256_file(corpus_path)},
            "media_manifest": {"path": str(media_manifest_path), "bytes": media_manifest_path.stat().st_size, "sha256": sha256_file(media_manifest_path)},
        },
        "encoder": encoder.identity(),
        "implementation_files": {
            str(path): sha256_file(path)
            for path in (
                implementation_path,
                feature_implementation_path,
                data_implementation_path,
                selector_implementation_path,
            )
        },
        "development_config": {
            "path": str(development_config_path.resolve()),
            "sha256": sha256_file(development_config_path),
        },
        "source_commit": source_commit,
        "packet_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                processed_root / "train_pilot_packets.jsonl",
                processed_root / "validation_pilot_packets.jsonl",
            )
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch is not None and torch.cuda.is_available()
                else None
            ),
        },
        "claim_association_used_as_model_input": False,
        "gold_loaded_after_rankings_computed": True,
        "formal_test_repository_listed": False,
        "formal_test_annotations_accessed": False,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert_mocheg_development_sealed(dataset_root)
    return manifest
