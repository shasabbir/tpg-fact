from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import json
import os
import random
import time
from typing import TypeVar

from .registry import DATASETS


T = TypeVar("T")


def _hf_token() -> str | None:
    """Return an optional Hugging Face token without logging it."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _retry_hf_call(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int | None = None,
) -> T:
    """Retry rate limits and transient Hub failures while preserving partial files."""
    attempt_limit = attempts or int(os.environ.get("TRACE_SCV_HF_RETRIES", "5"))
    for attempt in range(attempt_limit):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
            retryable = retryable or isinstance(error, (ConnectionError, TimeoutError))
            if retryable and attempt + 1 >= attempt_limit and status == 429:
                raise RuntimeError(
                    "Hugging Face rate limiting persisted after retries. Add a free read-only "
                    "Kaggle secret named HF_TOKEN, restart the session, and run all cells again. "
                    "The token is read without being printed."
                ) from error
            if not retryable or attempt + 1 >= attempt_limit:
                raise
            delay = min(60.0, (2**attempt) + random.random())
            print(
                f"{description} received transient HTTP {status or 'network'}; "
                f"retrying in {delay:.1f}s ({attempt + 2}/{attempt_limit})."
            )
            time.sleep(delay)
    raise RuntimeError(f"unreachable retry state for {description}")


def _download_hf_file(
    *,
    repo_id: str,
    revision: str | None,
    filename: str,
    target: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = _retry_hf_call(
        lambda: hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=target,
            token=_hf_token(),
        ),
        description=f"Hugging Face download for {filename}",
    )
    return Path(downloaded)


def _sciver_release_files(target: Path) -> list[str]:
    """Return the exact paper and image files referenced by the official release rows."""
    development = json.loads((target / "valset.json").read_text(encoding="utf-8"))
    test_rows = json.loads((target / "testset.json").read_text(encoding="utf-8"))
    rows = [*development, *test_rows]
    files = {f"papers/{row['paperid']}.json" for row in rows}
    files.update(
        f"images/{Path(str(row['image_path'])).name}"
        for row in rows
        if row.get("image_path")
    )
    return sorted(files)


def _download_sciver_release(root: Path) -> Path:
    """Download only files referenced by the pinned 3,000-row SciVer release."""
    spec = DATASETS["sciver"]
    target = root / "sciver"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "valset.json", "testset.json"):
        if not (target / name).is_file():
            _download_hf_file(
                repo_id=spec.identifier,
                revision=spec.revision,
                filename=name,
                target=target,
            )

    required = _sciver_release_files(target)
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        from huggingface_hub import snapshot_download

        workers = max(1, int(os.environ.get("TRACE_SCV_HF_WORKERS", "2")))
        _retry_hf_call(
            lambda: snapshot_download(
                repo_id=spec.identifier,
                repo_type="dataset",
                revision=spec.revision,
                local_dir=target,
                allow_patterns=missing,
                max_workers=workers,
                token=_hf_token(),
            ),
            description="pinned SciVer release download",
        )
    still_missing = [name for name in required if not (target / name).is_file()]
    if still_missing:
        raise RuntimeError(
            f"SciVer download incomplete: {len(still_missing)} referenced files are missing; "
            f"first entries: {still_missing[:10]}"
        )
    return target


def download_huggingface_snapshot(
    dataset_key: str,
    root: Path,
    *,
    metadata_only: bool = False,
) -> Path:
    """Download one pinned official dataset snapshot with huggingface_hub."""
    spec = DATASETS[dataset_key]
    if spec.provider != "huggingface":
        raise ValueError(f"{dataset_key} requires its official author download flow")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency guidance
        raise RuntimeError("install trace-scv[data] before downloading") from error

    if dataset_key == "sciver" and not metadata_only:
        return _download_sciver_release(root)

    target = root / dataset_key
    target.mkdir(parents=True, exist_ok=True)
    allow_patterns = None
    if metadata_only and dataset_key == "sciver":
        allow_patterns = ["README.md", "valset.json", "testset.json"]
    elif metadata_only and dataset_key == "musciclaims":
        allow_patterns = ["README.md", "test_set.jsonl"]
    path = _retry_hf_call(
        lambda: snapshot_download(
            repo_id=spec.identifier,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=target,
            allow_patterns=allow_patterns,
            max_workers=max(1, int(os.environ.get("TRACE_SCV_HF_WORKERS", "2"))),
            token=_hf_token(),
        ),
        description=f"pinned {dataset_key} snapshot download",
    )
    return Path(path)


def download_sciver_subset(
    root: Path,
    *,
    max_train: int,
    max_validation: int,
    max_test: int,
    seed: int = 13,
    include_images: bool = True,
) -> Path:
    """Download only the official SciVer files needed by a bounded pilot."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install trace-scv[data] before downloading") from error
    from trace_scv.data.sciver import bounded_rows, group_split

    spec = DATASETS["sciver"]
    target = root / "sciver"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "valset.json", "testset.json"):
        _download_hf_file(
            repo_id=spec.identifier,
            revision=spec.revision,
            filename=name,
            target=target,
        )
    development = json.loads((target / "valset.json").read_text(encoding="utf-8"))
    test_rows = json.loads((target / "testset.json").read_text(encoding="utf-8"))
    test_groups = {str(row["paperid"]) for row in test_rows}
    buckets: dict[str, list[tuple[int, dict]]] = {"train": [], "validation": []}
    for index, row in enumerate(development):
        if str(row["paperid"]) in test_groups:
            continue
        split = group_split(str(row["paperid"]), 0.2, seed)
        buckets[split].append((index, row))
    selected = [
        *[row for _, row in bounded_rows(buckets["train"], max_train, seed=seed, split="train")],
        *[row for _, row in bounded_rows(buckets["validation"], max_validation, seed=seed, split="validation")],
        *[
            row
            for _, row in bounded_rows(
                list(enumerate(test_rows)), max_test, seed=seed, split="test"
            )
        ],
    ]
    files = {f"papers/{row['paperid']}.json" for row in selected}
    if include_images:
        files.update(
            f"images/{Path(str(row['image_path'])).name}"
            for row in selected
            if row.get("image_path")
        )
    for name in sorted(files):
        _download_hf_file(
            repo_id=spec.identifier,
            revision=spec.revision,
            filename=name,
            target=target,
        )
    return target
