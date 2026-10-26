from __future__ import annotations

from pathlib import Path

from .registry import DATASETS


def download_huggingface_snapshot(dataset_key: str, root: Path) -> Path:
    """Download one pinned official dataset snapshot with huggingface_hub."""
    spec = DATASETS[dataset_key]
    if spec.provider != "huggingface":
        raise ValueError(f"{dataset_key} requires its official author download flow")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency guidance
        raise RuntimeError("install trace-scv[data] before downloading") from error

    target = root / dataset_key
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=spec.identifier,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=target,
    )
    return Path(path)
